#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


C_SOUND = 343.0  # m/s (approx at 20°C)


@dataclass
class GccPhatResult:
    tau: float          # seconds
    peak: float         # peak value
    peak_ratio: float   # peak / second_peak (rough confidence)


def parse_int_list(s: str) -> List[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def read_u16_pcm_4ch(path: str, channels: int = 4) -> np.ndarray:
    """
    Read interleaved unsigned 16-bit PCM (little endian) with fixed channels.
    Returns float32 array shape [num_samples, channels] in [-1, 1).
    """
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size % channels != 0:
        raise ValueError(f"File length {raw.size} not divisible by {channels} channels.")

    frames = raw.reshape(-1, channels)

    # Convert unsigned 16-bit to float [-1, 1)
    x = frames.astype(np.float32)
    x = (x - 32768.0) / 32768.0
    return x


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """
    x: [N] -> frames: [T, frame_len]
    """
    N = x.shape[0]
    if N < frame_len:
        return np.zeros((0, frame_len), dtype=x.dtype)
    T = 1 + (N - frame_len) // hop
    idx = (np.arange(frame_len)[None, :] + hop * np.arange(T)[:, None]).astype(np.int64)
    return x[idx]


def gcc_phat(sig: np.ndarray,
             refsig: np.ndarray,
             fs: int,
             max_tau: Optional[float] = None,
             interp: int = 16,
             fmin: float = 300.0,
             fmax: float = 3000.0) -> GccPhatResult:
    """
    GCC-PHAT time delay estimate.
    - Uses frequency band [fmin, fmax] to reduce low-frequency hum / high-frequency noise.
    - interp: oversampling factor in FFT domain for sub-sample peak.
    """
    if sig.shape != refsig.shape:
        raise ValueError("sig and refsig must have same shape")

    n = sig.shape[0]
    nfft = 1
    while nfft < n * interp:
        nfft *= 2

    SIG = np.fft.rfft(sig, n=nfft)
    REFSIG = np.fft.rfft(refsig, n=nfft)

    R = SIG * np.conj(REFSIG)

    # PHAT weighting
    denom = np.abs(R)
    denom[denom < 1e-12] = 1e-12
    R /= denom

    # Band-limit in frequency domain
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    R *= band.astype(np.float32)

    cc = np.fft.irfft(R, n=nfft)

    # shift for circular correlation
    max_shift = nfft // 2
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    if max_tau is not None:
        max_shift = min(int(round(max_tau * fs * interp)), max_shift)
        mid = len(cc) // 2
        cc = cc[mid - max_shift: mid + max_shift + 1]

    # Find peak and second peak (confidence)
    abscc = np.abs(cc)
    peak_idx = int(np.argmax(abscc))
    peak_val = float(abscc[peak_idx])

    # crude second peak: suppress neighborhood around peak
    sup = abscc.copy()
    guard = max(2, int(0.002 * fs * interp))  # ~2ms guard
    lo = max(0, peak_idx - guard)
    hi = min(sup.size, peak_idx + guard + 1)
    sup[lo:hi] = 0.0
    second = float(np.max(sup)) if sup.size else 0.0
    peak_ratio = peak_val / (second + 1e-12)

    # Convert index to time delay
    # center of cc is at len(cc)//2
    mid = len(cc) // 2
    shift = peak_idx - mid
    tau = shift / float(fs * interp)

    return GccPhatResult(tau=tau, peak=peak_val, peak_ratio=peak_ratio)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cutoff = 0.5 * np.sum(w)
    idx = int(np.searchsorted(cw, cutoff))
    return float(v[min(idx, len(v) - 1)])


def estimate_angle_two_mics(taus: np.ndarray, weights: np.ndarray, mic_distance: float) -> Tuple[float, float]:
    """
    Returns (angle_deg, tau_est)
    Convention: tau = t(micB) - t(micA) where micA is first selected mic, micB is second.
    For linear array, tau = (d * sin(theta)) / c where theta=0° is broadside (in front), +theta towards micB.
    """
    tau_est = weighted_median(taus, weights) if len(taus) else 0.0

    # Clamp physically possible
    tau_max = mic_distance / C_SOUND
    tau_est = float(np.clip(tau_est, -tau_max, tau_max))

    s = (tau_est * C_SOUND) / mic_distance
    s = float(np.clip(s, -1.0, 1.0))
    theta = math.degrees(math.asin(s))
    return theta, tau_est


def fit_angle_multimic(taus_by_pair: List[Tuple[Tuple[int, int], float, float]],
                       mic_positions: np.ndarray) -> Tuple[float, float]:
    """
    Fit a single theta assuming all mics lie on a line with known positions x_i (meters).
    For pair (i,j): tau_ij ≈ ( (x_j - x_i) * sin(theta) ) / c
    Solve for sin(theta) using weighted least squares.
    taus_by_pair: [ ((i,j), tau, weight), ... ] where i/j are indices into mic_positions (0..M-1)
    """
    if len(taus_by_pair) == 0:
        return 0.0, 0.0

    A = []
    b = []
    w = []
    for (i, j), tau, weight in taus_by_pair:
        dx = mic_positions[j] - mic_positions[i]
        if abs(dx) < 1e-6:
            continue
        A.append([dx / C_SOUND])
        b.append([tau])
        w.append(weight)

    if len(A) == 0:
        return 0.0, 0.0

    A = np.array(A, dtype=np.float64)  # [P,1]
    b = np.array(b, dtype=np.float64)  # [P,1]
    W = np.diag(np.array(w, dtype=np.float64) + 1e-12)

    # Weighted least squares for s = sin(theta):
    # b ≈ A * s
    # s = (A^T W A)^-1 A^T W b
    ATA = A.T @ W @ A
    ATb = A.T @ W @ b
    s = float((np.linalg.pinv(ATA) @ ATb).squeeze())

    s = float(np.clip(s, -1.0, 1.0))
    theta = math.degrees(math.asin(s))
    return theta, s


def main():
    ap = argparse.ArgumentParser(
        description="GCC-PHAT DOA for 4ch u16 PCM (ch1=spk ref, ch2-4=mics). Outputs a final angle in degrees."
    )
    ap.add_argument("--file", "-f", required=True, help="Path to 4-channel u16 PCM file.")
    ap.add_argument("--sr", type=int, required=True, help="Sample rate (Hz), e.g., 16000.")
    ap.add_argument("--mics", required=True,
                    help="Mic channels to use, 1-indexed channel numbers in the file, e.g. '2,3' or '2,3,4'.")
    ap.add_argument("--spk_ch", type=int, default=1,
                    help="Speaker reference channel index (1-indexed). Default: 1.")
    ap.add_argument("--mic_distance", type=float, default=0.06,
                    help="Distance between two mics (meters) when using exactly 2 mics. Default 0.06.")
    ap.add_argument("--mic_positions", type=str, default="",
                    help="Mic positions along a line in meters for selected mics, comma-separated. "
                         "Length must match number of selected mics. Example: '0,0.06,0.12'.")
    ap.add_argument("--mic_spacing", type=float, default=0.06,
                    help="If mic_positions not provided and >=3 mics selected, assume equal spacing. Default 0.06.")
    ap.add_argument("--frame_ms", type=float, default=20.0, help="Frame length in ms. Default 20.")
    ap.add_argument("--hop_ms", type=float, default=10.0, help="Hop length in ms. Default 10.")
    ap.add_argument("--interp", type=int, default=16, help="GCC-PHAT interpolation factor. Default 16.")
    ap.add_argument("--fmin", type=float, default=300.0, help="Bandpass min freq for GCC-PHAT. Default 300.")
    ap.add_argument("--fmax", type=float, default=3000.0, help="Bandpass max freq for GCC-PHAT. Default 3000.")
    ap.add_argument("--vad_ratio", type=float, default=1.5,
                    help="VAD energy threshold ratio over median energy. Default 1.5.")
    ap.add_argument("--spk_gate_ratio", type=float, default=2.5,
                    help="Skip frames if spk RMS > spk_gate_ratio * median_spk_rms. Default 2.5.")
    ap.add_argument("--min_peak_ratio", type=float, default=2.0,
                    help="Discard frame if GCC peak_ratio < this. Default 2.0.")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="Optional limit of frames to process (0 = no limit).")
    args = ap.parse_args()

    fs = args.sr
    data = read_u16_pcm_4ch(args.file, channels=4)  # [N,4]

    # Parse channels (1-indexed)
    mic_ch = parse_int_list(args.mics)
    if len(mic_ch) < 2:
        print("ERROR: need at least 2 mic channels.", file=sys.stderr)
        sys.exit(2)

    if any(ch < 1 or ch > 4 for ch in mic_ch):
        print("ERROR: mic channels must be in [1..4].", file=sys.stderr)
        sys.exit(2)

    spk_ch = args.spk_ch
    if spk_ch < 1 or spk_ch > 4:
        print("ERROR: spk_ch must be in [1..4].", file=sys.stderr)
        sys.exit(2)

    # Extract signals
    spk = data[:, spk_ch - 1]
    mics = [data[:, ch - 1] for ch in mic_ch]

    frame_len = int(round(args.frame_ms * fs / 1000.0))
    hop = int(round(args.hop_ms * fs / 1000.0))
    if frame_len <= 0 or hop <= 0:
        print("ERROR: invalid frame/hop.", file=sys.stderr)
        sys.exit(2)

    spk_frames = frame_signal(spk, frame_len, hop)
    mic_frames = [frame_signal(m, frame_len, hop) for m in mics]

    T = spk_frames.shape[0]
    if T == 0:
        print("ERROR: audio too short for given frame length.", file=sys.stderr)
        sys.exit(2)

    if args.max_frames and T > args.max_frames:
        T = args.max_frames
        spk_frames = spk_frames[:T]
        mic_frames = [mf[:T] for mf in mic_frames]

    # Energy / gating
    spk_rms = np.sqrt(np.mean(spk_frames ** 2, axis=1) + 1e-12)
    spk_med = float(np.median(spk_rms) + 1e-12)
    spk_gate_th = args.spk_gate_ratio * spk_med

    # VAD based on mic sum energy
    mic_sum = np.zeros((T, frame_len), dtype=np.float32)
    for mf in mic_frames:
        mic_sum += mf[:T]
    mic_rms = np.sqrt(np.mean(mic_sum ** 2, axis=1) + 1e-12)
    mic_med = float(np.median(mic_rms) + 1e-12)
    vad_th = args.vad_ratio * mic_med

    # Determine max_tau for each pair based on geometry
    def max_tau_from_d(d: float) -> float:
        return abs(d) / C_SOUND

    # Positions handling for >=3 mics
    M = len(mics)
    if args.mic_positions.strip():
        pos = np.array([float(x.strip()) for x in args.mic_positions.split(",") if x.strip() != ""], dtype=np.float64)
        if pos.size != M:
            print(f"ERROR: mic_positions length {pos.size} must match number of selected mics {M}.", file=sys.stderr)
            sys.exit(2)
        mic_positions = pos
    else:
        # assume equally spaced on a line
        mic_positions = np.arange(M, dtype=np.float64) * float(args.mic_spacing)

    # Collect per-frame per-pair tau estimates
    taus = []
    weights = []

    # For multi-mic fitting: store pairwise taus
    taus_by_pair: List[Tuple[Tuple[int, int], float, float]] = []

    # Iterate frames
    for t in range(T):
        # Skip if speaker reference too strong
        if spk_rms[t] > spk_gate_th:
            continue
        # Skip non-speech
        if mic_rms[t] < vad_th:
            continue

        # For each pair
        for i in range(M):
            for j in range(i + 1, M):
                si = mic_frames[i][t]
                sj = mic_frames[j][t]

                d = float(mic_positions[j] - mic_positions[i])
                max_tau = max_tau_from_d(d)

                res = gcc_phat(si, sj, fs=fs, max_tau=max_tau, interp=args.interp, fmin=args.fmin, fmax=args.fmax)
                if res.peak_ratio < args.min_peak_ratio:
                    continue

                # weight: combine peak & peak_ratio (both help)
                w = float(res.peak * min(res.peak_ratio, 10.0))
                taus_by_pair.append(((i, j), res.tau, w))

                # If exactly 2 mics, collect directly
                if M == 2:
                    taus.append(res.tau)
                    weights.append(w)

    if M == 2:
        if len(taus) < 5:
            print("WARN: Too few valid speech frames for a confident estimate; result may be unreliable.", file=sys.stderr)
        taus_np = np.array(taus, dtype=np.float64)
        w_np = np.array(weights, dtype=np.float64) if len(weights) else np.ones_like(taus_np)
        angle_deg, tau_est = estimate_angle_two_mics(taus_np, w_np, mic_distance=float(args.mic_distance))
        print(f"{angle_deg:.2f}")
        return

    # >=3 mics: fit a single angle under colinear assumption
    if len(taus_by_pair) < 10:
        print("WARN: Too few valid pairwise estimates; result may be unreliable.", file=sys.stderr)

    angle_deg, s = fit_angle_multimic(taus_by_pair, mic_positions=mic_positions)
    print(f"{angle_deg:.2f}")


if __name__ == "__main__":
    main()
