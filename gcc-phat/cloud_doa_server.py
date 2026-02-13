#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import socket
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


MAGIC = b"DOA1"
HEADER_FMT = "!4s6I"  # magic, sr, ch, mic_left, mic_right, frames, payload_bytes
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def recv_exact(conn: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        data = conn.recv(remaining)
        if not data:
            raise ConnectionError("peer closed while receiving")
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def parse_header(buf: bytes):
    magic, sr, ch, mic_left, mic_right, frames, payload_bytes = struct.unpack(HEADER_FMT, buf)
    if magic != MAGIC:
        raise ValueError(f"invalid magic: {magic!r}")
    return {
        "sr": sr,
        "ch": ch,
        "mic_left": mic_left,
        "mic_right": mic_right,
        "frames": frames,
        "payload_bytes": payload_bytes,
    }


def run_gcc_phat(gcc_script: Path, payload: bytes, sr: int, ch: int, mic_left: int, mic_right: int) -> str:
    if ch <= 0:
        raise ValueError("channel count must be > 0")
    if mic_left >= ch or mic_right >= ch:
        raise ValueError(f"mic index out of range: left={mic_left}, right={mic_right}, ch={ch}")

    # 固件发送的是小端 int16；gcc-phat.py 当前读取 u16，这里做一次转换适配
    pcm_i16 = np.frombuffer(payload, dtype="<i2")
    if pcm_i16.size % ch != 0:
        raise ValueError(f"payload samples {pcm_i16.size} not divisible by channels {ch}")
    pcm_u16 = (pcm_i16.astype(np.int32) + 32768).astype(np.uint16)

    with tempfile.NamedTemporaryFile(prefix="doa_", suffix=".pcm", delete=False) as fp:
        tmp = Path(fp.name)
        pcm_u16.tofile(fp)

    # gcc-phat.py 使用 1-based 通道编号
    mics = f"{mic_left + 1},{mic_right + 1}"
    cmd = [
        sys.executable,
        str(gcc_script),
        "--file",
        str(tmp),
        "--sr",
        str(sr),
        "--mics",
        mics,
        "--spk_ch",
        "1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or f"gcc-phat exited {proc.returncode}")
        return proc.stdout.strip()
    finally:
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Receive PCM from ESP and run gcc-phat DOA parser.")
    ap.add_argument("--host", default="0.0.0.0", help="Listen host, default 0.0.0.0")
    ap.add_argument("--port", type=int, default=5001, help="Listen port, default 5001")
    ap.add_argument(
        "--gcc_script",
        default=str(Path(__file__).resolve().parent / "gcc-phat.py"),
        help="Path to gcc-phat.py",
    )
    args = ap.parse_args()

    gcc_script = Path(args.gcc_script).resolve()
    if not gcc_script.exists():
        raise FileNotFoundError(f"gcc-phat script not found: {gcc_script}")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(5)
    print(f"[server] listening on {args.host}:{args.port}, gcc={gcc_script}")

    while True:
        conn, addr = server.accept()
        with conn:
            peer = f"{addr[0]}:{addr[1]}"
            try:
                header_buf = recv_exact(conn, HEADER_SIZE)
                hdr = parse_header(header_buf)
                payload = recv_exact(conn, hdr["payload_bytes"])
                angle = run_gcc_phat(
                    gcc_script=gcc_script,
                    payload=payload,
                    sr=hdr["sr"],
                    ch=hdr["ch"],
                    mic_left=hdr["mic_left"],
                    mic_right=hdr["mic_right"],
                )
                print(
                    f"[doa] peer={peer} angle={angle} "
                    f"sr={hdr['sr']} ch={hdr['ch']} frames={hdr['frames']} mics=({hdr['mic_left']},{hdr['mic_right']})"
                )
            except Exception as e:
                print(f"[error] peer={peer} {e}")


if __name__ == "__main__":
    main()
