/*
   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"
#pragma GCC diagnostic ignored "-Wint-conversion"
#pragma GCC diagnostic ignored "-Wreturn-mismatch"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_wn_iface.h"
#include "esp_wn_models.h"
#include "dl_lib_coefgetter_if.h"
#include "esp_afe_sr_iface.h"
#include "esp_afe_sr_models.h"
#include "esp_mn_iface.h"
#include "esp_mn_models.h"
#include "esp_board_init.h"
#include "model_path.h"
#include "ringbuf.h"
#include "esp_nsn_models.h"
#include "model_path.h"
#include "esp_doa.h"

// ---------- 行为配置区 ----------
// DOA 角度分辨率（度）：越小越细，但计算开销可能增加
#define DOA_RESOLUTION_DEG           20.0f
// 两个 DOA 麦克风之间的物理间距（米）
#define DOA_MIC_DISTANCE_M           0.065f
// 唤醒后回放窗口长度（毫秒）
#define WAKEWORD_REPLAY_MS           900
// 唤醒后用于 DOA 估算的窗口长度（毫秒）
#define WAKEWORD_DOA_WINDOW_MS       500
// 两次唤醒触发之间的最小间隔（毫秒），用于防抖
#define WAKE_REPLAY_COOLDOWN_MS      1500
// 本示例采样率
#define SAMPLE_RATE_HZ               16000
// 回放通道索引（基于 raw feed 多通道顺序，RMMM 时: 0=R, 1=M0, 2=M1, 3=M2）
#define RAW_PLAYBACK_FEED_CH_INDEX    3
// 每次唤醒时是否打印 raw 各通道电平统计（RMS/Peak），用于排查“某通道声音小”
#define DEBUG_PRINT_WAKE_CHANNEL_LEVEL 1

static const esp_afe_sr_iface_t *afe_handle = NULL;
static volatile int task_flag = 0;
// AFE 每次 feed/fetch 的帧长（sample 点数）
static int g_feed_chunk_samples = 0;
// 原始 feed 数据的通道数（例如 4）
static int g_feed_channel_count = 0;
// 用于 DOA 的左右麦通道索引（来自输入格式字符串中 'M' 的位置）
static int g_mic_idx_left = 0;
static int g_mic_idx_right = 1;
// 回放使用的通道索引（可宏配置）
static int g_playback_ch_idx = 0;

/**
 * 缓存最近一段“原始多通道”数据，用于：
 * 1) 唤醒后从指定通道回放
 * 2) 唤醒后提取左右麦做 DOA
 * 数据格式为交错存储: [ch0, ch1, ..., chN-1, ch0, ch1, ...]
 */
static int16_t *g_raw_mic_ring = NULL;
static int g_raw_mic_ring_len = 0;
static int g_raw_mic_ring_write_idx = 0;
static bool g_raw_mic_ring_full = false;
static portMUX_TYPE g_raw_mic_ring_lock = portMUX_INITIALIZER_UNLOCKED;

/**
 * @brief 向“原始多通道环形缓冲”写入一帧交错数据。
 *
 * feed_task 持续写入，detect_task 可能在任意时刻读取，因此这里必须用临界区，
 * 保证 “数据 + 写指针 + 满标记” 一起更新，避免读线程拿到不一致快照。
 */
static void raw_mic_ring_push(const int16_t *interleaved, int samples, int channels)
{
    if (!g_raw_mic_ring || !interleaved || samples <= 0 || channels <= 0) {
        return;
    }

    int values = samples * channels;
    taskENTER_CRITICAL(&g_raw_mic_ring_lock);
    for (int i = 0; i < values; i++) {
        g_raw_mic_ring[g_raw_mic_ring_write_idx] = interleaved[i];
        g_raw_mic_ring_write_idx = (g_raw_mic_ring_write_idx + 1) % g_raw_mic_ring_len;
        if (g_raw_mic_ring_write_idx == 0) {
            g_raw_mic_ring_full = true;
        }
    }
    taskEXIT_CRITICAL(&g_raw_mic_ring_lock);
}

/**
 * @brief 取出最近 window_samples 帧“原始多通道交错数据”。
 *
 * 输出格式不变：`[ch0,ch1,...,chN-1, ch0,ch1,...]`
 * 返回值是实际拷贝的“帧数”（每帧包含 channels 个 int16_t）。
 */
static int raw_mic_ring_copy_recent_interleaved(int16_t *out_interleaved, int window_samples, int channels)
{
    if (!g_raw_mic_ring || !out_interleaved || window_samples <= 0 || channels <= 0) {
        return 0;
    }

    // 目标拷贝长度（按 int16_t 计数）
    int want_values = window_samples * channels;
    taskENTER_CRITICAL(&g_raw_mic_ring_lock);
    int valid_values = g_raw_mic_ring_full ? g_raw_mic_ring_len : g_raw_mic_ring_write_idx;
    int copy_values = (want_values < valid_values) ? want_values : valid_values;
    // 保证拷贝长度是完整帧（frame = channels 个值），避免拆帧
    copy_values -= (copy_values % channels);
    if (copy_values <= 0) {
        taskEXIT_CRITICAL(&g_raw_mic_ring_lock);
        return 0;
    }

    int start_idx = g_raw_mic_ring_write_idx - copy_values;
    if (start_idx < 0) {
        start_idx += g_raw_mic_ring_len;
    }

    // 环形缓冲可能跨尾部回绕，分两段 memcpy
    int first_part = g_raw_mic_ring_len - start_idx;
    if (first_part > copy_values) {
        first_part = copy_values;
    }
    memcpy(out_interleaved, g_raw_mic_ring + start_idx, first_part * sizeof(int16_t));
    if (copy_values > first_part) {
        memcpy(out_interleaved + first_part, g_raw_mic_ring, (copy_values - first_part) * sizeof(int16_t));
    }
    taskEXIT_CRITICAL(&g_raw_mic_ring_lock);

    return copy_values / channels;
}

/**
 * @brief 从 raw ring 中提取最近 window_samples 帧里的某一个通道，输出为单声道线性数组。
 *
 * `channel_idx` 为 feed 原始通道索引（0 ~ feed_channel-1）。
 * 返回值是输出的样本数（每个样本一个 int16_t）。
 */
static int raw_mic_ring_copy_recent_channel(int16_t *out_channel, int window_samples, int channel_idx)
{
    if (!out_channel || window_samples <= 0 || channel_idx < 0 || channel_idx >= g_feed_channel_count) {
        return 0;
    }

    // 先拿一份多通道快照，再从中抽取目标通道，避免和写线程竞争
    int16_t *snapshot = malloc(window_samples * g_feed_channel_count * sizeof(int16_t));
    if (!snapshot) {
        return 0;
    }

    int copied_frames = raw_mic_ring_copy_recent_interleaved(snapshot, window_samples, g_feed_channel_count);
    for (int i = 0; i < copied_frames; i++) {
        out_channel[i] = snapshot[i * g_feed_channel_count + channel_idx];
    }

    free(snapshot);
    return copied_frames;
}

/**
 * @brief 调试辅助：打印最近窗口内每个通道的 RMS 与峰值。
 *
 * 用途：
 * - 快速判断 RAW_PLAYBACK_FEED_CH_INDEX 选中的通道是不是天然电平偏小
 * - 辅助确认当前板卡上的 raw 通道映射（哪路是参考，哪路是麦）
 */
static void debug_print_recent_channel_levels(int window_samples)
{
    if (g_feed_channel_count <= 0 || window_samples <= 0) {
        return;
    }

    int16_t *snapshot = malloc(window_samples * g_feed_channel_count * sizeof(int16_t));
    if (!snapshot) {
        return;
    }

    int frames = raw_mic_ring_copy_recent_interleaved(snapshot, window_samples, g_feed_channel_count);
    if (frames <= 0) {
        free(snapshot);
        return;
    }

    char *fmt = esp_get_input_format();
    printf("raw level stats (frames=%d):\n", frames);
    for (int ch = 0; ch < g_feed_channel_count; ch++) {
        double sum_sq = 0.0;
        int peak = 0;
        for (int i = 0; i < frames; i++) {
            int v = snapshot[i * g_feed_channel_count + ch];
            int a = (v >= 0) ? v : -v;
            if (a > peak) {
                peak = a;
            }
            sum_sq += (double)v * (double)v;
        }
        int rms = (int)sqrt(sum_sq / frames);
        printf("  ch%d('%c'): rms=%d peak=%d%s\n",
               ch,
               fmt[ch],
               rms,
               peak,
               (ch == g_playback_ch_idx) ? "  <-- playback" : "");
    }

    free(snapshot);
}

/**
 * 使用最近的双麦窗口估算一次 DOA（对多个 chunk 结果取均值）。
 */
static float estimate_wake_doa_from_recent(int window_samples, int left_ch, int right_ch)
{
    if (g_feed_chunk_samples <= 0 || window_samples < g_feed_chunk_samples || !g_raw_mic_ring || g_feed_channel_count <= 0) {
        return 0.0f;
    }
    if (left_ch < 0 || right_ch < 0 || left_ch >= g_feed_channel_count || right_ch >= g_feed_channel_count || left_ch == right_ch) {
        return 0.0f;
    }

    // snapshot 保存最近窗口的原始多通道数据
    int16_t *snapshot = malloc(window_samples * g_feed_channel_count * sizeof(int16_t));
    int16_t *left = malloc(g_feed_chunk_samples * sizeof(int16_t));
    int16_t *right = malloc(g_feed_chunk_samples * sizeof(int16_t));
    if (!snapshot || !left || !right) {
        free(snapshot);
        free(left);
        free(right);
        return 0.0f;
    }

    int copied_samples = raw_mic_ring_copy_recent_interleaved(snapshot, window_samples, g_feed_channel_count);
    if (copied_samples < g_feed_chunk_samples) {
        free(snapshot);
        free(left);
        free(right);
        return 0.0f;
    }

    doa_handle_t *doa = esp_doa_create(SAMPLE_RATE_HZ, DOA_RESOLUTION_DEG, DOA_MIC_DISTANCE_M, g_feed_chunk_samples);
    if (!doa) {
        free(snapshot);
        free(left);
        free(right);
        return 0.0f;
    }

    float doa_sum = 0.0f;
    int doa_count = 0;
    // DOA 算法按 chunk 处理：每个 chunk 算一次角度，最后取均值更稳定
    int total_chunks = copied_samples / g_feed_chunk_samples;
    for (int c = 0; c < total_chunks; c++) {
        int base = c * g_feed_chunk_samples;
        for (int i = 0; i < g_feed_chunk_samples; i++) {
            int frame_idx = base + i;
            left[i] = snapshot[frame_idx * g_feed_channel_count + left_ch];
            right[i] = snapshot[frame_idx * g_feed_channel_count + right_ch];
        }
        doa_sum += esp_doa_process(doa, left, right);
        doa_count++;
    }

    esp_doa_destroy(doa);
    free(snapshot);
    free(left);
    free(right);

    if (doa_count == 0) {
        return 0.0f;
    }
    return doa_sum / doa_count;
}

void feed_Task(void *arg)
{
    esp_afe_sr_data_t *afe_data = arg;
    int audio_chunksize = afe_handle->get_feed_chunksize(afe_data);
    g_feed_chunk_samples = audio_chunksize;
    int nch = afe_handle->get_feed_channel_num(afe_data);
    int feed_channel = esp_get_feed_channel();
    g_feed_channel_count = feed_channel;
    assert(nch == feed_channel);
    int16_t *i2s_buff = malloc(audio_chunksize * sizeof(int16_t) * feed_channel);
    assert(i2s_buff);


    // 输入格式来自 bsp_get_input_format()，这里是 raw feed 顺序（is_get_raw_channel=true）
    // 当前 BSP 配置为 "RMMM"；若走 is_get_raw_channel=false 的重排顺序才是 "MMMR"
    char* str = esp_get_input_format();
    int length = nch;
    int positions[10];
    int count = 0;

    for (int i = 0; i < length; i++) {
        if (str[i] == 'M') {
            positions[count++] = i;
        }
    }

    if (count < 2) {
        printf("Invalid mic format %s, fallback to channel[0] and channel[1]\n", str);
        positions[0] = 0;
        positions[1] = 1;
    }
    g_mic_idx_left = positions[0];
    g_mic_idx_right = positions[1];
    // 回放固定使用 raw feed 的指定通道索引
    g_playback_ch_idx = RAW_PLAYBACK_FEED_CH_INDEX;
    if (g_playback_ch_idx < 0 || g_playback_ch_idx >= feed_channel) {
        g_playback_ch_idx = g_mic_idx_left;
    }
    if (str[g_playback_ch_idx] != 'M') {
        printf("Warning: playback channel ch%d is '%c' (not microphone)\n", g_playback_ch_idx, str[g_playback_ch_idx]);
    }
    printf("DOA mic pair: ch%d ch%d, playback channel: ch%d\n", g_mic_idx_left, g_mic_idx_right, g_playback_ch_idx);

    while (task_flag) {
        // 主循环职责：
        // 1) 读原始多通道数据
        // 2) 写入 raw ring（给唤醒后的回放和 DOA 用）
        // 3) feed 给 AFE（保证唤醒检测正常运行）
        esp_get_feed_data(true, i2s_buff, audio_chunksize * sizeof(int16_t) * feed_channel);
        raw_mic_ring_push(i2s_buff, audio_chunksize, feed_channel);
        afe_handle->feed(afe_data, i2s_buff);
    }
    if (i2s_buff) {
        free(i2s_buff);
        i2s_buff = NULL;
    }
    vTaskDelete(NULL);
}

void detect_Task(void *arg)
{
    esp_afe_sr_data_t *afe_data = arg;
    int replay_total_samples = (WAKEWORD_REPLAY_MS * SAMPLE_RATE_HZ) / 1000;
    int16_t *replay_linear = malloc(replay_total_samples * sizeof(int16_t));
    assert(replay_linear);
    TickType_t last_replay_tick = 0;

    printf("------------detect start------------\n");

    while (task_flag) {
        // 从 AFE 拉取识别结果（这里主要关注 wakeup_state）
        afe_fetch_result_t *res = afe_handle->fetch(afe_data);
        if (!res || res->ret_value == ESP_FAIL) {
            printf("fetch error!\n");
            continue;
        }

        if (res->wakeup_state == WAKENET_DETECTED) {
            // 冷却时间内忽略重复触发，避免连续回放/连续打印
            TickType_t now = xTaskGetTickCount();
            if ((now - last_replay_tick) < pdMS_TO_TICKS(WAKE_REPLAY_COOLDOWN_MS)) {
                continue;
            }
            last_replay_tick = now;

            printf("wakeword detected\n");
            printf("model index:%d, word index:%d\n", res->wakenet_model_index, res->wake_word_index);
            printf("-----------PLAY WAKE WORD AUDIO-----------\n");
#if DEBUG_PRINT_WAKE_CHANNEL_LEVEL
            debug_print_recent_channel_levels(replay_total_samples);
#endif

            // 回放源：raw ring 指定通道（原始麦克风数据），不是 AFE 单声道输出
            int replay_samples = raw_mic_ring_copy_recent_channel(replay_linear, replay_total_samples, g_playback_ch_idx);
            if (replay_samples > 0) {
                // esp_audio_play 的长度单位是“字节”
                esp_audio_play(replay_linear, replay_samples * sizeof(int16_t), portMAX_DELAY);
            }

            // DOA 只在唤醒时估算，窗口同样聚焦在唤醒词附近
            int doa_window_samples = (WAKEWORD_DOA_WINDOW_MS * SAMPLE_RATE_HZ) / 1000;
            float wake_doa = estimate_wake_doa_from_recent(doa_window_samples, g_mic_idx_left, g_mic_idx_right);
            printf("wakeword doa(avg): %.2f deg\n", wake_doa);
        }
    }

    if (replay_linear) {
        free(replay_linear);
        replay_linear = NULL;
    }

    vTaskDelete(NULL);
}

void app_main()
{
    // DAC 初始化在板级里完成，这里再显式把播放音量拉到最大（通常 100）
    ESP_ERROR_CHECK(esp_board_init(16000, 1, 16));
    ESP_ERROR_CHECK(esp_audio_set_play_vol(100));

    // 初始化语音模型与 AFE
    srmodel_list_t *models = esp_srmodel_init("model");
    afe_config_t *afe_config = afe_config_init(esp_get_input_format(), models, AFE_TYPE_SR, AFE_MODE_LOW_COST);
    printf("%s\n", esp_get_input_format());
    afe_config->aec_init = false;
    afe_config_print(afe_config);
    afe_handle = esp_afe_handle_from_config(afe_config);
    esp_afe_sr_data_t *afe_data = afe_handle->create_from_config(afe_config);
    afe_config_free(afe_config);

    // 为 raw ring 分配容量：
    // 取 max(回放窗口, DOA窗口)，保证两个功能都能拿到完整历史数据
    int feed_channel = esp_get_feed_channel();
    int raw_window_ms = (WAKEWORD_DOA_WINDOW_MS > WAKEWORD_REPLAY_MS) ? WAKEWORD_DOA_WINDOW_MS : WAKEWORD_REPLAY_MS;
    int raw_ring_samples = (raw_window_ms * SAMPLE_RATE_HZ) / 1000;
    int raw_ring_values = raw_ring_samples * feed_channel;
    g_raw_mic_ring = calloc(raw_ring_values, sizeof(int16_t));
    g_raw_mic_ring_len = raw_ring_values;
    assert(g_raw_mic_ring);

    task_flag = 1;
    xTaskCreatePinnedToCore(&feed_Task, "feed", 8 * 1024, (void*)afe_data, 5, NULL, 0);
    xTaskCreatePinnedToCore(&detect_Task, "detect", 8 * 1024, (void*)afe_data, 5, NULL, 1);
}
