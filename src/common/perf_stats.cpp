// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/perf_stats.h"

namespace Common {

PerfStats& PerfStats::Instance() {
    static PerfStats instance;
    return instance;
}

void PerfStats::NextFrame() {
    // exchange(0) atomically reads each accumulated value and clears it for the
    // next frame, then fold it into the rolling window.
    sum_finishes += cur_finishes.exchange(0, std::memory_order_relaxed);
    sum_submits += cur_submits.exchange(0, std::memory_order_relaxed);
    sum_draws += cur_draws.exchange(0, std::memory_order_relaxed);
    sum_dispatches += cur_dispatches.exchange(0, std::memory_order_relaxed);
    sum_pipeline_compiles += cur_pipeline_compiles.exchange(0, std::memory_order_relaxed);
    sum_pipeline_compile_us += cur_pipeline_compile_us.exchange(0, std::memory_order_relaxed);
    sum_readback_bytes += cur_readback_bytes.exchange(0, std::memory_order_relaxed);
    sum_gpu_wait_us += cur_gpu_wait_us.exchange(0, std::memory_order_relaxed);
    ++win_frames;

    if (win_frames < kWindow) {
        return;
    }

    // Publish per-frame averages over the window.
    const float inv = 1.0f / static_cast<float>(win_frames);
    disp_finishes.store(static_cast<float>(sum_finishes) * inv, std::memory_order_relaxed);
    disp_submits.store(static_cast<float>(sum_submits) * inv, std::memory_order_relaxed);
    disp_draws.store(static_cast<float>(sum_draws) * inv, std::memory_order_relaxed);
    disp_dispatches.store(static_cast<float>(sum_dispatches) * inv, std::memory_order_relaxed);
    disp_pipeline_compiles.store(static_cast<float>(sum_pipeline_compiles) * inv,
                                 std::memory_order_relaxed);
    disp_pipeline_compile_ms.store(static_cast<float>(sum_pipeline_compile_us) * inv / 1000.0f,
                                   std::memory_order_relaxed);
    disp_readback_mb.store(static_cast<float>(sum_readback_bytes) * inv / (1024.0f * 1024.0f),
                           std::memory_order_relaxed);
    disp_gpu_wait_ms.store(static_cast<float>(sum_gpu_wait_us) * inv / 1000.0f,
                           std::memory_order_relaxed);

    // Reset the window.
    win_frames = 0;
    sum_finishes = 0;
    sum_submits = 0;
    sum_draws = 0;
    sum_dispatches = 0;
    sum_pipeline_compiles = 0;
    sum_pipeline_compile_us = 0;
    sum_readback_bytes = 0;
    sum_gpu_wait_us = 0;
}

} // namespace Common
