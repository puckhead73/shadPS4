// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include "common/types.h"

namespace Common {

/**
 * Lightweight, always-on per-frame performance counters.
 *
 * Hot paths bump the live atomics with relaxed ordering (negligible cost).
 * Once per presented frame, NextFrame() folds the accumulators into a rolling
 * window and, every `kWindow` frames, publishes per-frame *averages* to the
 * display atomics. The overlay reads Last() — averaged values change slowly and
 * are actually readable, unlike raw per-frame counts which fluctuate wildly
 * (a Finish() or GPU wait may land on one frame in ten).
 *
 * Independent of Tracy: compiled in regardless of build type so we always have
 * a coarse breakdown of the per-frame GPU-stall budget.
 */
class PerfStats {
public:
    // Per-frame averages over the most recent window.
    struct Frame {
        float gpu_finishes;        // scheduler.Finish() (full pipeline drains) / frame
        float gpu_submits;         // command buffer submissions / frame
        float draws;               // graphics draw calls / frame
        float dispatches;          // compute dispatches / frame
        float pipeline_compiles;   // graphics/compute pipelines built / frame
        float pipeline_compile_ms; // wall time compiling pipelines / frame
        float readback_mb;         // GPU->CPU MB copied back to guest memory / frame
        float gpu_wait_ms;         // wall time CPU blocked on GPU timeline waits / frame
    };

    static PerfStats& Instance();

    void AddFinish() {
        cur_finishes.fetch_add(1, std::memory_order_relaxed);
    }
    void AddSubmit() {
        cur_submits.fetch_add(1, std::memory_order_relaxed);
    }
    void AddDraw() {
        cur_draws.fetch_add(1, std::memory_order_relaxed);
    }
    void AddDispatch() {
        cur_dispatches.fetch_add(1, std::memory_order_relaxed);
    }
    void AddPipelineCompile(float ms) {
        cur_pipeline_compiles.fetch_add(1, std::memory_order_relaxed);
        cur_pipeline_compile_us.fetch_add(static_cast<u64>(ms * 1000.0f),
                                          std::memory_order_relaxed);
    }
    void AddReadbackBytes(u64 bytes) {
        cur_readback_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    void AddGpuWaitTime(float ms) {
        cur_gpu_wait_us.fetch_add(static_cast<u64>(ms * 1000.0f), std::memory_order_relaxed);
    }

    // Fold this frame's accumulators into the window; publish averages every
    // kWindow frames. Call once per presented frame.
    void NextFrame();

    Frame Last() const {
        return Frame{
            disp_finishes.load(std::memory_order_relaxed),
            disp_submits.load(std::memory_order_relaxed),
            disp_draws.load(std::memory_order_relaxed),
            disp_dispatches.load(std::memory_order_relaxed),
            disp_pipeline_compiles.load(std::memory_order_relaxed),
            disp_pipeline_compile_ms.load(std::memory_order_relaxed),
            disp_readback_mb.load(std::memory_order_relaxed),
            disp_gpu_wait_ms.load(std::memory_order_relaxed),
        };
    }

private:
    // Number of frames averaged before the display is refreshed (~1-1.5s in-game).
    static constexpr u32 kWindow = 30;

    // Live per-frame accumulators (bumped from multiple threads).
    std::atomic<u32> cur_finishes{0};
    std::atomic<u32> cur_submits{0};
    std::atomic<u32> cur_draws{0};
    std::atomic<u32> cur_dispatches{0};
    std::atomic<u32> cur_pipeline_compiles{0};
    std::atomic<u64> cur_pipeline_compile_us{0};
    std::atomic<u64> cur_readback_bytes{0};
    std::atomic<u64> cur_gpu_wait_us{0};

    // Window accumulators (only touched in NextFrame, single-threaded).
    u32 win_frames{0};
    u64 sum_finishes{0};
    u64 sum_submits{0};
    u64 sum_draws{0};
    u64 sum_dispatches{0};
    u64 sum_pipeline_compiles{0};
    u64 sum_pipeline_compile_us{0};
    u64 sum_readback_bytes{0};
    u64 sum_gpu_wait_us{0};

    // Published per-frame averages (read by the UI thread).
    std::atomic<float> disp_finishes{0.0f};
    std::atomic<float> disp_submits{0.0f};
    std::atomic<float> disp_draws{0.0f};
    std::atomic<float> disp_dispatches{0.0f};
    std::atomic<float> disp_pipeline_compiles{0.0f};
    std::atomic<float> disp_pipeline_compile_ms{0.0f};
    std::atomic<float> disp_readback_mb{0.0f};
    std::atomic<float> disp_gpu_wait_ms{0.0f};
};

} // namespace Common
