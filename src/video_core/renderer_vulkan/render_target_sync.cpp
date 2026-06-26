// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/render_target_sync.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

RenderTargetSync::RenderTargetSync(const Instance& instance_, Scheduler& scheduler_,
                                   VideoCore::TextureCache& texture_cache_)
    : instance{instance_}, scheduler{scheduler_}, texture_cache{texture_cache_} {}

RenderTargetSync::~RenderTargetSync() = default;

void RenderTargetSync::RecordRtWrite(VAddr addr, VideoCore::ImageId id) {
    if (!texture_cache.IsImageValid(id))
        return;
    pending_rt_writes_[addr] = {id, texture_cache.GetImage(id).image_uid};
    // New RT content at this address — old dedup is stale.
    pending_rt_copied_.erase(addr);
}

void RenderTargetSync::CopyFromLastRt(VAddr addr, VideoCore::ImageId tex_id, u32 copy_w,
                                      u32 copy_h) {
    auto it = pending_rt_writes_.find(addr);
    if (it == pending_rt_writes_.end())
        return;

    const PendingRt rt = it->second;
    // The recorded RT may have been freed (and its slot possibly reused by a different image)
    // during texture-cache churn. Dereferencing a stale id reads destructed memory and hands a
    // dangling VkImage to the driver, so verify the slot is live and still the same image.
    if (!texture_cache.IsImageValid(rt.id) ||
        texture_cache.GetImage(rt.id).image_uid != rt.uid) {
        pending_rt_writes_.erase(it);
        pending_rt_copied_.erase(addr);
        return;
    }

    const VideoCore::ImageId rt_id = rt.id;
    auto& rt_image = texture_cache.GetImage(rt_id);

    if (rt_id == tex_id)
        return;
    if (!texture_cache.IsImageValid(tex_id))
        return;
    auto& tex_image = texture_cache.GetImage(tex_id);

    // When the RT is SMALLER than the consumer texture, the meaning depends on whether they are a
    // genuine alias:
    //  - SAME pixel format → the game rendered a sub-region into a larger (usually pow2) allocation
    //    and now samples it. This is the post-processing pyramid (1600x900 into 2048x2048, etc.);
    //    we must copy the overlap or the consumer reads black → cutscene black-rectangle flicker.
    //  - DIFFERENT format → address reuse for an unrelated resource (a FALSE alias). Copying would
    //    corrupt the real texture (observed: the scene/ice RT bleeding black+noise). Skip it.
    // CopyRtToAlias clamps the copy extent to the overlap, so the same-format case never reads OOB.
    const bool rt_smaller =
        rt_image.info.size.width < copy_w || rt_image.info.size.height < copy_h;
    if (rt_smaller && rt_image.info.pixel_format != tex_image.info.pixel_format)
        return;

    // Dedup: each tex_id only pulls once per submit from this addr's RT.
    auto& copied = pending_rt_copied_[addr];
    if (!copied.insert(tex_id).second)
        return;

    CopyRtToAlias(rt_image, tex_image);
}

void RenderTargetSync::PushPendingRtAliases() {
    for (auto& [addr, rt] : pending_rt_writes_) {
        if (texture_cache.IsImageValid(rt.id) &&
            texture_cache.GetImage(rt.id).image_uid == rt.uid) {
            PushRtToAliases(addr, rt.id);
        }
    }
    pending_rt_writes_.clear();
}

void RenderTargetSync::ClearRecords() {
    pending_rt_writes_.clear();
    pending_rt_copied_.clear();
}

void RenderTargetSync::Schedule1x1Readback(VideoCore::ImageId image_id) {
    texture_cache.AddDownload(image_id);
}

void RenderTargetSync::PushRtToAliases(VAddr addr, VideoCore::ImageId rt_id) {
    if (!texture_cache.IsImageValid(rt_id))
        return;
    auto& rt_image = texture_cache.GetImage(rt_id);

    const u64 page = addr >> VideoCore::TextureCache::Traits::PageBits;
    const auto& page_table = texture_cache.GetPageTable();
    const auto page_it = page_table.find(page);
    if (!page_it)
        return;

    for (VideoCore::ImageId alias_id : *page_it) {
        if (alias_id == rt_id)
            continue;
        if (!texture_cache.IsImageValid(alias_id))
            continue;
        auto& alias_image = texture_cache.GetImage(alias_id);
        if (alias_image.info.guest_address != addr)
            continue;
        if (alias_image.info.props.is_depth)
            continue;
        if (rt_image.info.size.width < alias_image.info.size.width)
            continue;
        if (rt_image.info.size.height < alias_image.info.size.height)
            continue;

        // Skip aliases that are themselves pending RTs — avoids RT↔RT feedback loops.
        auto pend_it = pending_rt_writes_.find(addr);
        if (pend_it != pending_rt_writes_.end() && pend_it->second.id == alias_id)
            continue;

        CopyRtToAlias(rt_image, alias_image);
    }
}

void RenderTargetSync::CopyRtToAlias(VideoCore::Image& rt_image, VideoCore::Image& alias_image) {
    // During heavy texture-cache churn (e.g. cycling menu/jersey assets) a recorded RT or its
    // alias can be freed/recreated, leaving the ImageId pointing at a slot with no live backing.
    // Issuing a copy with a VK_NULL_HANDLE (or freed) VkImage faults inside the GPU driver, so bail
    // out before touching either handle.
    if (!rt_image.HasBackingImage() || !alias_image.HasBackingImage()) {
        LOG_WARNING(Render_Vulkan,
                    "[CopyRtToAlias] Skipping copy, missing backing image (rt valid={}, alias "
                    "valid={}) @ {:#x}",
                    rt_image.HasBackingImage(), alias_image.HasBackingImage(),
                    alias_image.info.guest_address);
        return;
    }

    // Clamp the copy to the region both images actually cover, so neither the read (RT) nor the
    // write (alias) goes out of bounds when the two differ in size (post-processing pyramids,
    // pow2-allocated render textures, etc.).
    const u32 copy_w = std::min(rt_image.info.size.width, alias_image.info.size.width);
    const u32 copy_h = std::min(rt_image.info.size.height, alias_image.info.size.height);

    if (rt_image.info.num_samples != alias_image.info.num_samples) {
        scheduler.EndRendering();
        auto cmdbuf = scheduler.CommandBuffer();

        VideoCore::UniqueImage temp{instance.GetDevice(), instance.GetAllocator()};
        temp.Create({
            .flags =
                vk::ImageCreateFlagBits::eMutableFormat | vk::ImageCreateFlagBits::eExtendedUsage,
            .imageType = vk::ImageType::e2D,
            .format = rt_image.info.pixel_format,
            .extent = {copy_w, copy_h, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
        });

        rt_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                         {});
        {
            const vk::ImageMemoryBarrier2 temp_barrier = {
                .srcStageMask = vk::PipelineStageFlagBits2::eNone,
                .srcAccessMask = vk::AccessFlagBits2::eNone,
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .image = temp,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                       .pImageMemoryBarriers = &temp_barrier});
        }
        const vk::ImageResolve resolve_region = {
            .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .srcOffset = {0, 0, 0},
            .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .dstOffset = {0, 0, 0},
            .extent = {copy_w, copy_h, 1},
        };
        cmdbuf.resolveImage(rt_image.GetImage(), vk::ImageLayout::eTransferSrcOptimal, temp,
                            vk::ImageLayout::eTransferDstOptimal, resolve_region);

        {
            const vk::ImageMemoryBarrier2 temp_barrier = {
                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .image = temp,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                       .pImageMemoryBarriers = &temp_barrier});
        }
        alias_image.Transit(vk::ImageLayout::eTransferDstOptimal,
                            vk::AccessFlagBits2::eTransferWrite, {});
        const vk::ImageCopy copy_region = {
            .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .srcOffset = {0, 0, 0},
            .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .dstOffset = {0, 0, 0},
            .extent = {copy_w, copy_h, 1},
        };
        cmdbuf.copyImage(temp, vk::ImageLayout::eTransferSrcOptimal, alias_image.GetImage(),
                         vk::ImageLayout::eTransferDstOptimal, copy_region);

        rt_image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
                         vk::AccessFlagBits2::eColorAttachmentWrite, {});
        alias_image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal,
                            vk::AccessFlagBits2::eShaderRead, {});
        scheduler.DeferOperation([temp = std::move(temp)]() mutable { temp.Destroy(); });
        return;
    }

    // Same sample count: direct copyImage.
    rt_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
    alias_image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
                        {});
    const vk::ImageCopy region = {
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .srcOffset = {0, 0, 0},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstOffset = {0, 0, 0},
        .extent = {copy_w, copy_h, 1},
    };
    scheduler.CommandBuffer().copyImage(rt_image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                        alias_image.GetImage(),
                                        vk::ImageLayout::eTransferDstOptimal, region);
    rt_image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
                     vk::AccessFlagBits2::eColorAttachmentWrite, {});
    alias_image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
                        {});
}

} // namespace Vulkan