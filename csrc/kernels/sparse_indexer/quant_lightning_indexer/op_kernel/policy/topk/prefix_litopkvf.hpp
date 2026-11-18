/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_TOPK_PREFIX_LITOPKVF_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_TOPK_PREFIX_LITOPKVF_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/detail/macros.hpp"
#include "quant_lightning_indexer_tiling_data.h"

namespace SparseIndexer::QuantLightningIndexer::Policy {

/**
 * Current production TopK policy. return_value=0 uses the unified vector block
 * (RegTensor prefix below topK, LiTopKVF above topK). The scalar block is kept
 * only for the non-default public-value path and is not a performance claim.
 */
template <class VectorBlock_, class ScalarBlock_>
struct PrefixLiTopKVFPolicy {
    using VectorBlock = VectorBlock_;
    using ScalarBlock = ScalarBlock_;

    CATLASS_DEVICE static bool UseVector(__gm__ QuantLightningIndexerTilingData const* tiling)
    {
        return tiling->returnValue == 0;
    }

    template <class Scheduler_, class SchedulePolicy_>
    CATLASS_DEVICE static void Run(Catlass::Arch::Resource<Catlass::Arch::Ascend950>& resource,
        Scheduler_& scheduler, __gm__ QuantLightningIndexerTilingData const* tiling,
        AscendC::GlobalTensor<float>& scores, AscendC::GlobalTensor<int32_t> const& outIndices,
        AscendC::GlobalTensor<float> const& outValues)
    {
        AscendC::GlobalTensor<int32_t> dummyOffset;
        typename SchedulePolicy_::Coord coord{};
        if (UseVector(tiling)) {
            typename VectorBlock::Params params{};
            params.topK = tiling->topk;
            params.s2Stride = tiling->maxSeqLenK;
            params.s2Tile = tiling->topkS2Tile;
            params.hasOutputIdxOffset = false;
            VectorBlock topk(resource, scores, outIndices, params, dummyOffset, outValues);
            while (scheduler.GetNextTopKCoord(coord)) {
                const auto task = SchedulePolicy_::MakeTask(coord, scheduler.source());
                topk(coord.qStart, coord.actualQ, coord.kStart, coord.actualK, coord.localQStart,
                    task.activeQ, task.activeK, task.residual);
            }
        } else {
            typename ScalarBlock::Params params{};
            params.topK = tiling->topk;
            params.s2Stride = tiling->maxSeqLenK;
            params.s2Tile = tiling->topkS2Tile;
            params.hasOutputIdxOffset = false;
            params.writeValues = tiling->returnValue != 0;
            ScalarBlock topk(resource, scores, outIndices, params, dummyOffset, outValues);
            while (scheduler.GetNextTopKCoord(coord)) {
                topk(coord.qStart, coord.actualQ, coord.kStart, coord.actualK);
            }
        }
    }
};

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_TOPK_PREFIX_LITOPKVF_HPP
