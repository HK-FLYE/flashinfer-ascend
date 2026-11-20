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
 * Current production TopK policy: RegTensor prefix below topK,
 * LiTopKVF above topK.
 */
template <class VectorBlock_>
struct PrefixLiTopKVFPolicy {
    using VectorBlock = VectorBlock_;

    template <class Scheduler_, class SchedulePolicy_>
    CATLASS_DEVICE static void Run(Catlass::Arch::Resource<Catlass::Arch::Ascend950>& resource,
        Scheduler_& scheduler, __gm__ QuantLightningIndexerTilingData const* tiling,
        AscendC::GlobalTensor<float>& scores, AscendC::GlobalTensor<int32_t> const& outIndices)
    {
        AscendC::GlobalTensor<int32_t> dummyOffset;
        typename SchedulePolicy_::Coord coord{};
        typename VectorBlock::Params params{};
        params.topK = tiling->topk;
        params.s2Stride = tiling->maxSeqLenK;
        params.s2Tile = tiling->topkS2Tile;
        params.hasOutputIdxOffset = false;
        VectorBlock topk(resource, scores, outIndices, params, dummyOffset);
        while (scheduler.GetNextTopKCoord(coord)) {
            const auto task = SchedulePolicy_::MakeTask(coord, scheduler.source());
            topk(coord.qStart, coord.actualQ, coord.kStart, coord.actualK, coord.localQStart,
                task.activeQ, task.activeK, task.residual);
        }
    }
};


/**
 * Inline TopK used after one gS1 has finished all S2 work on the same AIV.
 * LocalRows disables the generic TopK block's subblock row split, so each AIV
 * consumes exactly the rows it owns.
 */
template <class VectorBlock_>
struct InlinePrefixLiTopKVFPolicy {
    using VectorBlock = VectorBlock_;

    CATLASS_DEVICE static void RunLocal(Catlass::Arch::Resource<Catlass::Arch::Ascend950>& resource,
        __gm__ QuantLightningIndexerTilingData const* tiling, AscendC::GlobalTensor<float>& scores,
        AscendC::GlobalTensor<int32_t> const& outIndices, uint32_t s1Start, uint32_t rows,
        uint32_t kStart, uint32_t kLen, uint32_t localQStart, uint32_t qLen, int32_t residual)
    {
        if (rows == 0) {
            return;
        }
        AscendC::GlobalTensor<int32_t> dummyOffset;
        typename VectorBlock::Params params{};
        params.topK = tiling->topk;
        params.s2Stride = tiling->maxSeqLenK;
        params.s2Tile = tiling->topkS2Tile;
        params.hasOutputIdxOffset = false;
        VectorBlock topk(resource, scores, outIndices, params, dummyOffset);
        topk(s1Start, rows, kStart, kLen, localQStart, qLen, kLen, residual);
    }
};

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_TOPK_PREFIX_LITOPKVF_HPP
