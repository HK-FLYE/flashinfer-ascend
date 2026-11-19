/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_REDUCE_ROW_SPLIT_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_REDUCE_ROW_SPLIT_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/detail/macros.hpp"
#include "quant_lightning_indexer_tiling_data.h"

namespace SparseIndexer::QuantLightningIndexer::Policy {

/**
 * QLIv2-style reduce ownership: vector0 and vector1 each process half of the
 * query rows over the complete S2 tile. The mmad SPLIT_M policy writes the
 * matching half-M slices into each subblock's UB.
 */
template <class Block_>
struct RowSplitReducePolicy {
    using Block = Block_;
    using Params = typename Block::Params;

    CATLASS_DEVICE static Params MakeParams(__gm__ QuantLightningIndexerTilingData const* tiling)
    {
        Params params{};
        params.n1 = tiling->n1;
        params.s2 = tiling->maxSeqLenK;
        params.s1Tile = tiling->s1Tile;
        params.s2Tile = tiling->s2Tile;
        params.maskMode = tiling->maskMode;
        params.cmpRatio = tiling->cmpRatio;
        params.blockSize = tiling->blockSize;
        params.maxBlocksPerBatch = tiling->maxBlocksPerBatch;
        params.numPages = tiling->numPages;
        params.kScaleStride0 = tiling->kScaleStride0;
        return params;
    }

    template <class SyncPolicy_, class Scheduler_, class SchedulePolicy_>
    CATLASS_DEVICE static void Run(Catlass::Arch::Resource<Catlass::Arch::Ascend950>& resource,
        AscendC::LocalTensor<typename Block::DtypeScore>& ubC,
        AscendC::GlobalTensor<float> const& gmW, AscendC::GlobalTensor<float> const& gmQScale,
        AscendC::GlobalTensor<float> const& gmKScale, AscendC::GlobalTensor<int32_t> const& gmBlockTable,
        AscendC::GlobalTensor<float>& scores, __gm__ QuantLightningIndexerTilingData const* tiling,
        Scheduler_& scheduler)
    {
        typename Block::Params params{MakeParams(tiling)};
        Block reducer(ubC, resource, gmW, gmQScale, gmKScale, gmBlockTable, scores, params);
        SyncPolicy_::AivArriveVecReady();
        typename SchedulePolicy_::Coord coord{};
        while (scheduler.GetNextCoord(coord)) {
            const auto task = SchedulePolicy_::MakeTask(coord, scheduler.source());
            const uint32_t s2Tiles = SchedulePolicy_::CountS2Tiles(coord, scheduler.policy());
            for (uint32_t s2Idx = 0; s2Idx < s2Tiles; ++s2Idx) {
                const auto tile = SchedulePolicy_::MakeS2Coord(coord, s2Idx, scheduler.policy());
                SyncPolicy_::AivWaitCubeDone();
                reducer(
                    tile.qStart, tile.actualQ, tile.kStart, tile.actualK,
                    tile.localQStart, task.activeQ, task.activeK, task.requestKStart, task.residual,
                    task.maskActive, tile.batchIdx);
                SyncPolicy_::AivArriveVecReady();
            }
        }
    }
};

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_REDUCE_ROW_SPLIT_HPP
