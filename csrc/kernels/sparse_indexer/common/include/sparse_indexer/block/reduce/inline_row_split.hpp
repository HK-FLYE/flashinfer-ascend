/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_REDUCE_INLINE_ROW_SPLIT_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_REDUCE_INLINE_ROW_SPLIT_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/detail/macros.hpp"
#include "sparse_indexer/block/mask/visible_coord.hpp"
#include "sparse_indexer/block/reduce/runtime.hpp"
#include "sparse_indexer/block/selection/runtime.hpp"

namespace SparseIndexer::Common::Block {

/**
 * C2 reduce ownership. Each AIV processes half of the gS1 rows across all S2
 * tiles, then runs TopK for exactly those rows on the same core.
 */
template <class Block_>
struct InlineRowSplitReducePolicy {
    using Block = Block_;
    using Params = typename Block::Params;

    template <class SyncPolicy_, class Scheduler_, class SchedulePolicy_, class TopKPolicy_>
    CATLASS_DEVICE static void Run(Catlass::Arch::Resource<Catlass::Arch::Ascend950>& resource,
        AscendC::LocalTensor<typename Block::DtypeScore>& ubC,
        AscendC::GlobalTensor<float> const& gmW, AscendC::GlobalTensor<float> const& gmQScale,
        AscendC::GlobalTensor<float> const& gmKScale, AscendC::GlobalTensor<int32_t> const& gmBlockTable,
        AscendC::GlobalTensor<float>& scores, ReduceRuntimeParams<Block> const& runtime,
        SelectionRuntimeParams const& selection, Scheduler_& scheduler,
        AscendC::GlobalTensor<int32_t> const& outIndices)
    {
        Block reducer(ubC, resource, gmW, gmQScale, gmKScale, gmBlockTable, scores, runtime.block);
        const bool doubleBuffered = runtime.scoreBuffers > 1U;
        SyncPolicy_::AivArriveVecReady(0);
        if (doubleBuffered) {
            SyncPolicy_::AivArriveVecReady(1);
        }
        typename SchedulePolicy_::Coord coord{};
        while (scheduler.Next(coord)) {
            const auto task = SchedulePolicy_::MakeTask(coord, scheduler.source());
            const auto visibleCoord = VisibleCoord(runtime.mask, coord, task.activeQ, task.activeK, task.residual);
            const uint32_t s2Tiles = SchedulePolicy_::CountS2Tiles(visibleCoord, scheduler.policy());
            for (uint32_t s2Idx = 0; s2Idx < s2Tiles; ++s2Idx) {
                const auto tile = SchedulePolicy_::MakeS2Coord(visibleCoord, s2Idx, scheduler.policy());
                const uint32_t pingpong = doubleBuffered ? (s2Idx & 1U) : 0U;
                SyncPolicy_::AivWaitCubeDone(pingpong);
                reducer(
                    tile.qStart, tile.qRows, tile.kStart, tile.kRows,
                    tile.localQBegin, task.activeQ, task.activeK, task.requestKStart, task.residual,
                    task.maskActive, tile.requestId, pingpong);
                // The last tile also carries the score hand-off for this gS1 task.
                // Keep the cube/vector ready flag pending until the owning AIV has
                // finished TopK, otherwise the next gS1 task can overwrite UB.
                if (s2Idx + 1U < s2Tiles) {
                    SyncPolicy_::AivArriveVecReady(pingpong);
                }
            }

            reducer.WaitAllOutputs();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            const uint32_t sub = AscendC::GetSubBlockIdx();
            const uint32_t rowsPerAiv = SparseIndexer::Common::CeilDiv(task.coord.qRows, 2U);
            const uint32_t rowBegin = sub * rowsPerAiv;
            if (rowBegin < task.coord.qRows) {
                const uint32_t rows = AscendC::Std::min(rowsPerAiv, task.coord.qRows - rowBegin);
                TopKPolicy_::RunLocal(resource, selection, scores, outIndices,
                    task.coord.qStart + rowBegin, rows, task.requestKStart, task.activeK,
                    task.coord.localQBegin + rowBegin, task.activeQ, task.residual);
            }
            const uint32_t lastPingpong = doubleBuffered ? ((s2Tiles - 1U) & 1U) : 0U;
            SyncPolicy_::AivArriveVecReady(lastPingpong);
        }
    }
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_REDUCE_INLINE_ROW_SPLIT_HPP
