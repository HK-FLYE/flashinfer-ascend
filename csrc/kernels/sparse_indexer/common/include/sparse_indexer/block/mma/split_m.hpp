/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_MMAD_SPLIT_M_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_MMAD_SPLIT_M_HPP

#include "sparse_indexer/block/mma/block_mmad.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/detail/macros.hpp"
#include "catlass/gemm_coord.hpp"
#include "sparse_indexer/block/mask/visible_coord.hpp"
#include "sparse_indexer/block/mma/runtime.hpp"

namespace SparseIndexer::Common::Block {

/**
 * QLIv2-style row split. M is the flattened query x head axis, so the half-M
 * boundary must be exactly 2 * N1: vector0 owns query rows 0..1 and vector1
 * owns rows 2..3. The alignment is deliberately not a fixed four M rows.
 */
template <class Dtype_, class LoadPolicy_>
struct SplitMMmadPolicy {
    using Dtype = Dtype_;
    using LoadPolicy = LoadPolicy_;
    using InputType = typename LoadPolicy::Dtype;
    using Block = SparseIndexer::Common::Block::BlockMmad<Dtype, Dtype, typename LoadPolicy::Loader,
        Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using Params = typename Block::Params;

    template <class SyncPolicy_, class Scheduler_, class SchedulePolicy_>
    CATLASS_DEVICE static void Run(Catlass::Arch::Resource<Catlass::Arch::Ascend950>& resource,
        AscendC::LocalTensor<typename Block::DtypeC> const& ubC,
        AscendC::GlobalTensor<InputType> const& gmQ, AscendC::GlobalTensor<InputType> const& gmK,
        MmaRuntimeParams<Block> const& runtime,
        AscendC::GlobalTensor<int32_t> const& gmBlockTable,
        AscendC::GlobalTensor<InputType> const& gmZeroPage, Scheduler_& scheduler)
    {
        Block blockMmad(
            ubC, gmQ, gmK, runtime.problemShape, runtime.tileShape, resource, runtime.block,
            gmBlockTable, gmZeroPage);
        const bool doubleBuffered = runtime.scoreBuffers > 1U;
        const uint32_t scoreElems = runtime.scoreElemsPerAiv;
        uint32_t lastPingpong = 0;
        typename SchedulePolicy_::Coord coord{};
        while (scheduler.Next(coord)) {
            const auto task = SchedulePolicy_::MakeTask(coord, scheduler.source());
            const auto visibleCoord = VisibleCoord(runtime.mask, coord, task.activeQ, task.activeK, task.residual);
            const uint32_t s2Tiles = SchedulePolicy_::CountS2Tiles(visibleCoord, scheduler.policy());
            blockMmad.BeginTask(coord.qStart * runtime.heads, coord.qRows * runtime.heads);
            for (uint32_t s2Idx = 0; s2Idx < s2Tiles; ++s2Idx) {
                const auto tile = SchedulePolicy_::MakeS2Coord(visibleCoord, s2Idx, scheduler.policy());
                const SparseIndexer::Common::Block::KeyCoord keyCoord{
                    tile.kStart, tile.kStart - task.requestKStart, tile.requestId};
                if (s2Idx == 0U) {
                    blockMmad.PrefetchKey(0, keyCoord, tile.kRows);
                }
                if (s2Idx + 1U < s2Tiles) {
                    const auto nextTile = SchedulePolicy_::MakeS2Coord(visibleCoord, s2Idx + 1U, scheduler.policy());
                    const SparseIndexer::Common::Block::KeyCoord nextKeyCoord{
                        nextTile.kStart, nextTile.kStart - task.requestKStart, nextTile.requestId};
                    blockMmad.PrefetchKey((s2Idx + 1U) % 3U, nextKeyCoord, nextTile.kRows);
                }
                const uint32_t keyBufferIdx = s2Idx % 3U;
                const uint32_t pingpong = doubleBuffered ? (s2Idx & 1U) : 0U;
                if (doubleBuffered) {
                    blockMmad.Compute(
                        tile.qStart * runtime.heads, tile.qRows * runtime.heads, keyBufferIdx, tile.kRows);
                    SyncPolicy_::AicWaitCubeReady(pingpong);
                    blockMmad.SetOutputOffset(pingpong * scoreElems);
                    blockMmad.Store(tile.qRows * runtime.heads, tile.kRows);
                } else {
                    SyncPolicy_::AicWaitCubeReady(0);
                    blockMmad.Compute(
                        tile.qStart * runtime.heads, tile.qRows * runtime.heads, keyBufferIdx, tile.kRows);
                    blockMmad.Store(tile.qRows * runtime.heads, tile.kRows);
                }
                SyncPolicy_::AicArriveCubeDone(pingpong);
                lastPingpong = pingpong;
            }
        }
        SyncPolicy_::AicFinalWait(lastPingpong);
    }
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_MMAD_SPLIT_M_HPP
