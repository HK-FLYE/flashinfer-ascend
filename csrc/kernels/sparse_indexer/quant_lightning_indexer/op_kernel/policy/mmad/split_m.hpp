/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_MMAD_SPLIT_M_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_MMAD_SPLIT_M_HPP

#include "block/block_mmad.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/detail/macros.hpp"
#include "catlass/gemm_coord.hpp"
#include "quant_lightning_indexer_tiling_data.h"

namespace SparseIndexer::QuantLightningIndexer::Policy {

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

    CATLASS_DEVICE static Params MakeParams(__gm__ QuantLightningIndexerTilingData const* tiling)
    {
        Params params{};
        params.keyLoader.d = tiling->d;
        params.keyLoader.blockSize = tiling->blockSize;
        params.keyLoader.maxBlocksPerBatch = tiling->maxBlocksPerBatch;
        params.keyLoader.numPages = tiling->numPages;
        params.keyLoader.keyStride0 = tiling->keyStride0;
        params.mSplitAlignment = 2U * tiling->n1;
        return params;
    }

    template <class SyncPolicy_, class Scheduler_, class SchedulePolicy_>
    CATLASS_DEVICE static void Run(Catlass::Arch::Resource<Catlass::Arch::Ascend950>& resource,
        AscendC::LocalTensor<typename Block::DtypeC> const& ubC,
        AscendC::GlobalTensor<InputType> const& gmQ, AscendC::GlobalTensor<InputType> const& gmK,
        __gm__ QuantLightningIndexerTilingData const* tiling,
        AscendC::GlobalTensor<int32_t> const& gmBlockTable,
        AscendC::GlobalTensor<InputType> const& gmZeroPage, Scheduler_& scheduler)
    {
        typename Block::Params mmadParams{MakeParams(tiling)};
        Block blockMmad(
            ubC, gmQ, gmK, Catlass::GemmCoord(tiling->tq * tiling->n1, tiling->maxSeqLenK, tiling->d),
            Catlass::GemmCoord(
                SparseIndexer::Common::AlignUp(tiling->s1Tile * tiling->n1, 16U), tiling->s2Tile, tiling->d),
            resource, mmadParams, gmBlockTable, gmZeroPage);
        typename SchedulePolicy_::Coord coord{};
        while (scheduler.GetNextCoord(coord)) {
            const auto task = SchedulePolicy_::MakeTask(coord, scheduler.source());
            const uint32_t s2Tiles = SchedulePolicy_::CountS2Tiles(coord, scheduler.policy());
            for (uint32_t s2Idx = 0; s2Idx < s2Tiles; ++s2Idx) {
                const auto tile = SchedulePolicy_::MakeS2Coord(coord, s2Idx, scheduler.policy());
                const SparseIndexer::Common::Block::KeyCoord keyCoord{
                    tile.kStart, tile.kStart - task.requestKStart, tile.batchIdx};
                SyncPolicy_::AicWaitCubeReady();
                blockMmad(
                    tile.qStart * tiling->n1, tile.actualQ * tiling->n1, keyCoord, tile.actualK);
                SyncPolicy_::AicArriveCubeDone();
            }
        }
        SyncPolicy_::AicFinalWait();
    }
};

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_MMAD_SPLIT_M_HPP
