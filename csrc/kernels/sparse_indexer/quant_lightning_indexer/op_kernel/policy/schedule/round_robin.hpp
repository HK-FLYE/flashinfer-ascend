/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_SCHEDULE_ROUND_ROBIN_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_SCHEDULE_ROUND_ROBIN_HPP

#include "block/block_scheduler.hpp"
#include "catlass/detail/macros.hpp"
#include "arch35/quant_lightning_indexer_source.h"
#include "arch35/quant_lightning_indexer_task.h"
#include "quant_lightning_indexer_tiling_data.h"

namespace SparseIndexer::QuantLightningIndexer::Policy {

/**
 * C0 compatibility schedule. This is the historical request x qTile x kTile
 * round-robin mapping; later phases replace it with a gS1-oriented policy.
 */
struct RoundRobinSchedulePolicy {
    using RequestSource = QuantLightningIndexerKernelDetail::QuantLightningIndexerRequestSource;
    using BlockPolicy = SparseIndexer::Common::Block::RequestBlockDiagonalPolicy;
    using Scheduler = SparseIndexer::Common::Block::BlockScheduler<RequestSource, BlockPolicy>;
    using Params = typename Scheduler::Params;
    using Coord = SparseIndexer::Common::Block::ScheduleCoord;
    using Task = QuantLightningIndexerKernelDetail::QuantLightningIndexerTask;

    CATLASS_DEVICE static Params MakeParams(
        __gm__ QuantLightningIndexerTilingData const* tiling, GM_ADDR cuSeqlensQ, GM_ADDR sequsedK,
        GM_ADDR cmpResidualK)
    {
        Params params{};
        params.policy.s1Tile = tiling->s1Tile;
        params.policy.s2Tile = tiling->s2Tile;
        params.policy.topkRowsPerTask = tiling->topkRowsPerTask;
        params.source.InitFromGm(tiling, cuSeqlensQ, sequsedK, cmpResidualK);
        return params;
    }

    CATLASS_DEVICE static uint32_t CountS2Tiles(Coord const& coord, BlockPolicy::Params const& params)
    {
        return BlockPolicy::CountS2Tiles(coord, params);
    }

    CATLASS_DEVICE static Coord MakeS2Coord(
        Coord const& coord, uint32_t s2Idx, BlockPolicy::Params const& params)
    {
        return BlockPolicy::MakeS2Coord(coord, s2Idx, params);
    }

    CATLASS_DEVICE static Task MakeTask(Coord const& coord, RequestSource const& source)
    {
        return QuantLightningIndexerKernelDetail::MakeQuantLightningIndexerTask(coord, source);
    }
};

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_SCHEDULE_ROUND_ROBIN_HPP
