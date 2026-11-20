/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_SCHEDULE_STATIC_GSA1_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_SCHEDULE_STATIC_GSA1_HPP

#include "sparse_indexer/block/schedule/block_scheduler.hpp"
#include "catlass/detail/macros.hpp"
#include "arch35/quant_lightning_indexer_source.h"
#include "arch35/quant_lightning_indexer_task.h"
#include "quant_lightning_indexer_tiling_data.h"

namespace SparseIndexer::QuantLightningIndexer::Policy {

/**
 * One work item per bN2/gS1 block; all S2 tiles execute inside that core.
 * TopK keeps the existing row-group schedule after the global reduce barrier.
 */
class StaticGSA1BlockPolicy : public SparseIndexer::Common::Block::RequestBlockDiagonalPolicy {
public:
    using Base = SparseIndexer::Common::Block::RequestBlockDiagonalPolicy;
    using Params = typename Base::Params;

    template <class RequestRangeSource>
    CATLASS_DEVICE static uint64_t CountMmad(RequestRangeSource& source, Params const& params)
    {
        uint64_t total = 0;
        for (uint32_t b = 0; b < source.Batch(); ++b) {
            const uint32_t qLen = source.QLen(b);
            if (qLen == 0 || source.KLen(b) == 0) {
                continue;
            }
            total += SparseIndexer::Common::Block::detail::CeilDiv(qLen, params.s1Tile);
        }
        return total;
    }

    template <class RequestRangeSource>
    CATLASS_DEVICE static bool FindMmad(
        RequestRangeSource& source, Params const& params, uint64_t tileId,
        SparseIndexer::Common::Block::ScheduleCoord& coord)
    {
        uint64_t accumulated = 0;
        for (uint32_t b = 0; b < source.Batch(); ++b) {
            const uint32_t requestQStart = source.QBegin(b);
            const uint32_t qLen = source.QLen(b);
            const uint32_t kLen = source.KLen(b);
            if (qLen == 0 || kLen == 0) {
                continue;
            }
            const uint32_t qDim = SparseIndexer::Common::Block::detail::CeilDiv(qLen, params.s1Tile);
            if (tileId >= accumulated + qDim) {
                accumulated += qDim;
                continue;
            }
            const uint32_t qBlock = static_cast<uint32_t>(tileId - accumulated);
            coord.requestId = b;
            coord.localQBegin = qBlock * params.s1Tile;
            coord.qStart = requestQStart + coord.localQBegin;
            coord.qRows = (qBlock + 1U == qDim) ? (qLen - coord.localQBegin) : params.s1Tile;
            coord.kStart = source.KBegin(b);
            coord.kRows = kLen;
            return true;
        }
        return false;
    }

    CATLASS_DEVICE static uint32_t CountS2Tiles(
        SparseIndexer::Common::Block::ScheduleCoord const& coord, Params const& params)
    {
        return SparseIndexer::Common::Block::detail::CeilDiv(coord.kRows, params.s2Tile);
    }

    CATLASS_DEVICE static SparseIndexer::Common::Block::ScheduleCoord MakeS2Coord(
        SparseIndexer::Common::Block::ScheduleCoord const& coord, uint32_t s2Idx, Params const& params)
    {
        auto tile = coord;
        tile.kStart = coord.kStart + s2Idx * params.s2Tile;
        tile.kRows = AscendC::Std::min(params.s2Tile, coord.kRows - s2Idx * params.s2Tile);
        return tile;
    }
};

struct StaticGSA1SchedulePolicy {
    using RequestSource = QuantLightningIndexerKernelDetail::QuantLightningIndexerRequestSource;
    using Scheduler = SparseIndexer::Common::Block::BlockScheduler<RequestSource, StaticGSA1BlockPolicy>;
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

    CATLASS_DEVICE static uint32_t CountS2Tiles(
        Coord const& coord, StaticGSA1BlockPolicy::Params const& params)
    {
        return StaticGSA1BlockPolicy::CountS2Tiles(coord, params);
    }

    CATLASS_DEVICE static Coord MakeS2Coord(
        Coord const& coord, uint32_t s2Idx, StaticGSA1BlockPolicy::Params const& params)
    {
        return StaticGSA1BlockPolicy::MakeS2Coord(coord, s2Idx, params);
    }

    CATLASS_DEVICE static Task MakeTask(Coord const& coord, RequestSource const& source)
    {
        return QuantLightningIndexerKernelDetail::MakeQuantLightningIndexerTask(coord, source);
    }
};

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_SCHEDULE_STATIC_GSA1_HPP
