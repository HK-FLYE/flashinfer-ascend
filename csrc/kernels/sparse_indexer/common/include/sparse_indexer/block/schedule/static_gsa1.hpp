/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_SCHEDULE_STATIC_GSA1_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_SCHEDULE_STATIC_GSA1_HPP

#include "catlass/detail/macros.hpp"
#include "sparse_indexer/block/schedule/block_scheduler.hpp"
#include "sparse_indexer/foundation/contracts/problem.hpp"

namespace SparseIndexer::Common::Block {

/**
 * One work item per bN2/gS1 block; all S2 tiles execute inside that core.
 * The policy is parameterized by a variant request source and task factory.
 */
class StaticGSA1BlockPolicy : public RequestBlockDiagonalPolicy {
public:
    using Base = RequestBlockDiagonalPolicy;
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
            total += detail::CeilDiv(qLen, params.s1Tile);
        }
        return total;
    }

    template <class RequestRangeSource>
    CATLASS_DEVICE static bool FindMmad(
        RequestRangeSource& source, Params const& params, uint64_t tileId, WorkItem& coord)
    {
        uint64_t accumulated = 0;
        for (uint32_t b = 0; b < source.Batch(); ++b) {
            const uint32_t requestQStart = source.QBegin(b);
            const uint32_t qLen = source.QLen(b);
            const uint32_t kLen = source.KLen(b);
            if (qLen == 0 || kLen == 0) {
                continue;
            }
            const uint32_t qDim = detail::CeilDiv(qLen, params.s1Tile);
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

    CATLASS_DEVICE static uint32_t CountS2Tiles(WorkItem const& coord, Params const& params)
    {
        return detail::CeilDiv(coord.kRows, params.s2Tile);
    }

    CATLASS_DEVICE static WorkItem MakeS2Coord(WorkItem const& coord, uint32_t s2Idx, Params const& params)
    {
        auto tile = coord;
        tile.kStart = coord.kStart + s2Idx * params.s2Tile;
        tile.kRows = AscendC::Std::min(params.s2Tile, coord.kRows - s2Idx * params.s2Tile);
        return tile;
    }
};

template <class RequestSource_, class TaskFactory_>
struct StaticGSA1SchedulePolicy {
    using RequestSource = RequestSource_;
    using TaskFactory = TaskFactory_;
    using Scheduler = BlockScheduler<RequestSource, StaticGSA1BlockPolicy>;
    using Params = typename Scheduler::Params;
    using Coord = WorkItem;
    using Task = typename TaskFactory::Task;

    template <class Tiling>
    CATLASS_DEVICE static Params MakeParams(
        Tiling const* tiling, GM_ADDR cuSeqlensQ, GM_ADDR sequsedK, GM_ADDR cmpResidualK)
    {
        Params params{};
        params.policy.s1Tile = tiling->s1Tile;
        params.policy.s2Tile = tiling->s2Tile;
        params.policy.topkRowsPerTask = tiling->topkRowsPerTask;
        params.source.InitFromGm(tiling, cuSeqlensQ, sequsedK, cmpResidualK);
        return params;
    }

    CATLASS_DEVICE static uint32_t CountS2Tiles(Coord const& coord, StaticGSA1BlockPolicy::Params const& params)
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
        return TaskFactory::Make(coord, source);
    }
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_SCHEDULE_STATIC_GSA1_HPP
