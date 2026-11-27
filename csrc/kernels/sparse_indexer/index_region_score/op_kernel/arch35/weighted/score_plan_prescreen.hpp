#ifndef IRS_WEIGHTED_SCORE_PLAN_PRESCREEN_HPP
#define IRS_WEIGHTED_SCORE_PLAN_PRESCREEN_HPP

namespace IndexerRegionScoreWeighted {

template <bool CAUSAL>
__aicore__ inline uint64_t EstimateScoreSpan(const PagedTilingData &tiling, uint32_t qRows,
    int64_t position)
{
    uint32_t visible = tiling.seqK;
    if constexpr (CAUSAL) {
        const uint64_t causalEnd = position > 0 ? static_cast<uint64_t>(position) / tiling.regionSize : 0U;
        visible = causalEnd < visible ? static_cast<uint32_t>(causalEnd) : visible;
    }
    const uint32_t macros = (visible + PAGED_K_MACRO - 1U) / PAGED_K_MACRO;
    const uint32_t kEnd = PagedMin(tiling.seqK, macros * PAGED_K_MACRO);
    const uint64_t nTiles = (kEnd + PAGED_NR - 1U) / PAGED_NR;
    const uint64_t mTiles = (static_cast<uint64_t>(qRows) * tiling.headsPerGroup + 15U) / 16U;
    return nTiles * (mTiles * tiling.groups + 1U);
}

template <bool CAUSAL>
__aicore__ __attribute__((noinline)) bool EstimateSmallScoreGrid(
    const PagedTilingData &tiling, GM_ADDR cu, GM_ADDR kv, uint64_t planningCost)
{
    const auto bounds = reinterpret_cast<__gm__ const int32_t *>(cu);
    const auto lengths = reinterpret_cast<__gm__ const int32_t *>(kv);
    const DeviceScoreEpoch all{0U, tiling.totalTokens, 0U, tiling.numRequests};
    uint32_t shard = 0U;
    uint32_t shardEnd = ScoreGridQEnd(tiling, all, 1U);
    uint64_t gridPeak = 0U;
    uint64_t gridLoad = 0U;
    uint64_t total = 0U;
    int32_t previous = bounds[0];
    for (uint32_t request = 0U; request < tiling.numRequests; ++request) {
        const int32_t next = bounds[request + 1U];
        if (previous < 0 || next < previous || static_cast<uint32_t>(next) > tiling.totalTokens) {
            return true;
        }
        const uint32_t start = static_cast<uint32_t>(previous);
        const uint32_t stop = static_cast<uint32_t>(next);
        previous = next;
        if (start == stop) {
            continue;
        }
        int64_t position = 0;
        if constexpr (CAUSAL) {
            position = static_cast<int64_t>(lengths[request]) - 1;
        }
        const uint64_t requestWork = EstimateScoreSpan<CAUSAL>(tiling, stop - start, position);
        total += requestWork;
        uint32_t begin = start;
        while (begin < stop) {
            const uint32_t end = PagedMin(shardEnd, stop);
            if (end > begin) {
                const uint64_t work = begin == start && end == stop ? requestWork :
                    EstimateScoreSpan<CAUSAL>(tiling, end - begin, position - (stop - end));
                gridLoad += ShareScoreGridWork(work, tiling.kShardCount);
                begin = end;
            }
            if (begin < shardEnd || shard + 1U >= tiling.qShardCount) {
                break;
            }
            gridPeak = gridLoad > gridPeak ? gridLoad : gridPeak;
            gridLoad = 0U;
            shardEnd = ScoreGridQEnd(tiling, all, ++shard + 1U);
        }
    }
    gridPeak = gridLoad > gridPeak ? gridLoad : gridPeak;
    return gridPeak > ShareScoreGridWork(total, tiling.usedCubeCores) + planningCost;
}

template <bool CAUSAL>
__aicore__ inline bool ShouldPrepareScorePartition(
    const PagedTilingData &tiling, GM_ADDR cu, GM_ADDR kv, uint32_t qBand)
{
    const uint32_t pairs = tiling.usedCubeCores;
    if (pairs <= 1U) {
        return false;
    }
    const uint64_t qTiles = (static_cast<uint64_t>(tiling.totalTokens) + tiling.qTile - 1U) / tiling.qTile;
    const uint64_t maxQ = (qTiles + tiling.qShardCount - 1U) / tiling.qShardCount * tiling.qTile;
    const uint64_t kMacros = (tiling.seqK + PAGED_K_MACRO - 1U) / PAGED_K_MACRO;
    const uint64_t maxN = ((kMacros + tiling.kShardCount - 1U) / tiling.kShardCount) *
        (PAGED_K_MACRO / PAGED_NR);
    if (tiling.numRequests == 1U && tiling.kShardCount == 1U && kMacros <= 1U && maxQ <= qBand) {
        // Each original pair owns at most one Q band and one K macro. Its
        // entire cost fits the cut-rounding/reload allowance in PreferScoreCuts,
        // so that predicate cannot select weighted ownership for any KV length.
        // The same bound covers every smaller bounded-score epoch.
        return false;
    }
    const uint64_t meanQ = (static_cast<uint64_t>(tiling.totalTokens) + tiling.numRequests - 1U) /
        tiling.numRequests;
    const uint64_t qFragments = (maxQ + meanQ - 1U) / meanQ;
    const uint64_t planningCost = 4U * pairs + (tiling.numRequests + 1U) / 2U;
    if (tiling.numRequests == 1U) {
        // An exactly tiled rectangle already attains the grid lower bound.
        if constexpr (!CAUSAL) {
            if (tiling.totalTokens % qBand == 0U && qTiles % tiling.qShardCount == 0U &&
                tiling.seqK % PAGED_K_MACRO == 0U && kMacros % tiling.kShardCount == 0U) {
                return false;
            }
        }
        return true;
    }
    const uint64_t mTiles = (maxQ * tiling.headsPerGroup + 15U) / 16U;
    if (maxN * (mTiles * tiling.groups + 2U * qFragments) <= planningCost) {
        // Even the largest estimated grid shard is in the range of the
        // shared-planning cost. Keep the original loop without a rendezvous.
        return false;
    }
    // Small metadata scans are cheaper than publishing a full solve. Larger
    // request lists stay with the shared planner and are never rescanned by
    // every pair.
    if (tiling.numRequests <= SCORE_METADATA_CHUNK / 4U && cu != nullptr) {
        return EstimateSmallScoreGrid<CAUSAL>(tiling, cu, kv, planningCost);
    }
    return true;
}

} // namespace IndexerRegionScoreWeighted

#endif
