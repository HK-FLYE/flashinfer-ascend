#ifndef IRS_DEVICE_DISPATCH_HPP
#define IRS_DEVICE_DISPATCH_HPP
#include <type_traits>

namespace IndexerRegionScorePaged {

struct DispatchShape {
    uint32_t numRequests = 0U;
    uint32_t totalTokens = 0U;
    uint32_t qShardCount = 0U;
    uint32_t qTile = 0U;
    uint32_t batchTokens = 0U;
};

// Every participant reads the same immutable metadata and reaches the same branch.
// Only equal complete requests with matching grid boundaries are certified here.
template <bool CAUSAL>
__aicore__ inline bool IsUniformRequestGrid(const DispatchShape &t, GM_ADDR cu, GM_ADDR kv)
{
    if (t.numRequests == 0U || cu == nullptr || t.qShardCount == 0U || t.qTile == 0U ||
        t.totalTokens % t.numRequests != 0U || t.numRequests % t.qShardCount != 0U) {
        return false;
    }
    if (t.batchTokens != 0U && t.batchTokens < t.totalTokens) {
        return false;
    }
    const uint32_t rows = t.totalTokens / t.numRequests;
    if (rows == 0U || rows % t.qTile != 0U) {
        return false;
    }
    // A bounded exact scan avoids replicating a large-B metadata pass.
    if (t.numRequests > 64U) {
        return false;
    }
    const auto bounds = reinterpret_cast<__gm__ const int32_t *>(cu);
    const auto lengths = reinterpret_cast<__gm__ const int32_t *>(kv);
    if (bounds[0] != 0) {
        return false;
    }
    int32_t firstKv = 0;
    if constexpr (CAUSAL) {
        if (kv == nullptr) {
            return false;
        }
        firstKv = lengths[0];
    }
    uint32_t expected = 0U;
    for (uint32_t request = 0U; request < t.numRequests; ++request) {
        expected += rows;
        if (bounds[request + 1U] != static_cast<int32_t>(expected)) {
            return false;
        }
        if constexpr (CAUSAL) {
            if (lengths[request] != firstKv) {
                return false;
            }
        }
    }
    return true;
}

template <typename CacheStrategy, typename MaskStrategy, typename SelectionStrategy, bool ALLOW_H4_TAIL = false>
__aicore__ inline void RunWithStrategies(GM_ADDR query, GM_ADDR weights, GM_ADDR keySummary,
    GM_ADDR actualKvLengths, GM_ADDR pagedKeySummary, GM_ADDR blockTable, GM_ADDR cuSeqLens,
    GM_ADDR output, GM_ADDR selectedRegionIds, GM_ADDR workspace, GM_ADDR tiling,
    AscendC::TPipe *pipe)
{
    DispatchShape decoded{};
#if IRS_DEVICE_DISPATCH_MODE == 3 || (defined(IRS_DEVICE_DISPATCH_TRACE) && IRS_DEVICE_DISPATCH_TRACE)
    if constexpr (CacheStrategy::IS_PAGED) {
        decoded.numRequests = reinterpret_cast<__gm__ const uint32_t *>(tiling)[offsetof(PagedTilingData, numRequests) / sizeof(uint32_t)];
    } else {
        decoded.numRequests = reinterpret_cast<__gm__ const uint32_t *>(tiling)[offsetof(RegionScoreTilingData, numRequests) / sizeof(uint32_t)];
    }
#elif IRS_DEVICE_DISPATCH_MODE == 1
    if constexpr (CacheStrategy::IS_PAGED) {
        decoded.numRequests = reinterpret_cast<__gm__ const uint32_t *>(tiling)[offsetof(PagedTilingData, numRequests) / sizeof(uint32_t)];
    }
#endif
#if IRS_DEVICE_DISPATCH_MODE == 1
    if constexpr (CacheStrategy::IS_PAGED) {
        decoded.totalTokens = reinterpret_cast<__gm__ const uint32_t *>(tiling)[offsetof(PagedTilingData, totalTokens) / sizeof(uint32_t)];
        decoded.qShardCount = reinterpret_cast<__gm__ const uint32_t *>(tiling)[offsetof(PagedTilingData, qShardCount) / sizeof(uint32_t)];
        decoded.qTile = reinterpret_cast<__gm__ const uint32_t *>(tiling)[offsetof(PagedTilingData, qTile) / sizeof(uint32_t)];
        decoded.batchTokens = reinterpret_cast<__gm__ const uint32_t *>(tiling)[offsetof(PagedTilingData, batchTokens) / sizeof(uint32_t)];
    } else {
        // Continuous qTile is derived by NormalizeContinuousTiling, not copied from its POD qTile field.
        GET_TILING_DATA_WITH_STRUCT(RegionScoreTilingData, data, tiling);
        const auto normalized = NormalizeContinuousTiling(data);
        decoded = {normalized.numRequests, normalized.totalTokens, normalized.qShardCount,
            normalized.qTile, normalized.batchTokens};
    }
#endif
#if IRS_DEVICE_DISPATCH_MODE == 0
    const bool direct = true;
#elif IRS_DEVICE_DISPATCH_MODE == 2
    const bool direct = false;
#elif IRS_DEVICE_DISPATCH_MODE == 3
    // Keep both code paths linked while forcing grid for every valid request count.
    const bool direct = decoded.numRequests != 0U;
#else
    const bool direct = IsUniformRequestGrid<MaskStrategy::CAUSAL>(decoded, cuSeqLens, actualKvLengths);
#endif
#if defined(IRS_DEVICE_DISPATCH_TRACE) && IRS_DEVICE_DISPATCH_TRACE
    if ASCEND_IS_AIC {
        if (GetBlockIdx() == 0U) {
            AscendC::PRINTF("[IRS dispatch] mode=%u direct=%u requests=%u\n",
                IRS_DEVICE_DISPATCH_MODE, static_cast<uint32_t>(direct), decoded.numRequests);
        }
    }
#endif
    if (direct) {
        RunGridWithStrategies<CacheStrategy, MaskStrategy, SelectionStrategy, ALLOW_H4_TAIL>(
            query, weights, keySummary, actualKvLengths, pagedKeySummary, blockTable,
            cuSeqLens, output, selectedRegionIds, workspace, tiling, pipe);
        return;
    }
    using WeightedMask = typename std::conditional<MaskStrategy::CAUSAL,
        IndexerRegionScoreWeighted::RightDownCausalMaskStrategy,
        IndexerRegionScoreWeighted::NoMaskStrategy>::type;
    if ASCEND_IS_AIC {
        IndexerRegionScoreWeighted::RunCubeWithStrategies<CacheStrategy, WeightedMask, SelectionStrategy, ALLOW_H4_TAIL>(
            query, weights, keySummary, actualKvLengths, pagedKeySummary, blockTable,
            cuSeqLens, output, selectedRegionIds, workspace, tiling, pipe);
    } else {
        IndexerRegionScoreWeighted::RunVectorWithStrategies<CacheStrategy, WeightedMask, SelectionStrategy, ALLOW_H4_TAIL>(
            query, weights, keySummary, actualKvLengths, pagedKeySummary, blockTable,
            cuSeqLens, output, selectedRegionIds, workspace, tiling, pipe);
    }
}
} // namespace IndexerRegionScorePaged
#endif
