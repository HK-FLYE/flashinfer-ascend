/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_CORE_SCORE_BUFFER_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_CORE_SCORE_BUFFER_HPP

#include "sparse_indexer/block/schedule/block_scheduler.hpp"
#include "sparse_indexer/block/mask/mask_utils.hpp"
#include "quant_lightning_indexer_tiling_data.h"
#include "sparse_indexer/foundation/contracts/common.hpp"

namespace SparseIndexer::QuantLightningIndexer::Policy {

/// One AIV owns half the rows of a gS1 tile.
CATLASS_DEVICE inline uint32_t ScoreElementsPerAiv(
    __gm__ QuantLightningIndexerTilingData const* tiling)
{
    const uint32_t workRows = SparseIndexer::Common::CeilDiv(tiling->s1Tile, 2U);
    const uint32_t colPerAiv = SparseIndexer::Common::AlignUp(tiling->s2Tile, 8U);
    return workRows * tiling->n1 * colPerAiv;
}

/// Two score tiles make the cube/vector hand-off a ping-pong pipeline. Keep
/// enough UB for the reducer scratch before enabling it; pathological large
/// head counts retain the single-buffer path.
CATLASS_DEVICE inline uint32_t ScoreBufferCount(
    __gm__ QuantLightningIndexerTilingData const* tiling)
{
    constexpr uint64_t kUbSize = 248U * 1024U;
    constexpr uint64_t kScratchReserve = 64U * 1024U;
    const uint64_t twoScores = 2ULL * ScoreElementsPerAiv(tiling) * sizeof(float);
    return twoScores + kScratchReserve <= kUbSize ? 2U : 1U;
}

/// Causal mask3 only needs the prefix visible to the last row of a gS1 tile.
/// Keep one tile for an empty-visible request so the CV hand-off remains paired.
CATLASS_DEVICE inline SparseIndexer::Common::Block::ScheduleCoord VisibleCoord(
    __gm__ QuantLightningIndexerTilingData const* tiling,
    SparseIndexer::Common::Block::ScheduleCoord const& coord,
    uint32_t activeQ, uint32_t activeK, int32_t residual)
{
    auto visible = coord;
    if (tiling->maskMode != 3 || coord.qRows == 0U) {
        return visible;
    }
    const uint32_t lastLocalQ = coord.localQBegin + coord.qRows - 1U;
    const uint32_t visibleK = SparseIndexer::Common::Block::Mask3VisibleLen(
        lastLocalQ, activeQ, activeK, tiling->cmpRatio, residual);
    visible.kRows = AscendC::Std::max(
        visibleK, AscendC::Std::min(coord.kRows, tiling->s2Tile));
    return visible;
}

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_CORE_SCORE_BUFFER_HPP
