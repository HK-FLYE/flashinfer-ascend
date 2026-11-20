/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_MASK_VISIBLE_COORD_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_MASK_VISIBLE_COORD_HPP

#include "catlass/detail/macros.hpp"
#include "kernel_operator.h"
#include "sparse_indexer/block/mask/mask_utils.hpp"
#include "sparse_indexer/block/mask/runtime.hpp"
#include "sparse_indexer/foundation/contracts/problem.hpp"

namespace SparseIndexer::Common::Block {

/**
 * Causal mask3 only needs the prefix visible to the last row of a gS1 tile.
 * Keep one tile for an empty-visible request so the CV hand-off stays paired.
 */
CATLASS_DEVICE inline WorkItem VisibleCoord(MaskRuntimeParams const& mask, WorkItem const& coord,
    uint32_t activeQ, uint32_t activeK, int32_t residual)
{
    auto visible = coord;
    if (mask.maskMode != 3 || coord.qRows == 0U) {
        return visible;
    }
    const uint32_t lastLocalQ = coord.localQBegin + coord.qRows - 1U;
    const uint32_t visibleK = Mask3VisibleLen(lastLocalQ, activeQ, activeK, mask.cmpRatio, residual);
    visible.kRows = AscendC::Std::max(visibleK, AscendC::Std::min(coord.kRows, mask.s2Tile));
    return visible;
}

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_MASK_VISIBLE_COORD_HPP
