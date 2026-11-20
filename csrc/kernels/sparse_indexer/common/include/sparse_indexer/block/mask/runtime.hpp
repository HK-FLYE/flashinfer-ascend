/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_MASK_RUNTIME_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_MASK_RUNTIME_HPP

#include <cstdint>

namespace SparseIndexer::Common::Block {

struct MaskRuntimeParams {
    int32_t maskMode{0};
    uint32_t cmpRatio{1};
    uint32_t s2Tile{1};
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_MASK_RUNTIME_HPP
