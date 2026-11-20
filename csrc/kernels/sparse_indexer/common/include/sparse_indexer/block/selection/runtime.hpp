/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_SELECTION_RUNTIME_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_SELECTION_RUNTIME_HPP

#include <cstdint>

namespace SparseIndexer::Common::Block {

struct SelectionRuntimeParams {
    uint32_t topK{0};
    uint32_t s2Stride{0};
    uint32_t s2Tile{0};
    bool hasOutputIdxOffset{false};
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_SELECTION_RUNTIME_HPP
