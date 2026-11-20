/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_MMA_RUNTIME_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_MMA_RUNTIME_HPP

#include <cstdint>

#include "catlass/gemm_coord.hpp"
#include "sparse_indexer/block/mask/runtime.hpp"

namespace SparseIndexer::Common::Block {

template <class Block_>
struct MmaRuntimeParams {
    using Block = Block_;
    typename Block::Params block{};
    Catlass::GemmCoord problemShape{};
    Catlass::GemmCoord tileShape{};
    uint32_t heads{0};
    uint32_t scoreBuffers{1};
    uint32_t scoreElemsPerAiv{0};
    MaskRuntimeParams mask{};
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_MMA_RUNTIME_HPP
