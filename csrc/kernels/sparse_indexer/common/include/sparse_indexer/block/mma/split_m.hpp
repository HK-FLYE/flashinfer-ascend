/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_MMA_SPLIT_M_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_MMA_SPLIT_M_HPP

#include "sparse_indexer/block/mma/block_mmad.hpp"

namespace SparseIndexer::Common::Block {

/**
 * QLIv2-style row split. M is the flattened query x head axis, so the half-M
 * boundary is exactly 2 * N1: vector0 owns query rows 0..1 and vector1 owns
 * rows 2..3. The alignment is deliberately not a fixed four M rows.
 */
template <class Dtype_, class LoadPolicy_>
struct SplitMMmadPolicy {
    using Dtype = Dtype_;
    using LoadPolicy = LoadPolicy_;
    using InputType = typename LoadPolicy::Dtype;
    using Block = SparseIndexer::Common::Block::BlockMmad<Dtype, Dtype, typename LoadPolicy::Loader,
        Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using Params = typename Block::Params;
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_MMA_SPLIT_M_HPP
