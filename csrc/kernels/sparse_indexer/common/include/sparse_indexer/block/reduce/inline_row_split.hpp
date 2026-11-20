/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_BLOCK_REDUCE_INLINE_ROW_SPLIT_HPP
#define FLASHINFER_SPARSE_INDEXER_BLOCK_REDUCE_INLINE_ROW_SPLIT_HPP

namespace SparseIndexer::Common::Block {

/**
 * Reduce policy descriptor for the inline row-split epilogue. The pipeline
 * owns iteration and vector ownership; the block owns tile reduction.
 */
template <class Block_>
struct InlineRowSplitReducePolicy {
    using Block = Block_;
    using Params = typename Block::Params;
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_BLOCK_REDUCE_INLINE_ROW_SPLIT_HPP
