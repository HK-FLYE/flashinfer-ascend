/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_LOAD_PAGED_KEY_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_LOAD_PAGED_KEY_HPP

#include "block/key_loader.hpp"

namespace SparseIndexer::QuantLightningIndexer::Policy {

/**
 * PA_BBND key loader policy. The load policy owns the key address mapping and
 * is composed into the mmad policy instead of being selected inside the kernel.
 */
template <class Dtype_>
struct PagedKeyLoadPolicy {
    using Dtype = Dtype_;
    using Loader = SparseIndexer::Common::Block::PagedKeyLoader<
        Catlass::Arch::Ascend950, Catlass::Gemm::GemmType<Dtype, Catlass::layout::ColumnMajor>>;
};

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_LOAD_PAGED_KEY_HPP
