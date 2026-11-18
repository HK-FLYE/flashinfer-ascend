/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_CURRENT_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_CURRENT_HPP

#include "arch35/block_reduce_quant_head.hpp"
#include "block/block_topk.hpp"
#include "block/block_topk_vf.hpp"
#include "load/paged_key.hpp"
#include "mmad/split_n.hpp"
#include "reduce/column_split.hpp"
#include "resource/current.hpp"
#include "schedule/round_robin.hpp"
#include "sync/block_barrier.hpp"
#include "topk/prefix_litopkvf.hpp"
#include "core/traits.hpp"

namespace SparseIndexer::QuantLightningIndexer::Policy {

using QkDtype = fp8_e4m3fn_t;
using C0KeyLoad = PagedKeyLoadPolicy<QkDtype>;
using C0Mmad = SplitNMmadPolicy<QkDtype, C0KeyLoad>;
using C0Reduce = ColumnSplitReducePolicy<QuantLightningIndexerKernelDetail::BlockReduceQuantHead<>>;
using C0TopK = PrefixLiTopKVFPolicy<
    SparseIndexer::Common::Block::BlockTopKVector<float, int32_t>,
    SparseIndexer::Common::Block::BlockTopK<float, int32_t>>;
using CurrentPolicy = PolicyBundle<RoundRobinSchedulePolicy, C0KeyLoad, C0Mmad, C0Reduce, C0TopK,
    BlockBarrierSyncPolicy, CurrentResourcePolicy<Catlass::Arch::Ascend950>, CurrentPerformanceTraits>;

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_CURRENT_HPP
