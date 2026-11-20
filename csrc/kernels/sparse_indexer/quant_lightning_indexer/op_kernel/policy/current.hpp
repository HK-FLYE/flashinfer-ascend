/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_CURRENT_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_CURRENT_HPP

#include "arch35/block_reduce_quant_head.hpp"
#include "sparse_indexer/block/selection/topk_vf.hpp"
#include "sparse_indexer/block/load/paged_key.hpp"
#include "mmad/split_m.hpp"
#include "reduce/inline_row_split.hpp"
#include "sparse_indexer/foundation/resource/current_policy.hpp"
#include "schedule/static_gsa1.hpp"
#include "sparse_indexer/foundation/sync/subblock_barrier.hpp"
#include "topk/prefix_litopkvf.hpp"
#include "sparse_indexer/foundation/contracts/policy.hpp"

namespace SparseIndexer::QuantLightningIndexer::Policy {

using QkDtype = fp8_e4m3fn_t;
using CommonKeyLoad = PagedKeyLoadPolicy<QkDtype>;
using C2Sync = SubblockPairBlockBarrierSyncPolicy;
using CommonResource = CurrentResourcePolicy<Catlass::Arch::Ascend950>;

// One gS1 task stays on its owning AIC/AIV pair. TopK is an explicit policy
// boundary and executes after all S2 tiles for that task have been reduced.
using InlineVectorTopK = InlinePrefixLiTopKVFPolicy<
    SparseIndexer::Common::Block::BlockTopKVector<float, int32_t, true>>;
using C2Mmad = SplitMMmadPolicy<QkDtype, CommonKeyLoad>;
using C2Reduce = InlineRowSplitReducePolicy<
    QuantLightningIndexerKernelDetail::BlockReduceQuantHead<
        SparseIndexer::Common::Block::LiV2MaskPolicy<float>, true>>;
using CurrentPolicy = SparseIndexer::Common::PolicyBundle<StaticGSA1SchedulePolicy, CommonKeyLoad, C2Mmad,
    C2Reduce, InlineVectorTopK, C2Sync, CommonResource>;

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_CURRENT_HPP
