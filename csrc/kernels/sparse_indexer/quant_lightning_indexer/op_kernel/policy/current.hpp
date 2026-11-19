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
#include "mmad/split_m.hpp"
#include "mmad/split_n.hpp"
#include "reduce/column_split.hpp"
#include "reduce/row_split.hpp"
#include "resource/current.hpp"
#include "schedule/round_robin.hpp"
#include "schedule/static_gsa1.hpp"
#include "sync/block_barrier.hpp"
#include "topk/prefix_litopkvf.hpp"
#include "core/traits.hpp"

namespace SparseIndexer::QuantLightningIndexer::Policy {

using QkDtype = fp8_e4m3fn_t;
using CommonKeyLoad = PagedKeyLoadPolicy<QkDtype>;
using CommonTopK = PrefixLiTopKVFPolicy<
    SparseIndexer::Common::Block::BlockTopKVector<float, int32_t>,
    SparseIndexer::Common::Block::BlockTopK<float, int32_t>>;
using CommonSync = BlockBarrierSyncPolicy;
using CommonResource = CurrentResourcePolicy<Catlass::Arch::Ascend950>;

using C0Mmad = SplitNMmadPolicy<QkDtype, CommonKeyLoad>;
using C0Reduce = ColumnSplitReducePolicy<QuantLightningIndexerKernelDetail::BlockReduceQuantHead<>>;
using C0Policy = PolicyBundle<RoundRobinSchedulePolicy, CommonKeyLoad, C0Mmad, C0Reduce, CommonTopK,
    CommonSync, CommonResource, CurrentPerformanceTraits>;

using C1Mmad = SplitMMmadPolicy<QkDtype, CommonKeyLoad>;
using C1Reduce = RowSplitReducePolicy<
    QuantLightningIndexerKernelDetail::BlockReduceQuantHead<
        SparseIndexer::Common::Block::LiV2MaskPolicy<float>, true>>;
using C1Policy = PolicyBundle<RoundRobinSchedulePolicy, CommonKeyLoad, C1Mmad, C1Reduce, CommonTopK,
    CommonSync, CommonResource, RowSplitPerformanceTraits>;

// C2 is a compile-time experiment. The naive gS1/S2 inner loop currently
// re-loads Q for every S2 tile and regresses long-K cases; keep C1 as the
// production default until the load/pipeline policies are implemented.
using C2Policy = PolicyBundle<StaticGSA1SchedulePolicy, CommonKeyLoad, C1Mmad, C1Reduce, CommonTopK,
    CommonSync, CommonResource, StaticGSA1PerformanceTraits>;

using CurrentPolicy = C1Policy;

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_CURRENT_HPP
