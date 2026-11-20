/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_CONFIG_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_CONFIG_HPP

#include "arch35/quant_lightning_indexer_source.h"
#include "arch35/quant_lightning_indexer_task.h"
#include "quant_lightning_indexer_tiling_data.h"
#include "sparse_indexer/block/load/paged_key.hpp"
#include "sparse_indexer/block/mma/split_m.hpp"
#include "sparse_indexer/block/reduce/inline_row_split.hpp"
#include "sparse_indexer/block/reduce/quant_head.hpp"
#include "sparse_indexer/block/schedule/static_gsa1.hpp"
#include "sparse_indexer/block/selection/prefix_select.hpp"
#include "sparse_indexer/block/selection/topk_vf.hpp"
#include "sparse_indexer/foundation/contracts/policy.hpp"
#include "sparse_indexer/foundation/resource/current_policy.hpp"
#include "sparse_indexer/foundation/sync/subblock_barrier.hpp"

namespace SparseIndexer::QuantLightningIndexer::Config {

using QkDtype = fp8_e4m3fn_t;
using KeyLoad = SparseIndexer::Common::Block::PagedKeyLoadPolicy<QkDtype>;
using Sync = SparseIndexer::Common::SubblockPairBlockBarrierSyncPolicy;
using Resource = SparseIndexer::Common::CurrentResourcePolicy<Catlass::Arch::Ascend950>;

using Schedule = SparseIndexer::Common::Block::StaticGSA1SchedulePolicy<
    QuantLightningIndexerKernelDetail::QuantLightningIndexerRequestSource,
    QuantLightningIndexerKernelDetail::QuantLightningIndexerTaskFactory>;

using Mmad = SparseIndexer::Common::Block::SplitMMmadPolicy<
    QkDtype, KeyLoad, QuantLightningIndexerTilingData>;

using QuantHead = SparseIndexer::Common::Block::BlockReduceQuantHead<
    SparseIndexer::Common::Block::LiV2MaskPolicy<float>, true>;
using Reduce = SparseIndexer::Common::Block::InlineRowSplitReducePolicy<
    QuantHead, QuantLightningIndexerTilingData>;

using Selection = SparseIndexer::Common::Block::InlinePrefixLiTopKVFPolicy<
    SparseIndexer::Common::Block::BlockTopKVector<float, int32_t, true>,
    QuantLightningIndexerTilingData>;

using CurrentPolicy = SparseIndexer::Common::PolicyBundle<
    Schedule, KeyLoad, Mmad, Reduce, Selection, Sync, Resource>;

} // namespace SparseIndexer::QuantLightningIndexer::Config

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_CONFIG_HPP
