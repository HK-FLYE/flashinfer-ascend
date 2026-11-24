#ifndef INDEXER_REGION_SCORE_BLOCK_TOPK_HPP
#define INDEXER_REGION_SCORE_BLOCK_TOPK_HPP

#include <cstdint>
#include "stage_registry.hpp"
#include "kernel_operator.h"
#include "sync/subblock_barrier.hpp"
#include "../detail/topk_selection.hpp"

namespace IndexerRegionScore::Block {

template <class Tiling>
struct TopKArguments {
    GM_ADDR output;
    GM_ADDR actualKvLengths;
    GM_ADDR cuSeqLens;
    GM_ADDR scratch;
    const Tiling &tiling;
    AscendC::TPipe *pipe;
    bool alignedSelection;

    template <auto Select>
    __aicore__ inline void SelectRows() const
    {
        Select(output, actualKvLengths, cuSeqLens, scratch,
            tiling.groups, tiling.totalTokens,
            tiling.outputStrideGroup, tiling.outputStrideToken,
            static_cast<uint32_t>(tiling.outputStrideToken),
            tiling.seqK, tiling.selectionConfig & 0xffffU,
            tiling.maskMode, tiling.regionSize,
            (tiling.selectionConfig & 0x10000U) != 0U,
            tiling.numRequests, tiling.usedCubeCores, pipe,
            alignedSelection, tiling.batchBegin, tiling.batchCount);
    }
};

template <class SelectionPolicy, auto Compact>
struct BlockTopK {
    template <class Tiling>
    __aicore__ static inline void Run(GM_ADDR output, GM_ADDR actualKvLengths,
        GM_ADDR cuSeqLens, GM_ADDR selectedRegionIds, GM_ADDR workspace,
        const Tiling &tiling, uint64_t scratchOffset, AscendC::TPipe *pipe)
    {
        SparseIndexer::Common::SubblockPairBlockBarrierSyncPolicy::GlobalBarrier();
        const bool alignedSelection = (tiling.selectionConfig & 0x20000U) != 0U;
        GM_ADDR scratch = alignedSelection ? workspace + scratchOffset : selectedRegionIds;
        if ASCEND_IS_AIV {
            pipe->Reset();
            SelectionPolicy::Select(TopKArguments<Tiling>{
                output, actualKvLengths, cuSeqLens, scratch, tiling, pipe, alignedSelection});
        }
        if (alignedSelection) {
            SparseIndexer::Common::SubblockPairBlockBarrierSyncPolicy::GlobalBarrier();
            if ASCEND_IS_AIV {
                pipe->Reset();
                Compact(scratch, selectedRegionIds, tiling.groups,
                    tiling.totalTokens, tiling.selectionConfig & 0xffffU,
                    tiling.usedCubeCores, pipe, tiling.batchBegin, tiling.batchCount);
            }
        }
    }
};

template <class SelectionPolicy, auto Compact>
struct StageImplementation<Stage::TopK, SelectionStagePolicy<SelectionPolicy, Compact>> {
    using Type = BlockTopK<SelectionPolicy, Compact>;
};

} // namespace IndexerRegionScore::Block

namespace IndexerRegionScorePaged {
struct FusedTopKPolicy {
    template <class Arguments>
    __aicore__ static inline void Select(const Arguments &arguments)
    {
        arguments.template SelectRows<IndexerRegionScore::FusedSelect::Run>();
    }
};

using FusedTopKStrategy = IndexerRegionScore::Block::StageType<
    IndexerRegionScore::Block::Stage::TopK,
    IndexerRegionScore::Block::SelectionStagePolicy<FusedTopKPolicy, IndexerRegionScore::FusedSelect::Compact>>;

} // 命名空间结束。

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && \
    defined(ORIG_DTYPE_QUERY) && defined(DT_FLOAT16) && defined(DT_BF16) && \
    ((ORIG_DTYPE_QUERY == DT_FLOAT16) || (ORIG_DTYPE_QUERY == DT_BF16))
namespace IndexerRegionScorePaged {
struct IdCompactBatchTopKPolicy {
    template <class Arguments>
    __aicore__ static inline void Select(const Arguments &arguments)
    {
        if ((arguments.tiling.selectionConfig & 0x10000U) != 0U) {
            arguments.template SelectRows<IndexerRegionScore::FusedSelect::RunIdCompactBatch>();
        } else {
            arguments.template SelectRows<IndexerRegionScore::FusedSelect::RunSingleWindow>();
        }
    }
};

using IdCompactBatchTopKStrategy = IndexerRegionScore::Block::StageType<
    IndexerRegionScore::Block::Stage::TopK,
    IndexerRegionScore::Block::SelectionStagePolicy<IdCompactBatchTopKPolicy, IndexerRegionScore::FusedSelect::Compact>>;
using Native512TopKStrategy = IdCompactBatchTopKStrategy;

} // 命名空间结束。

#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && \
    defined(ORIG_DTYPE_QUERY) && defined(DT_FLOAT16) && defined(DT_BF16) && \
    ((ORIG_DTYPE_QUERY == DT_FLOAT16) || (ORIG_DTYPE_QUERY == DT_BF16))
namespace IndexerRegionScorePaged {
struct B8Native512TopKPolicy {
    template <class Arguments>
    __aicore__ static inline void Select(const Arguments &arguments)
    {
        const auto &tiling = arguments.tiling;
        if ((tiling.selectionConfig & 0x10000U) != 0U) {
            arguments.template SelectRows<IndexerRegionScore::FusedSelect::RunIdCompactBatch>();
        } else if (IndexerRegionScore::FusedSelect::Min(
            tiling.seqK, static_cast<uint32_t>(tiling.outputStrideToken)) <=
            IndexerRegionScore::FusedSelect::kShortRowCapacity) {
            arguments.template SelectRows<IndexerRegionScore::FusedSelect::RunShortRows>();
        } else {
            arguments.template SelectRows<IndexerRegionScore::FusedSelect::RunSingleWindow>();
        }
    }
};

using B8Native512TopKStrategy = IndexerRegionScore::Block::StageType<
    IndexerRegionScore::Block::Stage::TopK,
    IndexerRegionScore::Block::SelectionStagePolicy<B8Native512TopKPolicy, IndexerRegionScore::FusedSelect::Compact>>;

}
#endif

#endif
