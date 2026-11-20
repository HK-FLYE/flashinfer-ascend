/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_KERNEL_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_KERNEL_HPP

#include <type_traits>

#include "catlass/arch/resource.hpp"
#include "catlass/detail/macros.hpp"
#include "kernel_operator.h"
#include "sparse_indexer/foundation/contracts/policy.hpp"

namespace SparseIndexer::Common::Pipeline {

/**
 * Static Indexer executor. Schedule/load/mmad/reduce/topk/sync/resource are
 * compile-time types; there are no virtual calls or runtime policy switches.
 */
template <class Policies_, class Tiling_>
class IndexerKernel {
    using Tiling = Tiling_;
    static_assert(IsPolicyBundle<Policies_>::value, "IndexerKernel requires a PolicyBundle");

    using Schedule = typename Policies_::Schedule;
    using Load = typename Policies_::Load;
    using Mmad = typename Policies_::Mmad;
    using Reduce = typename Policies_::Reduce;
    using TopK = typename Policies_::TopK;
    using Sync = typename Policies_::Sync;
    using ResourcePolicy = typename Policies_::Resource;
    using ScheduleParams = typename Policies_::ScheduleRuntimeParams;
    using MmaParams = typename Policies_::MmaRuntimeParams;
    using ReduceParams = typename Policies_::ReduceRuntimeParams;
    using SelectionParams = typename Policies_::SelectionRuntimeParams;
    using InputType = typename Mmad::InputType;
    using DtypeC = typename Mmad::Block::DtypeC;

    static_assert(SparseIndexer::Common::HasSchedulePolicy<Schedule>::value,
        "SchedulePolicy must expose Scheduler/Params/Coord/Task");
    static_assert(std::is_same<typename Schedule::Coord, SparseIndexer::Common::WorkItem>::value,
        "SchedulePolicy Coord must be the common WorkItem");
    static_assert(SparseIndexer::Common::HasLoadPolicy<Load>::value,
        "LoadPolicy must expose Loader/Dtype");
    static_assert(SparseIndexer::Common::HasMmadPolicy<Mmad>::value,
        "MmadPolicy must expose InputType/Block/LoadPolicy");
    static_assert(SparseIndexer::Common::HasReducePolicy<Reduce>::value,
        "ReducePolicy must expose Block");
    static_assert(SparseIndexer::Common::HasSelectionPolicy<TopK>::value,
        "SelectionPolicy must expose VectorBlock");
    static_assert(SparseIndexer::Common::HasSyncPolicy<Sync>::value,
        "SyncPolicy must expose the cube/vector hand-off surface");
    static_assert(SparseIndexer::Common::HasResourcePolicy<ResourcePolicy>::value,
        "ResourcePolicy must expose Resource");
    static_assert(std::is_same<Load, typename Mmad::LoadPolicy>::value,
        "the bundle Load policy must be the policy embedded in the Mmad pipeline");

public:
    CATLASS_DEVICE
    IndexerKernel()
        : ubC_(ResourcePolicy::template GetUbC<DtypeC>(resource_))
    {
    }

    CATLASS_DEVICE void operator()(
        AscendC::GlobalTensor<InputType> const& gmQ, AscendC::GlobalTensor<InputType> const& gmK,
        AscendC::GlobalTensor<float> const& gmW, AscendC::GlobalTensor<float> const& gmQScale,
        AscendC::GlobalTensor<float> const& gmKScale, AscendC::GlobalTensor<int32_t> const& outIndices,
        AscendC::GlobalTensor<float>& scores,
        __gm__ Tiling const* tiling, GM_ADDR cuSeqlensQ, GM_ADDR sequsedK,
        GM_ADDR cmpResidualK, AscendC::GlobalTensor<int32_t> const& gmBlockTable,
        AscendC::GlobalTensor<InputType> const& gmZeroPage)
    {
        ScheduleParams scheduleParams =
            Policies_::MakeScheduleParams(tiling, cuSeqlensQ, sequsedK, cmpResidualK);
        typename Schedule::Scheduler blockScheduler(scheduleParams);

        if ASCEND_IS_AIC {
            MmaParams mmaParams = Policies_::MakeMmaParams(tiling);
            Mmad::template Run<Sync, typename Schedule::Scheduler, Schedule>(
                resource_, ubC_, gmQ, gmK, mmaParams, gmBlockTable, gmZeroPage, blockScheduler);
        } else {
            ReduceParams reduceParams = Policies_::MakeReduceParams(tiling);
            SelectionParams selectionParams = Policies_::MakeSelectionParams(tiling);
            Reduce::template Run<Sync, typename Schedule::Scheduler, Schedule, TopK>(
                resource_, ubC_, gmW, gmQScale, gmKScale, gmBlockTable, scores, reduceParams,
                selectionParams, blockScheduler, outIndices);
        }

        Sync::GlobalBarrier();
    }

    typename ResourcePolicy::Resource resource_;
    AscendC::LocalTensor<DtypeC> ubC_;
};

} // namespace SparseIndexer::Common::Pipeline

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_KERNEL_HPP
