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
#include "quant_lightning_indexer_tiling_data.h"
#include "core/policy_bundle.hpp"

/**
 * Static policy composer. Schedule/load/mmad/reduce/topk/sync/resource are
 * compile-time types; there are no virtual calls or runtime policy switches.
 */
template <class Policies_>
class QuantLightningIndexerKernel {
    static_assert(SparseIndexer::QuantLightningIndexer::Policy::IsPolicyBundle<Policies_>::value,
        "QuantLightningIndexerKernel requires a PolicyBundle");

    using Schedule = typename Policies_::Schedule;
    using Load = typename Policies_::Load;
    using Mmad = typename Policies_::Mmad;
    using Reduce = typename Policies_::Reduce;
    using TopK = typename Policies_::TopK;
    using Sync = typename Policies_::Sync;
    using ResourcePolicy = typename Policies_::Resource;
    using InputType = typename Mmad::InputType;
    using DtypeC = typename Mmad::Block::DtypeC;

    static_assert(std::is_same<Load, typename Mmad::LoadPolicy>::value,
        "the bundle Load policy must be the policy embedded in the Mmad pipeline");

public:
    CATLASS_DEVICE
    QuantLightningIndexerKernel()
        : ubC_(ResourcePolicy::template GetUbC<DtypeC>(resource_))
    {
    }

    CATLASS_DEVICE void operator()(
        AscendC::GlobalTensor<InputType> const& gmQ, AscendC::GlobalTensor<InputType> const& gmK,
        AscendC::GlobalTensor<float> const& gmW, AscendC::GlobalTensor<float> const& gmQScale,
        AscendC::GlobalTensor<float> const& gmKScale, AscendC::GlobalTensor<int32_t> const& outIndices,
        AscendC::GlobalTensor<float>& scores,
        __gm__ QuantLightningIndexerTilingData const* tiling, GM_ADDR cuSeqlensQ, GM_ADDR sequsedK,
        GM_ADDR cmpResidualK, AscendC::GlobalTensor<int32_t> const& gmBlockTable,
        AscendC::GlobalTensor<InputType> const& gmZeroPage)
    {
        typename Schedule::Params schedulerParams =
            Schedule::MakeParams(tiling, cuSeqlensQ, sequsedK, cmpResidualK);
        typename Schedule::Scheduler blockScheduler(schedulerParams);

        if ASCEND_IS_AIC {
            Mmad::template Run<Sync, typename Schedule::Scheduler, Schedule>(
                resource_, ubC_, gmQ, gmK, tiling, gmBlockTable, gmZeroPage, blockScheduler);
        } else {
            Reduce::template Run<Sync, typename Schedule::Scheduler, Schedule, TopK>(
                resource_, ubC_, gmW, gmQScale, gmKScale, gmBlockTable, scores, tiling, blockScheduler,
                outIndices);
        }

        Sync::GlobalBarrier();
    }

    typename ResourcePolicy::Resource resource_;
    AscendC::LocalTensor<DtypeC> ubC_;
};

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_KERNEL_HPP
