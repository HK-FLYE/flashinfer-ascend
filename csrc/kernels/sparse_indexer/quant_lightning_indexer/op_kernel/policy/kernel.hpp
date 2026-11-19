/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_KERNEL_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_KERNEL_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/detail/macros.hpp"
#include "kernel_operator.h"
#include "quant_lightning_indexer_tiling_data.h"
#include "core/traits.hpp"

/**
 * Static policy composer. Schedule/load/mmad/reduce/topk/sync/resource are
 * compile-time types; there are no virtual calls or runtime policy switches.
 */
template <class Policies_>
class QuantLightningIndexerKernel {
    static_assert(SparseIndexer::QuantLightningIndexer::Policy::IsPolicyBundle<Policies_>::value,
        "QuantLightningIndexerKernel requires a PolicyBundle");

    using Schedule = typename Policies_::Schedule;
    using Mmad = typename Policies_::Mmad;
    using Reduce = typename Policies_::Reduce;
    using TopK = typename Policies_::TopK;
    using Sync = typename Policies_::Sync;
    using ResourcePolicy = typename Policies_::Resource;
    using InputType = typename Mmad::InputType;
    using DtypeC = typename Mmad::Block::DtypeC;

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
        AscendC::GlobalTensor<float> const& outValues,
        AscendC::GlobalTensor<bfloat16_t> const& publicValues, AscendC::GlobalTensor<float>& scores,
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
            if constexpr (Policies_::Traits::kTopKPerGSA1) {
                Reduce::template Run<Sync, typename Schedule::Scheduler, Schedule>(
                    resource_, ubC_, gmW, gmQScale, gmKScale, gmBlockTable, scores, tiling, blockScheduler,
                    outIndices, outValues);
            } else {
                Reduce::template Run<Sync, typename Schedule::Scheduler, Schedule>(
                    resource_, ubC_, gmW, gmQScale, gmKScale, gmBlockTable, scores, tiling, blockScheduler);
            }
        }

        Sync::GlobalBarrier();
        if ASCEND_IS_AIV {
            if constexpr (Policies_::Traits::kTopKPerGSA1) {
                // The vector path performs TopK in the owning gS1 task. The
                // scalar public-value fallback still uses the global scheduler.
                if (tiling->returnValue != 0) {
                    TopK::template Run<typename Schedule::Scheduler, Schedule>(
                        resource_, blockScheduler, tiling, scores, outIndices, outValues);
                }
            } else {
                TopK::template Run<typename Schedule::Scheduler, Schedule>(
                    resource_, blockScheduler, tiling, scores, outIndices, outValues);
            }

            if (tiling->returnValue != 0) {
                // TopK writes float scratch; the public output is BF16.
                // Only AIVs participate in this second barrier (AIC has
                // already finished its cube work).
                AscendC::SyncAll<true>();
                if (AscendC::GetBlockIdx() == 0) {
                    CopyValuesToPublic(outValues, publicValues,
                        static_cast<uint64_t>(tiling->tq) * tiling->topk);
                }
            }
        }
    }

private:
    CATLASS_DEVICE void CopyValuesToPublic(AscendC::GlobalTensor<float> const& src,
        AscendC::GlobalTensor<bfloat16_t> const& dst, uint64_t count)
    {
        constexpr uint32_t kChunk = 4096U;
        auto ubFloat = resource_.ubBuf.template GetBufferByByte<float>(0);
        auto ubBf16 = resource_.ubBuf.template GetBufferByByte<bfloat16_t>(kChunk * sizeof(float));
        for (uint64_t off = 0; off < count; off += kChunk) {
            const uint64_t remaining = count - off;
            const uint32_t n = static_cast<uint32_t>(remaining < kChunk ? remaining : kChunk);
            AscendC::DataCopyExtParams copyIn{
                1, static_cast<uint32_t>(n * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(ubFloat, src[off], copyIn, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(ubBf16, ubFloat, AscendC::RoundMode::CAST_RINT, n);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopyExtParams copyOut{
                1, static_cast<uint32_t>(n * sizeof(bfloat16_t)), 0, 0, 0};
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(dst[off], ubBf16, copyOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    typename ResourcePolicy::Resource resource_;
    AscendC::LocalTensor<DtypeC> ubC_;
};

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_KERNEL_HPP
