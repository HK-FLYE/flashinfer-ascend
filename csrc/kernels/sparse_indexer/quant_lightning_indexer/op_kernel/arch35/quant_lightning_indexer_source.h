/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_ARCH35_SOURCE_H
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_ARCH35_SOURCE_H

#include <cstdint>

#include "kernel_operator.h"
#include "quant_lightning_indexer_tiling_data.h"
#include "sparse_indexer_common.hpp"
#include "sparse_indexer_problem.hpp"

namespace QuantLightningIndexerKernelDetail {

class QuantLightningIndexerRequestSource {
public:
    static constexpr uint32_t kMaxBatch = SparseIndexer::Common::kMaxBatch;

    CATLASS_DEVICE void InitFromGm(
        __gm__ QuantLightningIndexerTilingData const* tiling, GM_ADDR cuSeqlensQ, GM_ADDR sequsedK,
        GM_ADDR cmpResidualK)
    {
        tq_ = tiling->tq;
        maxSeqLenK_ = tiling->maxSeqLenK;
        batch_ = tiling->batch == 0 ? 1U : tiling->batch;
        if (batch_ > kMaxBatch) batch_ = kMaxBatch;

        AscendC::GlobalTensor<int32_t> gmCuQ;
        gmCuQ.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cuSeqlensQ));
        int64_t previousQ = 0;
        for (uint32_t i = 0; i <= batch_; ++i) {
            int64_t value = i == 0 ? 0 : static_cast<int64_t>(gmCuQ.GetValue(i));
            value = value < previousQ ? previousQ : value;
            value = value > static_cast<int64_t>(tq_) ? static_cast<int64_t>(tq_) : value;
            cuQ_[i] = static_cast<int32_t>(value);
            previousQ = value;
        }
        // The public contract requires the prefix sum to end at Tq. If malformed
        // metadata arrived through a direct ACLNN call, make the last batch cover
        // the remaining rows rather than wrapping an unsigned length.
        cuQ_[batch_] = static_cast<int32_t>(tq_);

        AscendC::GlobalTensor<int32_t> gmSeqK;
        gmSeqK.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequsedK));
        AscendC::GlobalTensor<int32_t> gmResidual;
        gmResidual.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cmpResidualK));
        const int32_t residualLimit = tiling->cmpRatio > 0 ? tiling->cmpRatio : 1;
        for (uint32_t i = 0; i < batch_; ++i) {
            int64_t seqK = gmSeqK.GetValue(i);
            seqK = seqK < 0 ? 0 : seqK;
            seqK = seqK > static_cast<int64_t>(maxSeqLenK_) ? static_cast<int64_t>(maxSeqLenK_) : seqK;
            sequsedK_[i] = static_cast<int32_t>(seqK);

            int64_t residual = gmResidual.GetValue(i);
            residual = residual < 0 ? 0 : residual;
            residual = residual >= residualLimit ? residualLimit - 1 : residual;
            cmpResidualK_[i] = static_cast<int32_t>(residual);
        }

        problem_.batch = batch_;
        problem_.totalQ = tq_;
        problem_.groups = 1U;
        problem_.heads = tiling->n1;
        problem_.headDim = tiling->d;
        problem_.totalK = maxSeqLenK_;
        problem_.cacheKind = SparseIndexer::Common::IndexerCacheKind::Paged;
        problem_.maskKind = SparseIndexer::Common::IndexerMaskKind::CompressedMask3;
        problem_.outputKind = SparseIndexer::Common::IndexerOutputKind::IndicesOnly;
        problem_.selectionOrder = SparseIndexer::Common::IndexerSelectionOrder::QliRawSlot;
    }

    CATLASS_DEVICE uint32_t Batch() const { return batch_; }
    CATLASS_DEVICE uint32_t QBegin(uint32_t b) const { return static_cast<uint32_t>(cuQ_[b]); }
    CATLASS_DEVICE uint32_t QLen(uint32_t b) const { return static_cast<uint32_t>(cuQ_[b + 1] - cuQ_[b]); }
    CATLASS_DEVICE uint32_t KBegin(uint32_t) const { return 0; }
    CATLASS_DEVICE uint32_t KLen(uint32_t) const { return maxSeqLenK_; }
    CATLASS_DEVICE uint32_t ActiveQ(uint32_t b) const { return QLen(b); }
    CATLASS_DEVICE uint32_t ActiveK(uint32_t b) const { return static_cast<uint32_t>(sequsedK_[b]); }
    CATLASS_DEVICE int32_t Residual(uint32_t b) const { return cmpResidualK_[b]; }
    CATLASS_DEVICE bool MaskActive() const { return true; }
    CATLASS_DEVICE SparseIndexer::Common::IndexerProblem const& GetProblem() const { return problem_; }

private:
    SparseIndexer::Common::IndexerProblem problem_{};
    uint32_t tq_{0};
    uint32_t maxSeqLenK_{0};
    uint32_t batch_{1};
    int32_t cuQ_[kMaxBatch + 1]{0};
    int32_t sequsedK_[kMaxBatch]{0};
    int32_t cmpResidualK_[kMaxBatch]{0};
};

} // namespace QuantLightningIndexerKernelDetail

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_ARCH35_SOURCE_H
