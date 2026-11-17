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
        if (batch_ > kMaxBatch) batch_ = 1U;

        AscendC::GlobalTensor<int32_t> gmCuQ;
        gmCuQ.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cuSeqlensQ));
        for (uint32_t i = 0; i <= batch_; ++i) cuQ_[i] = gmCuQ.GetValue(i);

        AscendC::GlobalTensor<int32_t> gmSeqK;
        gmSeqK.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequsedK));
        AscendC::GlobalTensor<int32_t> gmResidual;
        gmResidual.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cmpResidualK));
        for (uint32_t i = 0; i < batch_; ++i) {
            sequsedK_[i] = gmSeqK.GetValue(i);
            cmpResidualK_[i] = gmResidual.GetValue(i);
        }
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

private:
    uint32_t tq_{0};
    uint32_t maxSeqLenK_{0};
    uint32_t batch_{1};
    int32_t cuQ_[kMaxBatch + 1]{0};
    int32_t sequsedK_[kMaxBatch]{0};
    int32_t cmpResidualK_[kMaxBatch]{0};
};

} // namespace QuantLightningIndexerKernelDetail

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_ARCH35_SOURCE_H
