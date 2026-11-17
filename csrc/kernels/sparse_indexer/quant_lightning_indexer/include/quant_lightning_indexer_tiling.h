/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_TILING_H
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_TILING_H

#include <cstdint>
#include <stdexcept>

#include "quant_lightning_indexer_tiling_data.h"

namespace QuantLightningIndexerTiling {

struct Params {
    uint32_t tq{0};
    uint32_t n1{0};
    uint32_t d{128};
    uint32_t topk{0};
    uint32_t batch{1};
    uint32_t blockSize{0};
    uint32_t maxBlocksPerBatch{0};
    uint32_t numPages{0};
    uint32_t maxSeqLenK{0};
    int32_t quantMode{1};
    int32_t maskMode{3};
    int32_t cmpRatio{4};
    int32_t returnValue{0};
    bool hasBlockTable{false};
    bool hasCuSeqlensQ{false};
    bool hasSequsedK{false};
    bool hasCmpResidualK{false};
};

inline void GetTiling(const Params& params, QuantLightningIndexerTilingData& tilingData)
{
    if (params.tq == 0 || params.n1 == 0 || params.d != 128 || params.topk == 0) {
        throw std::invalid_argument("QuantLightningIndexerTiling: invalid first-grid geometry");
    }
    if (params.batch == 0 || params.batch > QuantLightningIndexerTilingData::kMaxBatch) {
        throw std::invalid_argument("QuantLightningIndexerTiling: batch out of range");
    }
    if (!params.hasBlockTable || !params.hasCuSeqlensQ || !params.hasSequsedK ||
        !params.hasCmpResidualK) {
        throw std::invalid_argument(
            "QuantLightningIndexerTiling: first grid requires block_table/cu_seqlens_q/"
            "seqused_k/cmp_residual_k");
    }
    if (params.blockSize == 0 || params.numPages == 0 || params.maxBlocksPerBatch == 0 ||
        params.maxSeqLenK == 0) {
        throw std::invalid_argument("QuantLightningIndexerTiling: incomplete PA geometry");
    }
    if (params.quantMode != 1 || params.maskMode != 3 || params.cmpRatio != 4) {
        throw std::invalid_argument("QuantLightningIndexerTiling: attrs outside first grid");
    }
    if (params.topk > 2048) {
        throw std::invalid_argument("QuantLightningIndexerTiling: topk exceeds first-grid cap");
    }

    tilingData = {};
    tilingData.tq = params.tq;
    tilingData.n1 = params.n1;
    tilingData.d = params.d;
    tilingData.topk = params.topk;
    tilingData.batch = params.batch;
    // Conservative first-grid tiles: keep one request-friendly schedule until
    // the device path measures real UB/L1 pressure for FP8.
    tilingData.s1Tile = params.tq < 8 ? params.tq : 8;
    tilingData.s2Tile = params.maxSeqLenK < 256 ? params.maxSeqLenK : 256;
    tilingData.blockSize = params.blockSize;
    tilingData.maxBlocksPerBatch = params.maxBlocksPerBatch;
    tilingData.numPages = params.numPages;
    tilingData.maxSeqLenK = params.maxSeqLenK;
    tilingData.quantMode = params.quantMode;
    tilingData.maskMode = params.maskMode;
    tilingData.cmpRatio = params.cmpRatio;
    tilingData.returnValue = params.returnValue;
}

} // namespace QuantLightningIndexerTiling

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_TILING_H
