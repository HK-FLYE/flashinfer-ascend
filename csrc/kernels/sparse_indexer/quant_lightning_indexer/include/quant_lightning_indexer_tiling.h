/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_TILING_H
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_TILING_H

#include <cstdint>
#include <stdexcept>

#include "quant_lightning_indexer_tiling_data.h"
#include "sparse_indexer_common.hpp"

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
    uint64_t keyStride0{0};
    uint64_t kScaleStride0{0};
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
    if (params.keyStride0 < static_cast<uint64_t>(params.blockSize) * params.d ||
        params.kScaleStride0 < params.blockSize) {
        throw std::invalid_argument("QuantLightningIndexerTiling: invalid PA page stride");
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
    constexpr uint32_t kS2TileCap = 128;
    constexpr uint32_t kUbSize = 248 * 1024;
    constexpr uint32_t kL1Size = 512 * 1024;
    constexpr uint32_t kL0cSize = 256 * 1024;
    constexpr uint32_t kQkElemBytes = 1;
    constexpr uint32_t kWElemBytes = 4;
    constexpr uint32_t kAivPerAic = 2;
    constexpr uint32_t kCubeFractal = 16;
    constexpr uint32_t kS1Align = 8;
    constexpr uint32_t kTopkS2TileCap = 2048;
    constexpr uint32_t kTopkS2Align = 16;
    constexpr uint32_t kTopkRowsPerTask = 2;
    // The register-VF head reduction is independent of srcRep, so S2 tiling is
    // no longer constrained by the legacy multi-row ScaleHead limit.
    uint32_t s2Tile = params.maxSeqLenK <= kS2TileCap
        ? SparseIndexer::Common::AlignUp(params.maxSeqLenK, kCubeFractal)
        : kS2TileCap;
    const uint32_t s2Split = SparseIndexer::Common::AlignUp(s2Tile, kCubeFractal * kAivPerAic);
    const uint32_t wStride = SparseIndexer::Common::AlignUp(params.n1, 32 / kWElemBytes);
    // Score + out + W/qScale tiles + wBrcb/ubTmp(kFloatVec) + ubKScale pad.
    constexpr uint32_t kFloatVec = 64;
    const uint64_t ubBytesPerS1 = sizeof(float) * (s2Split / kAivPerAic) * (params.n1 + 1) +
        sizeof(float) * wStride * 3 + sizeof(float) * (8 + kFloatVec) +
        sizeof(float) * (s2Split / kAivPerAic) + 256U;
    const uint32_t l0cRows = kL0cSize / (s2Split * sizeof(float));
    uint32_t s1Tile = static_cast<uint32_t>(kUbSize / ubBytesPerS1);
    while (s1Tile > 0 &&
        SparseIndexer::Common::AlignUp(s1Tile * params.n1, kCubeFractal) > l0cRows) {
        --s1Tile;
    }
    const uint32_t l1BBytes = s2Tile * params.d * kQkElemBytes;
    const uint64_t l1Budget = l1BBytes < kL1Size ? kL1Size - l1BBytes : 0;
    while (s1Tile > 0 &&
        static_cast<uint64_t>(SparseIndexer::Common::AlignUp(s1Tile * params.n1, kCubeFractal)) *
            params.d * kQkElemBytes > l1Budget) {
        --s1Tile;
    }
    if (s1Tile >= kS1Align) s1Tile = s1Tile / kS1Align * kS1Align;
    if (s1Tile == 0) {
        throw std::invalid_argument("QuantLightningIndexerTiling: s1Tile is zero under resource limits");
    }
    // The 2D qTile x s2Tile scheduler must produce enough independent reduce
    // tasks. Resource-max s1Tile collapses short-K requests onto one or two
    // AIC blocks. Eight rows is the smallest block that passes the current
    // matrix anchors; the four-row SPLIT_M path remains a separate follow-up.
    constexpr uint32_t kS1TileCap = 8;
    if (s1Tile > kS1TileCap) {
        s1Tile = kS1TileCap;
    }
    tilingData.s1Tile = s1Tile;
    tilingData.s2Tile = s2Tile;
    tilingData.topkS2Tile = SparseIndexer::Common::AlignUp(
        params.maxSeqLenK < kTopkS2TileCap ? params.maxSeqLenK : kTopkS2TileCap, kTopkS2Align);
    tilingData.topkRowsPerTask = kTopkRowsPerTask;
    tilingData.blockSize = params.blockSize;
    tilingData.maxBlocksPerBatch = params.maxBlocksPerBatch;
    tilingData.numPages = params.numPages;
    tilingData.maxSeqLenK = params.maxSeqLenK;
    tilingData.keyStride0 = params.keyStride0;
    tilingData.kScaleStride0 = params.kScaleStride0;
    tilingData.quantMode = params.quantMode;
    tilingData.maskMode = params.maskMode;
    tilingData.cmpRatio = params.cmpRatio;
    tilingData.returnValue = params.returnValue;
}

} // namespace QuantLightningIndexerTiling

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_TILING_H
