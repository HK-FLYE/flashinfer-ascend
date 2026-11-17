/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * Correctness-first Ascend950 path for the first grid. Uses AIV scalar GM
 * access and a software FP8 e4m3fn decode so small shapes can match the CPU
 * golden before the cube/FP8 production schedule lands.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_KERNEL_ARCH35_H
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_KERNEL_ARCH35_H

#include "kernel_operator.h"
#include "sparse_indexer_common.hpp"
#include "block/mask_utils.hpp"
#include "quant_lightning_indexer_tiling_data.h"

namespace QuantLightningIndexerKernel {

CATLASS_DEVICE float NegInf()
{
    union {
        float f;
        uint32_t u;
    } bits;
    bits.u = SparseIndexer::Common::Block::kFloatNegInfBits;
    return bits.f;
}

CATLASS_DEVICE float DecodeE4M3FN(uint8_t bits)
{
    const int sign = (bits >> 7) & 0x1;
    const int exp = (bits >> 3) & 0xf;
    const int mant = bits & 0x7;
    float value = 0.0f;
    if (exp == 0) {
        if (mant != 0) {
            value = (static_cast<float>(mant) / 8.0f) * 0.015625f; // 2^(1-7)
        }
    } else if (exp == 0xf) {
        // OCP e4m3fn: no Inf; NaN when mant != 0. Map NaN to 0 for TopK safety.
        value = 0.0f;
    } else {
        int e = exp - 7;
        float scale = 1.0f;
        if (e > 0) {
            for (int i = 0; i < e; ++i) {
                scale *= 2.0f;
            }
        } else if (e < 0) {
            for (int i = 0; i < -e; ++i) {
                scale *= 0.5f;
            }
        }
        value = (1.0f + static_cast<float>(mant) / 8.0f) * scale;
    }
    return sign ? -value : value;
}

CATLASS_DEVICE float Relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

CATLASS_DEVICE void StableTopK(
    const float* scores, uint32_t scoreLen, uint32_t topk, int32_t* outIndices)
{
    for (uint32_t slot = 0; slot < topk; ++slot) {
        outIndices[slot] = -1;
    }
    for (uint32_t slot = 0; slot < topk; ++slot) {
        int32_t bestIdx = -1;
        float bestScore = NegInf();
        for (uint32_t i = 0; i < scoreLen; ++i) {
            const float score = scores[i];
            if (score > bestScore || (score == bestScore && (bestIdx < 0 || static_cast<int32_t>(i) < bestIdx))) {
                bool used = false;
                for (uint32_t prev = 0; prev < slot; ++prev) {
                    if (outIndices[prev] == static_cast<int32_t>(i)) {
                        used = true;
                        break;
                    }
                }
                if (used || SparseIndexer::Common::Block::IsNegInfFloat(score)) {
                    continue;
                }
                bestScore = score;
                bestIdx = static_cast<int32_t>(i);
            }
        }
        outIndices[slot] = bestIdx;
    }
}

CATLASS_DEVICE QuantLightningIndexerTilingData LoadTiling(
    __gm__ QuantLightningIndexerTilingData const* tiling)
{
    QuantLightningIndexerTilingData local;
    local.tq = tiling->tq;
    local.n1 = tiling->n1;
    local.d = tiling->d;
    local.topk = tiling->topk;
    local.batch = tiling->batch;
    local.s1Tile = tiling->s1Tile;
    local.s2Tile = tiling->s2Tile;
    local.blockSize = tiling->blockSize;
    local.maxBlocksPerBatch = tiling->maxBlocksPerBatch;
    local.numPages = tiling->numPages;
    local.maxSeqLenK = tiling->maxSeqLenK;
    local.quantMode = tiling->quantMode;
    local.maskMode = tiling->maskMode;
    local.cmpRatio = tiling->cmpRatio;
    local.returnValue = tiling->returnValue;
    return local;
}

CATLASS_DEVICE void RunFirstGrid(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR queryScale, GM_ADDR keyScale,
    GM_ADDR blockTable, GM_ADDR cuSeqlensQ, GM_ADDR sequsedK, GM_ADDR cmpResidualK,
    GM_ADDR indices, __gm__ QuantLightningIndexerTilingData const* tilingPtr)
{
    const QuantLightningIndexerTilingData tilingRef = LoadTiling(tilingPtr);
    AscendC::GlobalTensor<uint8_t> qGm;
    AscendC::GlobalTensor<uint8_t> kGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> qScaleGm;
    AscendC::GlobalTensor<float> kScaleGm;
    AscendC::GlobalTensor<int32_t> tableGm;
    AscendC::GlobalTensor<int32_t> cuQGm;
    AscendC::GlobalTensor<int32_t> seqKGm;
    AscendC::GlobalTensor<int32_t> residualGm;
    AscendC::GlobalTensor<int32_t> indicesGm;

    qGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(query));
    kGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(key));
    wGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    qScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(queryScale));
    kScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(keyScale));
    tableGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
    cuQGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cuSeqlensQ));
    seqKGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequsedK));
    residualGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cmpResidualK));
    indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));

    const uint32_t n1 = tilingRef.n1;
    const uint32_t d = tilingRef.d;
    const uint32_t topk = tilingRef.topk;
    const uint32_t blockSize = tilingRef.blockSize;
    const uint32_t maxBlocks = tilingRef.maxBlocksPerBatch;
    const uint32_t maxSeq = tilingRef.maxSeqLenK;
    const int32_t cmpRatio = tilingRef.cmpRatio;

    // Local scratch in private arrays — first-grid cases stay small.
    float scoreRow[256];
    int32_t topIdx[64];
    if (maxSeq > 256 || topk > 64) {
        return;
    }

    for (uint32_t b = 0; b < tilingRef.batch; ++b) {
        const int32_t q0 = cuQGm.GetValue(b);
        const int32_t q1 = cuQGm.GetValue(b + 1);
        const uint32_t qLen = static_cast<uint32_t>(q1 - q0);
        const uint32_t kAct = static_cast<uint32_t>(seqKGm.GetValue(b));
        const int32_t residual = residualGm.GetValue(b);
        const uint32_t visibleCap = kAct < maxSeq ? kAct : maxSeq;

        for (uint32_t localQ = 0; localQ < qLen; ++localQ) {
            const uint32_t globalQ = static_cast<uint32_t>(q0) + localQ;
            for (uint32_t s = 0; s < maxSeq; ++s) {
                scoreRow[s] = NegInf();
            }
            const uint32_t visible = SparseIndexer::Common::Block::Mask3VisibleLen(
                localQ, qLen, visibleCap, cmpRatio, residual);
            for (uint32_t s = 0; s < visible; ++s) {
                const uint32_t pageSlot = tableGm.GetValue(b * maxBlocks + (s / blockSize));
                if (static_cast<int32_t>(pageSlot) < 0) {
                    continue;
                }
                const uint32_t pageOff = pageSlot * blockSize + (s % blockSize);
                float score = 0.0f;
                for (uint32_t h = 0; h < n1; ++h) {
                    const float qs = qScaleGm.GetValue(globalQ * n1 + h);
                    const float ks = kScaleGm.GetValue(pageOff);
                    const float w = wGm.GetValue(globalQ * n1 + h);
                    float dot = 0.0f;
                    const uint32_t qBase = (globalQ * n1 + h) * d;
                    const uint32_t kBase = pageOff * d;
                    for (uint32_t dim = 0; dim < d; ++dim) {
                        const float qv = DecodeE4M3FN(qGm.GetValue(qBase + dim));
                        const float kv = DecodeE4M3FN(kGm.GetValue(kBase + dim));
                        dot += (qv * qs) * (kv * ks);
                    }
                    score += w * Relu(dot);
                }
                scoreRow[s] = score;
            }
            StableTopK(scoreRow, maxSeq, topk, topIdx);
            const uint32_t outBase = globalQ * topk; // [TQ,1,K] with NK=1
            for (uint32_t slot = 0; slot < topk; ++slot) {
                indicesGm.SetValue(outBase + slot, topIdx[slot]);
            }
        }
    }
}

} // namespace QuantLightningIndexerKernel

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_KERNEL_ARCH35_H
