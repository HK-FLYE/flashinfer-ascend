/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif
#include "kernel_operator.h"
#include "quant_lightning_indexer_tiling_data.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/quant_lightning_indexer.h"
#else
#error "quant_lightning_indexer requires arch35 (__CCE_AICORE__ == 310)"
#endif

extern "C" __global__ __aicore__ void fi_quant_lightning_indexer(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR query_dequant_scale,
    GM_ADDR key_dequant_scale, GM_ADDR block_table, GM_ADDR cu_seqlens_q,
    GM_ADDR cu_seqlens_k, GM_ADDR seqused_q, GM_ADDR seqused_k,
    GM_ADDR cmp_residual_k, GM_ADDR indices, GM_ADDR values, GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(QuantLightningIndexerTilingData);
    AscendC::SetSysWorkspace(workspace);
    (void)cu_seqlens_k;
    (void)seqused_q;

    auto const* tilingData =
        reinterpret_cast<__gm__ QuantLightningIndexerTilingData const*>(tiling);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    const uint64_t scoreBytes =
        static_cast<uint64_t>(tilingData->tq) * tilingData->maxSeqLenK * sizeof(float);
    const uint64_t valueBytes = tilingData->returnValue == 0
        ? 0
        : static_cast<uint64_t>(tilingData->tq) * tilingData->topk * sizeof(float);
    GM_ADDR scores = userWorkspace;
    GM_ADDR valuesScratch = userWorkspace + scoreBytes;
    GM_ADDR zeroPage = userWorkspace + scoreBytes + valueBytes;
    QuantLightningIndexerKernelDetail::RunKernel(
        query, key, weights, query_dequant_scale, key_dequant_scale, block_table, cu_seqlens_q, seqused_k,
        cmp_residual_k, indices, valuesScratch, values, scores, zeroPage, tiling);
}
