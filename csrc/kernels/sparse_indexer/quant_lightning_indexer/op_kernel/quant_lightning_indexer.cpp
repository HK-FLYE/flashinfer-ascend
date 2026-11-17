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
#include "arch35/quant_lightning_indexer_kernel.h"
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
    (void)values;

    auto const* tilingData =
        reinterpret_cast<__gm__ QuantLightningIndexerTilingData const*>(tiling);
    if ASCEND_IS_AIC {
        return;
    }
    if ASCEND_IS_AIV {
        // First green: one AIV owns the whole first-grid case.
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        QuantLightningIndexerKernel::RunFirstGrid(
            query, key, weights, query_dequant_scale, key_dequant_scale, block_table,
            cu_seqlens_q, seqused_k, cmp_residual_k, indices, tilingData);
    }
}
