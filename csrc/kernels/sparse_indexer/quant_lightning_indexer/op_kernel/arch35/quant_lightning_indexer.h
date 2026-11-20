/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_ARCH35_H
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_ARCH35_H

#include "kernel_operator.h"

#include "config/quant_lightning_indexer_config.hpp"
#include "sparse_indexer/pipeline/indexer_kernel.hpp"

namespace QuantLightningIndexerKernelDetail {

using QkDtype = SparseIndexer::QuantLightningIndexer::Config::QkDtype;
using CurrentPolicy = SparseIndexer::QuantLightningIndexer::Config::CurrentPolicy;
using Kernel = SparseIndexer::Common::Pipeline::IndexerKernel<
    CurrentPolicy, QuantLightningIndexerTilingData>;

CATLASS_DEVICE inline void RunKernel(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR qScale, GM_ADDR kScale, GM_ADDR blockTable,
    GM_ADDR cuSeqlensQ, GM_ADDR sequsedK, GM_ADDR cmpResidualK, GM_ADDR indices,
    GM_ADDR scores, GM_ADDR zeroPage, GM_ADDR tiling)
{
    auto const* tilingData = reinterpret_cast<__gm__ QuantLightningIndexerTilingData const*>(tiling);
    AscendC::GlobalTensor<QkDtype> gmQ;
    AscendC::GlobalTensor<QkDtype> gmK;
    AscendC::GlobalTensor<QkDtype> gmZero;
    AscendC::GlobalTensor<float> gmW;
    AscendC::GlobalTensor<float> gmQScale;
    AscendC::GlobalTensor<float> gmKScale;
    AscendC::GlobalTensor<float> gmScores;
    AscendC::GlobalTensor<int32_t> gmTable;
    AscendC::GlobalTensor<int32_t> gmOut;
    gmQ.SetGlobalBuffer(reinterpret_cast<__gm__ QkDtype*>(query));
    gmK.SetGlobalBuffer(reinterpret_cast<__gm__ QkDtype*>(key));
    gmW.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    gmQScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(qScale));
    gmKScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(kScale));
    gmTable.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
    gmOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
    gmScores.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scores));
    gmZero.SetGlobalBuffer(reinterpret_cast<__gm__ QkDtype*>(zeroPage));
    Kernel kernel;
    kernel(
        gmQ, gmK, gmW, gmQScale, gmKScale, gmOut, gmScores, tilingData, cuSeqlensQ, sequsedK,
        cmpResidualK, gmTable, gmZero);
}

} // namespace QuantLightningIndexerKernelDetail

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_ARCH35_H
