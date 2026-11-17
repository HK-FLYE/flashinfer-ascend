/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_ARCH35_H
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_ARCH35_H

#include "kernel_operator.h"

#include "block/block_mmad.hpp"
#include "block/block_scheduler.hpp"
#include "block/block_topk.hpp"
#include "block/key_loader.hpp"
#include "block/mask_policy.hpp"
#include "block_reduce_quant_head.hpp"
#include "quant_lightning_indexer_kernel.h"

namespace QuantLightningIndexerKernelDetail {

using QkDtype = fp8_e4m3fn_t;
using RequestSource = QuantLightningIndexerRequestSource;
using BlockScheduler = SparseIndexer::Common::Block::BlockScheduler<
    RequestSource, SparseIndexer::Common::Block::RequestBlockDiagonalPolicy>;
using PagedKeyLoader = SparseIndexer::Common::Block::PagedKeyLoader<
    Catlass::Arch::Ascend950, Catlass::Gemm::GemmType<QkDtype, Catlass::layout::ColumnMajor>>;
using BlockMmad = SparseIndexer::Common::Block::BlockMmad<QkDtype, QkDtype, PagedKeyLoader>;
using BlockReduceHead = BlockReduceQuantHead<>;
using BlockTopK = SparseIndexer::Common::Block::BlockTopK<float, int32_t>;
using Kernel = QuantLightningIndexerKernel<BlockScheduler, BlockMmad, BlockReduceHead, BlockTopK>;

CATLASS_DEVICE inline void RunKernel(
    GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR qScale, GM_ADDR kScale, GM_ADDR blockTable,
    GM_ADDR cuSeqlensQ, GM_ADDR sequsedK, GM_ADDR cmpResidualK, GM_ADDR indices, GM_ADDR valuesScratch,
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
    AscendC::GlobalTensor<float> gmValues;
    AscendC::GlobalTensor<int32_t> gmTable;
    AscendC::GlobalTensor<int32_t> gmOut;
    gmQ.SetGlobalBuffer(reinterpret_cast<__gm__ QkDtype*>(query));
    gmK.SetGlobalBuffer(reinterpret_cast<__gm__ QkDtype*>(key));
    gmW.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
    gmQScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(qScale));
    gmKScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(kScale));
    gmTable.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTable));
    gmOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices));
    gmValues.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(valuesScratch));
    gmScores.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scores));
    gmZero.SetGlobalBuffer(reinterpret_cast<__gm__ QkDtype*>(zeroPage));
    Kernel kernel;
    kernel(
        gmQ, gmK, gmW, gmQScale, gmKScale, gmOut, gmValues, gmScores, tilingData, cuSeqlensQ, sequsedK,
        cmpResidualK, gmTable, gmZero);
}

} // namespace QuantLightningIndexerKernelDetail

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_ARCH35_H
