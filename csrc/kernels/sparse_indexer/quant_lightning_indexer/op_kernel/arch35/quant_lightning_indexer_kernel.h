/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * Ascend950 MIX path: AIC BlockMmad (FP8) → AIV quant reduce (ReLU / scales /
 * weights / mask3) → AIV TopK. Layout and sync follow general_indexer; scale
 * folding follows QLIv2 quant_mode=1.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_ARCH35_KERNEL_H
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_ARCH35_KERNEL_H

#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "kernel_operator.h"
#include "c_api/sync/sync.h"

#include "block/block_mmad.hpp"
#include "block/block_scheduler.hpp"
#include "block/block_topk.hpp"
#include "block/block_topk_vf.hpp"
#include "block/key_loader.hpp"
#include "quant_lightning_indexer_source.h"
#include "quant_lightning_indexer_task.h"

template <class BlockScheduler_, class BlockMmad_, class BlockReduceHead_, class BlockTopK_, class BlockTopKVector_>
class QuantLightningIndexerKernel {
    using BlockScheduler = BlockScheduler_;
    using BlockMmad = BlockMmad_;
    using BlockReduceHead = BlockReduceHead_;
    using BlockTopK = BlockTopK_;
    using BlockTopKVector = BlockTopKVector_;
    using InputType = typename BlockMmad::DtypeA;

    CATLASS_DEVICE static typename BlockScheduler::Params MakeSchedulerParams(
        __gm__ QuantLightningIndexerTilingData const* tiling, GM_ADDR cuSeqlensQ, GM_ADDR sequsedK,
        GM_ADDR cmpResidualK)
    {
        typename BlockScheduler::Params params{};
        params.policy.s1Tile = tiling->s1Tile;
        params.policy.s2Tile = tiling->s2Tile;
        params.policy.topkRowsPerTask = tiling->topkRowsPerTask;
        params.source.InitFromGm(tiling, cuSeqlensQ, sequsedK, cmpResidualK);
        return params;
    }

    CATLASS_DEVICE static typename BlockMmad::KeyLoaderParams MakeKeyLoaderParams(
        __gm__ QuantLightningIndexerTilingData const* tiling)
    {
        typename BlockMmad::KeyLoaderParams params{};
        params.d = tiling->d;
        params.blockSize = tiling->blockSize;
        params.maxBlocksPerBatch = tiling->maxBlocksPerBatch;
        params.numPages = tiling->numPages;
        params.keyStride0 = tiling->keyStride0;
        return params;
    }

public:
    CATLASS_DEVICE
    QuantLightningIndexerKernel() : ubC_(resource_.ubBuf.template GetBufferByByte<typename BlockMmad::DtypeC>(0)) {}

    CATLASS_DEVICE void operator()(
        AscendC::GlobalTensor<InputType> const& gmQ, AscendC::GlobalTensor<InputType> const& gmK,
        AscendC::GlobalTensor<float> const& gmW, AscendC::GlobalTensor<float> const& gmQScale,
        AscendC::GlobalTensor<float> const& gmKScale, AscendC::GlobalTensor<int32_t> const& outIndices,
        AscendC::GlobalTensor<float> const& outValues, AscendC::GlobalTensor<float>& scores,
        __gm__ QuantLightningIndexerTilingData const* tiling, GM_ADDR cuSeqlensQ, GM_ADDR sequsedK,
        GM_ADDR cmpResidualK, AscendC::GlobalTensor<int32_t> const& gmBlockTable,
        AscendC::GlobalTensor<InputType> const& gmZeroPage)
    {
        typename BlockScheduler::Params schedulerParams =
            MakeSchedulerParams(tiling, cuSeqlensQ, sequsedK, cmpResidualK);
        BlockScheduler blockScheduler(schedulerParams);
        if ASCEND_IS_AIC {
            typename BlockMmad::Params mmadParams{MakeKeyLoaderParams(tiling)};
            BlockMmad blockMmad(
                ubC_, gmQ, gmK, Catlass::GemmCoord(tiling->tq * tiling->n1, tiling->maxSeqLenK, tiling->d),
                Catlass::GemmCoord(
                    SparseIndexer::Common::AlignUp(tiling->s1Tile * tiling->n1, 16U), tiling->s2Tile, tiling->d),
                resource_, mmadParams, gmBlockTable, gmZeroPage);
            SparseIndexer::Common::Block::ScheduleCoord coord{};
            while (blockScheduler.GetNextCoord(coord)) {
                const auto task = QuantLightningIndexerKernelDetail::MakeQuantLightningIndexerTask(
                    coord, blockScheduler.source());
                const SparseIndexer::Common::Block::KeyCoord keyCoord{
                    task.coord.kStart, task.coord.kStart - task.requestKStart, task.coord.batchIdx};
                asc_sync_block_wait(PIPE_FIX, kVecToCube);
                blockMmad(
                    task.coord.qStart * tiling->n1, task.coord.actualQ * tiling->n1, keyCoord, task.coord.actualK);
                asc_sync_block_arrive(PIPE_FIX, kCubeToVec);
            }
            asc_sync_block_wait(PIPE_FIX, kVecToCube);
        } else {
            typename BlockReduceHead::Params params{};
            params.n1 = tiling->n1;
            params.s2 = tiling->maxSeqLenK;
            params.s1Tile = tiling->s1Tile;
            params.s2Tile = tiling->s2Tile;
            params.maskMode = tiling->maskMode;
            params.cmpRatio = tiling->cmpRatio;
            params.blockSize = tiling->blockSize;
            params.maxBlocksPerBatch = tiling->maxBlocksPerBatch;
            params.numPages = tiling->numPages;
            params.kScaleStride0 = tiling->kScaleStride0;
            BlockReduceHead reducer(ubC_, resource_, gmW, gmQScale, gmKScale, gmBlockTable, scores, params);
            asc_sync_block_arrive(PIPE_MTE3, kVecToCube);
            SparseIndexer::Common::Block::ScheduleCoord coord{};
            while (blockScheduler.GetNextCoord(coord)) {
                const auto task = QuantLightningIndexerKernelDetail::MakeQuantLightningIndexerTask(
                    coord, blockScheduler.source());
                asc_sync_block_wait(PIPE_V, kCubeToVec);
                reducer(
                    task.coord.qStart, task.coord.actualQ, task.coord.kStart, task.coord.actualK,
                    task.coord.localQStart, task.activeQ, task.activeK, task.requestKStart, task.residual,
                    task.maskActive, task.coord.batchIdx);
                asc_sync_block_arrive(PIPE_MTE3, kVecToCube);
            }
        }
        AscendC::SyncAll<false>();
        if ASCEND_IS_AIV {
            AscendC::GlobalTensor<int32_t> dummyOffset;
            SparseIndexer::Common::Block::ScheduleCoord coord{};
            if (tiling->returnValue == 0 && tiling->topk == 2048 && tiling->maxSeqLenK < tiling->topk) {
                typename BlockTopKVector::Params params{};
                params.topK = tiling->topk;
                params.s2Stride = tiling->maxSeqLenK;
                params.s2Tile = tiling->topkS2Tile;
                params.hasOutputIdxOffset = false;
                BlockTopKVector topk(resource_, scores, outIndices, params, dummyOffset, outValues);
                while (blockScheduler.GetNextTopKCoord(coord)) {
                    const auto task = QuantLightningIndexerKernelDetail::MakeQuantLightningIndexerTask(
                        coord, blockScheduler.source());
                    topk(coord.qStart, coord.actualQ, coord.kStart, coord.actualK, coord.localQStart,
                        task.activeQ, task.activeK, task.residual);
                }
            } else {
                typename BlockTopK::Params params{};
                params.topK = tiling->topk;
                params.s2Stride = tiling->maxSeqLenK;
                params.s2Tile = tiling->topkS2Tile;
                params.hasOutputIdxOffset = false;
                params.writeValues = tiling->returnValue != 0;
                BlockTopK topk(resource_, scores, outIndices, params, dummyOffset, outValues);
                while (blockScheduler.GetNextTopKCoord(coord)) {
                    topk(coord.qStart, coord.actualQ, coord.kStart, coord.actualK);
                }
            }
        }
    }

private:
    static constexpr uint16_t kVecToCube = 0;
    static constexpr uint16_t kCubeToVec = 1;
    Catlass::Arch::Resource<Catlass::Arch::Ascend950> resource_;
    AscendC::LocalTensor<typename BlockMmad::DtypeC> ubC_;
};

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_ARCH35_KERNEL_H
