/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_SYNC_BLOCK_BARRIER_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_SYNC_BLOCK_BARRIER_HPP

#include "c_api/sync/sync.h"
#include "catlass/detail/macros.hpp"
#include "kernel_operator.h"

namespace SparseIndexer::QuantLightningIndexer::Policy {

/** C0 cross-core synchronization: one cube/vector hand-off per scheduled tile. */
struct BlockBarrierSyncPolicy {
    static constexpr uint16_t kVecToCube = 0;
    static constexpr uint16_t kCubeToVec = 1;

    CATLASS_DEVICE static void AicWaitCubeReady()
    {
        asc_sync_block_wait(PIPE_FIX, kVecToCube);
    }

    CATLASS_DEVICE static void AicArriveCubeDone()
    {
        asc_sync_block_arrive(PIPE_FIX, kCubeToVec);
    }

    CATLASS_DEVICE static void AicFinalWait()
    {
        asc_sync_block_wait(PIPE_FIX, kVecToCube);
    }

    CATLASS_DEVICE static void AivArriveVecReady()
    {
        asc_sync_block_arrive(PIPE_MTE3, kVecToCube);
    }

    CATLASS_DEVICE static void AivWaitCubeDone()
    {
        asc_sync_block_wait(PIPE_V, kCubeToVec);
    }

    CATLASS_DEVICE static void GlobalBarrier()
    {
        AscendC::SyncAll<false>();
    }
};

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_SYNC_BLOCK_BARRIER_HPP
