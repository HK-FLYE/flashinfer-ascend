/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_TILING_DATA_H
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_TILING_DATA_H

#include <cstdint>

/**
 * Device-visible tiling for the Ascend950 first-grid quant lightning indexer.
 *
 * Host embeds static geometry only. Runtime lengths (cu_seqlens_q, seqused_k,
 * cmp_residual_k) and page addresses stay as GM tensor arguments — no separate
 * metadata product op.
 *
 * First grid locks: quant_mode=1, mask_mode=3, cmp_ratio=4, d=128, nk=1,
 * layout_q=TND, layout_k=PA_BBND, return_value=0 (public values empty).
 *
 * When common block headers are wired, prefer SparseIndexer::Common::kMaxBatch
 * via shared_native_roots; keep the literal until that BuildConfig change lands.
 */
struct QuantLightningIndexerTilingData {
    static constexpr uint32_t kMaxBatch = 16;

    uint32_t tq{0};
    uint32_t n1{0};
    uint32_t d{128};
    uint32_t topk{0};
    uint32_t batch{1};
    uint32_t s1Tile{0};
    uint32_t s2Tile{0};

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
};

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_TILING_DATA_H
