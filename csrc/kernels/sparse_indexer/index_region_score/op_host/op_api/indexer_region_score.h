/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef L0_INDEXER_REGION_SCORE_H
#define L0_INDEXER_REGION_SCORE_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *IndexerRegionScore(
    const aclTensor *query, const aclTensor *weights, const aclTensor *keySummary,
    const aclTensor *actualKvLengths,
    const aclTensor *blockTable, const aclTensor *cuSeqLens,
    int64_t maskMode, double headScale,
    int64_t cacheMode, int64_t maxRegionLen, int64_t regionSize,
    int64_t topK, int64_t sortMode,
    const aclTensor *out, const aclTensor *selectedRegionIds,
    aclOpExecutor *executor);
} // namespace l0op

#endif // L0_INDEXER_REGION_SCORE_H
