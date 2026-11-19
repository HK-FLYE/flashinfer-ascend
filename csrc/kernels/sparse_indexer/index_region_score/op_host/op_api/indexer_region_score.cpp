/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "indexer_region_score.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(IndexerRegionScore);

const aclTensor *IndexerRegionScore(
    const aclTensor *query, const aclTensor *weights, const aclTensor *keySummary,
    const aclTensor *actualKvLengths,
    const aclTensor *blockTable, const aclTensor *cuSeqLens,
    int64_t maskMode, double headScale,
    int64_t cacheMode, int64_t maxRegionLen, int64_t regionSize,
    int64_t topK, int64_t sortMode,
    const aclTensor *out, const aclTensor *selectedRegionIds,
    aclOpExecutor *executor)
{
    L0_DFX(IndexerRegionScore, query, weights, keySummary, actualKvLengths,
           blockTable, cuSeqLens, maskMode,
           headScale, cacheMode, maxRegionLen, regionSize, topK, sortMode,
           out, selectedRegionIds);

    const bool returnScore = out != nullptr;
    // GE retains two output slots; the omitted Score owns zero elements.
    const aclTensor *scoreOutput = returnScore ? out : executor->AllocTensor(op::Shape{0}, DataType::DT_FLOAT);
    if (scoreOutput == nullptr) {
        return nullptr;
    }
    const aclnnStatus ret = ADD_TO_LAUNCHER_LIST_AICORE(
        IndexerRegionScore,
        OP_INPUT(query, weights, keySummary, actualKvLengths, blockTable, cuSeqLens),
        OP_OUTPUT(scoreOutput, selectedRegionIds),
        OP_ATTR(maskMode, static_cast<float>(headScale), cacheMode, maxRegionLen, regionSize,
                topK, sortMode, returnScore, int64_t(0), int64_t(0), int64_t(0)));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "IndexerRegionScore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return scoreOutput;
}
} // namespace l0op
