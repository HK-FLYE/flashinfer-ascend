/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "quant_lightning_indexer_host.h"

ge::graphStatus InferQuantLightningIndexerShape(gert::InferShapeContext* context)
{
    using namespace quant_lightning_indexer_host;
    const auto* query = context->GetInputShape(QUERY);
    const auto* key = context->GetInputShape(KEY);
    if (query == nullptr || key == nullptr || query->GetDimNum() != 3 ||
        (key->GetDimNum() != 3 && key->GetDimNum() != 4)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t tq = query->GetDim(0);
    const int64_t d = query->GetDim(2);
    const int64_t topk = Attr<int64_t>(context, ATTR_TOPK, 0);
    const int64_t maxSeq = Attr<int64_t>(context, ATTR_MAX_SEQ_LEN_K, 0);
    if (tq <= 0 || d != 128 || topk <= 0 || maxSeq <= 0) {
        return ge::GRAPH_FAILED;
    }
    if (key->GetDim(key->GetDimNum() - 1) != d) {
        return ge::GRAPH_FAILED;
    }
    // Native ABI keeps [TQ, NK, K] with NK=1 for the first grid.
    constexpr int64_t nk = 1;
    auto* indices = context->GetOutputShape(0);
    auto* values = context->GetOutputShape(1);
    if (indices == nullptr || values == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (auto* out : {indices, values}) {
        out->SetDimNum(3);
        out->SetDim(0, tq);
        out->SetDim(1, nk);
        out->SetDim(2, topk);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferQuantLightningIndexerType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}
