/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <stdexcept>

#include "quant_lightning_indexer_host.h"
#include "quant_lightning_indexer_tiling.h"
#include "tiling/platform/platform_ascendc.h"

ge::graphStatus FiQuantLightningIndexerTiling(gert::TilingContext* context)
{
    using namespace quant_lightning_indexer_host;
    try {
        const auto* queryTensor = context->GetInputTensor(QUERY);
        const auto* keyTensor = context->GetInputTensor(KEY);
        const auto* weightsTensor = context->GetInputTensor(WEIGHTS);
        const auto* qScaleTensor = context->GetInputTensor(Q_SCALE);
        const auto* kScaleTensor = context->GetInputTensor(K_SCALE);
        const auto* tableTensor = context->GetInputTensor(BLOCK_TABLE);
        const auto* cuQTensor = context->GetInputTensor(CU_Q);
        const auto* seqKTensor = context->GetInputTensor(SEQ_K);
        const auto* residualTensor = context->GetInputTensor(CMP_RESIDUAL);
        if (queryTensor == nullptr || keyTensor == nullptr || weightsTensor == nullptr ||
            qScaleTensor == nullptr || kScaleTensor == nullptr || tableTensor == nullptr ||
            cuQTensor == nullptr || seqKTensor == nullptr || residualTensor == nullptr) {
            return ge::GRAPH_FAILED;
        }

        const auto& qs = queryTensor->GetStorageShape();
        const auto& ks = keyTensor->GetStorageShape();
        const auto& ws = weightsTensor->GetStorageShape();
        if (qs.GetDimNum() != 3 || ws.GetDimNum() != 2 ||
            (ks.GetDimNum() != 3 && ks.GetDimNum() != 4)) {
            return ge::GRAPH_FAILED;
        }
        const int64_t tq = qs.GetDim(0);
        const int64_t n1 = qs.GetDim(1);
        const int64_t d = qs.GetDim(2);
        if (tq <= 0 || n1 <= 0 || d != 128 || ws.GetDim(0) != tq || ws.GetDim(1) != n1) {
            return ge::GRAPH_FAILED;
        }
        if (ks.GetDim(ks.GetDimNum() - 1) != d) {
            return ge::GRAPH_FAILED;
        }

        const int64_t topk = Attr<int64_t>(context, ATTR_TOPK, 0);
        const int64_t quantMode = Attr<int64_t>(context, ATTR_QUANT_MODE, 1);
        const int64_t maskMode = Attr<int64_t>(context, ATTR_MASK_MODE, 3);
        const int64_t cmpRatio = Attr<int64_t>(context, ATTR_CMP_RATIO, 4);
        const int64_t maxSeq = Attr<int64_t>(context, ATTR_MAX_SEQ_LEN_K, 0);
        const int64_t returnValue = Attr<int64_t>(context, ATTR_RETURN_VALUE, 0);
        if (topk <= 0 || maxSeq <= 0) {
            return ge::GRAPH_FAILED;
        }

        const int64_t batch = cuQTensor->GetStorageShape().GetDim(0) - 1;
        if (batch < 1) {
            return ge::GRAPH_FAILED;
        }

        QuantLightningIndexerTiling::Params params;
        params.tq = static_cast<uint32_t>(tq);
        params.n1 = static_cast<uint32_t>(n1);
        params.d = static_cast<uint32_t>(d);
        params.topk = static_cast<uint32_t>(topk);
        params.batch = static_cast<uint32_t>(batch);
        params.blockSize = static_cast<uint32_t>(ks.GetDim(1));
        params.numPages = static_cast<uint32_t>(ks.GetDim(0));
        params.maxBlocksPerBatch = static_cast<uint32_t>(tableTensor->GetStorageShape().GetDim(1));
        params.maxSeqLenK = static_cast<uint32_t>(maxSeq);
        params.quantMode = static_cast<int32_t>(quantMode);
        params.maskMode = static_cast<int32_t>(maskMode);
        params.cmpRatio = static_cast<int32_t>(cmpRatio);
        params.returnValue = static_cast<int32_t>(returnValue);
        params.hasBlockTable = true;
        params.hasCuSeqlensQ = true;
        params.hasSequsedK = true;
        params.hasCmpResidualK = true;

        auto* data = context->GetTilingData<QuantLightningIndexerTilingData>();
        if (data == nullptr) {
            return ge::GRAPH_FAILED;
        }
        QuantLightningIndexerTiling::GetTiling(params, *data);

        platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
        const uint32_t cores = platform.GetCoreNumAic();
        if (cores == 0) {
            return ge::GRAPH_FAILED;
        }
        context->SetBlockDim(cores);
        context->SetTilingKey(0);
        context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
        return ge::GRAPH_SUCCESS;
    } catch (const std::exception&) {
        return ge::GRAPH_FAILED;
    }
}
