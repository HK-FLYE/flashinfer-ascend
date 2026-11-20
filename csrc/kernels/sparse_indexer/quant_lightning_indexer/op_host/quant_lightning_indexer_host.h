/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_HOST_H
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_HOST_H

#include <cstddef>
#include <cstdint>

#include "register/op_def_registry.h"

namespace quant_lightning_indexer_host {
constexpr size_t QUERY = 0;
constexpr size_t KEY = 1;
constexpr size_t WEIGHTS = 2;
constexpr size_t Q_SCALE = 3;
constexpr size_t K_SCALE = 4;
constexpr size_t BLOCK_TABLE = 5;
constexpr size_t CU_Q = 6;
constexpr size_t CU_K = 7;
constexpr size_t SEQ_Q = 8;
constexpr size_t SEQ_K = 9;
constexpr size_t CMP_RESIDUAL = 10;

// Attr order matches NativeBinding attributes.
constexpr size_t ATTR_TOPK = 0;
constexpr size_t ATTR_QUANT_MODE = 1;
constexpr size_t ATTR_MASK_MODE = 2;
constexpr size_t ATTR_CMP_RATIO = 3;
constexpr size_t ATTR_MAX_SEQ_LEN_K = 4;
constexpr size_t ATTR_MAX_SEQLEN_Q = 5;

template <typename T, typename Context>
T Attr(const Context* context, size_t index, T fallback)
{
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return fallback;
    }
    const T* value = attrs->template GetAttrPointer<T>(index);
    return value == nullptr ? fallback : *value;
}
} // namespace quant_lightning_indexer_host

ge::graphStatus InferQuantLightningIndexerShape(gert::InferShapeContext* context);
ge::graphStatus InferQuantLightningIndexerType(gert::InferDataTypeContext* context);
ge::graphStatus FiQuantLightningIndexerTiling(gert::TilingContext* context);

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_HOST_H
