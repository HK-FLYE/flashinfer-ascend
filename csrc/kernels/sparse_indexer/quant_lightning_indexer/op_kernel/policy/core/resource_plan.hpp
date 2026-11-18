/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_RESOURCE_PLAN_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_RESOURCE_PLAN_HPP

#include <cstdint>

namespace SparseIndexer::QuantLightningIndexer::Policy {

struct BufferSlot {
    uint32_t offsetBytes{0};
    uint32_t sizeBytes{0};
    uint32_t alignment{256};
};

struct UbResourcePlan {
    BufferSlot query{};
    BufferSlot key{};
    BufferSlot qk{};
    BufferSlot weight{};
    BufferSlot qScale{};
    BufferSlot kScale{};
    BufferSlot score{};
    BufferSlot temp{};
    BufferSlot indices{};
    uint32_t totalBytes{0};
};

constexpr uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return alignment == 0U ? value : (value + alignment - 1U) / alignment * alignment;
}

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_RESOURCE_PLAN_HPP
