/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_PROBLEM_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_PROBLEM_HPP

#include <cstdint>

namespace SparseIndexer::QuantLightningIndexer::Policy {

enum class QLayout : uint8_t { TND = 0, BSND = 1 };
enum class KLayout : uint8_t { PA_BBND = 0, TND = 1, BSND = 2 };

struct QuantIndexerProblem {
    uint32_t batch{0};
    uint32_t n1{0};
    uint32_t n2{1};
    uint32_t headDim{0};
    uint32_t groupSize{0};
    uint32_t topK{0};
    uint32_t maxSeqLenK{0};
    QLayout qLayout{QLayout::TND};
    KLayout kLayout{KLayout::PA_BBND};
    uint32_t maskMode{3};
    uint32_t cmpRatio{4};

    constexpr uint32_t S1Base() const
    {
        return topK > 6144U ? 2U : 4U;
    }

    constexpr uint32_t MBase() const
    {
        return S1Base() * groupSize;
    }

    constexpr uint32_t S2Base() const
    {
        return 128U;
    }
};

struct CorePlan {
    bool enabled{false};
    uint32_t bN2Start{0};
    uint32_t bN2End{0};
    uint32_t gS1Start{0};
    uint32_t gS1End{0};
    uint32_t s2Start{0};
    uint32_t s2End{0};
};

struct TileDesc {
    uint32_t coreIdx{0};
    uint32_t batchIdx{0};
    uint32_t n2Idx{0};
    uint32_t qStart{0};
    uint32_t qRows{0};
    uint32_t s2Start{0};
    uint32_t s2Rows{0};
    uint32_t validS2Len{0};
    bool firstS2{false};
    bool lastS2{false};
};

constexpr uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return divisor == 0U ? 0U : (value + divisor - 1U) / divisor;
}

constexpr uint32_t Gsa1BlockCount(const QuantIndexerProblem& problem, uint32_t qLen)
{
    return CeilDiv(qLen, problem.S1Base());
}

constexpr uint32_t S2BlockCount(const QuantIndexerProblem& problem, uint32_t validS2Len)
{
    return CeilDiv(validS2Len, problem.S2Base());
}

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_PROBLEM_HPP
