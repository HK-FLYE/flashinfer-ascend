/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_TRAITS_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_TRAITS_HPP

#include <type_traits>

#include "problem.hpp"

namespace SparseIndexer::QuantLightningIndexer::Policy {

struct PolicyBundleTag {};

template <class Schedule_, class Load_, class Mmad_, class Reduce_, class TopK_, class Sync_, class Resource_>
struct PolicyBundle : PolicyBundleTag {
    using Schedule = Schedule_;
    using Load = Load_;
    using Mmad = Mmad_;
    using Reduce = Reduce_;
    using TopK = TopK_;
    using Sync = Sync_;
    using Resource = Resource_;
};

template <class T>
struct IsPolicyBundle : std::is_base_of<PolicyBundleTag, T> {};

struct CurrentPerformanceTraits {
    static constexpr bool kSplitAivByRow = false;
    static constexpr bool kS2InnerLoop = false;
    static constexpr bool kTopKPerGSA1 = false;
    static constexpr bool kUseMetadataBalance = false;
    static constexpr uint32_t kS2Base = 128;
};

struct QliV2PerformanceTraits {
    static constexpr bool kSplitAivByRow = true;
    static constexpr bool kS2InnerLoop = true;
    static constexpr bool kTopKPerGSA1 = true;
    static constexpr bool kUseMetadataBalance = true;
    static constexpr uint32_t kS2Base = 128;
};

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_TRAITS_HPP
