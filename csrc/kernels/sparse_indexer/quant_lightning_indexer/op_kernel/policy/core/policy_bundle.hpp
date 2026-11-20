/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_BUNDLE_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_BUNDLE_HPP

#include <type_traits>

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

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_BUNDLE_HPP
