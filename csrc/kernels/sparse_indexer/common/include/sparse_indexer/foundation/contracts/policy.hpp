/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_FOUNDATION_CONTRACTS_POLICY_HPP
#define FLASHINFER_SPARSE_INDEXER_FOUNDATION_CONTRACTS_POLICY_HPP

#include <type_traits>
#include <utility>

namespace SparseIndexer::Common {

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

template <class T, class = void>
struct HasLoadPolicy : std::false_type {};
template <class T>
struct HasLoadPolicy<T, std::void_t<typename T::Loader, typename T::Dtype>> : std::true_type {};

template <class T, class = void>
struct HasSchedulePolicy : std::false_type {};
template <class T>
struct HasSchedulePolicy<T, std::void_t<typename T::Scheduler, typename T::Params, typename T::Coord,
    typename T::Task>> : std::true_type {};

template <class T, class = void>
struct HasMmadPolicy : std::false_type {};
template <class T>
struct HasMmadPolicy<T, std::void_t<typename T::InputType, typename T::Block, typename T::LoadPolicy>>
    : std::true_type {};

template <class T, class = void>
struct HasReducePolicy : std::false_type {};
template <class T>
struct HasReducePolicy<T, std::void_t<typename T::Block>> : std::true_type {};

template <class T, class = void>
struct HasSelectionPolicy : std::false_type {};
template <class T>
struct HasSelectionPolicy<T, std::void_t<typename T::VectorBlock>> : std::true_type {};

template <class T, class = void>
struct HasSyncPolicy : std::false_type {};
template <class T>
struct HasSyncPolicy<T, std::void_t<decltype(T::GlobalBarrier()), decltype(T::AicWaitCubeReady(0U)),
    decltype(T::AicArriveCubeDone(0U)), decltype(T::AivWaitCubeDone(0U)),
    decltype(T::AivArriveVecReady(0U))>> : std::true_type {};

template <class T, class = void>
struct HasResourcePolicy : std::false_type {};
template <class T>
struct HasResourcePolicy<T, std::void_t<typename T::Resource>> : std::true_type {};

} // namespace SparseIndexer::Common

#endif // FLASHINFER_SPARSE_INDEXER_FOUNDATION_CONTRACTS_POLICY_HPP
