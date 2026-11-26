// 版权所有华为技术有限公司 2026，遵循 CANN 开源软件许可协议 2.0。
#ifndef INDEXER_REGION_SCORE_DETAIL_TOPK_SELECTION_HPP
#define INDEXER_REGION_SCORE_DETAIL_TOPK_SELECTION_HPP

#include <cstdint>
#include "kernel_operator.h"
#include "request.hpp"
#include "primitive/topk_fp32.hpp"

namespace SparseTopK = SparseIndexer::Common::Primitive::TopK;
namespace topkb32gather = SparseTopK::RadixDetail;

namespace IndexerRegionScore {
namespace FusedSelect {
using namespace AscendC;

// Tile and ISA capacities shared by the IRS caller policies.
constexpr uint32_t kScoreTile = 1024U;
constexpr uint32_t kRadixTile = 8192U;
constexpr uint32_t kRadixMaxTopK = SparseTopK::kMaxKeepCount;
constexpr uint32_t kMaxTopK = kRadixMaxTopK;
constexpr uint32_t kNativeTopK = 512U;
constexpr uint32_t kNativeShortRowCapacity = 128U;
constexpr uint32_t kNativeShortRowBatch = kScoreTile / kNativeShortRowCapacity;
constexpr uint32_t kSmallHeapMaxTopK = 7U;
constexpr uint32_t kDmaBlockBytes = 32U;
constexpr uint32_t kWorkspaceAlignment = 4U * 1024U;
constexpr uint32_t kPairElements = 2U;
constexpr uint32_t kRadixBlockElements = SparseTopK::kHistogramInputElements;

__aicore__ constexpr uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

struct SelectionWorkspace {
    static constexpr uint32_t kScore = 0U;
    static constexpr uint32_t kIndex = kScore + kScoreTile * sizeof(float);
    static constexpr uint32_t kSort = kIndex + kScoreTile * sizeof(uint32_t);
    static constexpr uint32_t kChunk = kSort + kRadixTile * sizeof(uint32_t);
    static constexpr uint32_t kRetained = kChunk + kScoreTile * kPairElements * sizeof(float);
    static constexpr uint32_t kMerge = kRetained + kRadixTile * sizeof(uint32_t) + kDmaBlockBytes;
    static constexpr uint32_t kJointExtra = kMaxTopK / kPairElements;
    static constexpr uint32_t kOrderValues = kMerge + (kRadixTile + kJointExtra) * sizeof(uint32_t);
    static constexpr uint32_t kOrderIds = kOrderValues + kMaxTopK * sizeof(float);
    static constexpr uint32_t kBytes = AlignUp(kOrderIds + kMaxTopK * sizeof(uint32_t), kWorkspaceAlignment);
    static constexpr uint32_t kSingleWindowCapacity = kChunk / sizeof(uint32_t);
};
static_assert(SelectionWorkspace::kOrderIds >= SparseTopK::kWorkspaceBytes);
static_assert(kNativeShortRowBatch * kNativeTopK <= kRadixTile);
constexpr float kInvalidScore = -1.0e29F;
// 相同分数保留逻辑编号升序，稳定基数排序维持平局次序。
constexpr SortConfig kSortKeysDescending{SortType::RADIX_SORT, true};
constexpr Reg::CastTrait kIdIntToFloat = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT,
                                          Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

__aicore__ inline uint32_t Min(uint32_t left, uint32_t right)
{
    return left < right ? left : right;
}

__aicore__ inline uint32_t Align32(uint32_t value)
{
    return (value + 31U) / 32U * 32U;
}

// 将有效候选转为负编号并补负无穷，排序得到编号升序。
struct AscendingSortConstants {
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> limit;
    Reg::RegTensor<uint32_t> topKLimit;
    Reg::RegTensor<uint32_t> invalidId;
    Reg::RegTensor<float> one;
    Reg::RegTensor<float> zeroFloat;
    Reg::RegTensor<float> negativeInfinity;
};

__simd_callee__ inline void InitAscendingSortConstants(AscendingSortConstants &constants,
    uint32_t count, uint32_t topK)
{
    Reg::Duplicate(constants.zero, 0U);
    Reg::Duplicate(constants.limit, count);
    Reg::Duplicate(constants.topKLimit, topK);
    Reg::Duplicate(constants.invalidId, 0xffffffffU);
    Reg::Duplicate(constants.one, 1.0F);
    Reg::Duplicate(constants.zeroFloat, 0.0F);
    Reg::Duplicate(constants.negativeInfinity, -__builtin_inff());
}

__simd_vf__ void PrepareAscendingIdSort(__ubuf__ float *values,
                                        __ubuf__ uint32_t *indices,
                                        __ubuf__ float *validFlags,
                                        __ubuf__ uint32_t *selected,
                                        __ubuf__ uint32_t *keys,
                                        uint32_t count, uint32_t topK,
                                        uint32_t alignedTopK)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    AscendingSortConstants constants;
    InitAscendingSortConstants(constants, count, topK);
    for (uint32_t offset = 0U; offset < alignedTopK; offset += 64U) {
        Reg::RegTensor<uint32_t> ids;
        Reg::RegTensor<uint32_t> positions;
        Reg::RegTensor<uint32_t> safeIds;
        Reg::RegTensor<uint32_t> key;
        Reg::RegTensor<float> idFloat;
        Reg::RegTensor<float> negativeId;
        Reg::RegTensor<float> valid;
        Reg::MaskReg inRange;
        Reg::MaskReg inTopK;
        Reg::MaskReg hasKey;
        Reg::MaskReg keep;
        Reg::LoadAlign<uint32_t>(ids, selected + offset);
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(positions),
                    static_cast<int32_t>(offset));
        Reg::Compare<uint32_t, CMPMODE::LT>(inRange, ids, constants.limit, all);
        Reg::Compare<uint32_t, CMPMODE::LT>(inTopK, positions, constants.topKLimit, all);
        Reg::And(inRange, inRange, inTopK, all);
        Reg::Select(safeIds, ids, constants.zero, inRange);
        Reg::Gather(key, keys, safeIds, all);
        Reg::Compare<uint32_t, CMPMODE::GT>(hasKey, key, constants.zero, all);
        Reg::And(keep, inRange, hasKey, all);
        Reg::Cast<float, int32_t, kIdIntToFloat>(idFloat,
            reinterpret_cast<Reg::RegTensor<int32_t> &>(safeIds), all);
        Reg::Muls(negativeId, idFloat, -1.0F, all);
        Reg::Select(negativeId, negativeId, constants.negativeInfinity, keep);
        Reg::Select(ids, ids, constants.invalidId, keep);
        Reg::Select(valid, constants.one, constants.zeroFloat, keep);
        Reg::StoreAlign<float>(values + offset, negativeId, all);
        Reg::StoreAlign<uint32_t>(indices + offset, ids, all);
        Reg::StoreAlign<float>(validFlags + offset, valid, all);
    }
}

__simd_vf__ void ExtractAscendingIds(__ubuf__ uint32_t *indices,
                                     __ubuf__ uint32_t *sortedPairs,
                                     uint32_t topK)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    for (uint32_t offset = 0U; offset < topK; offset += 64U) {
        Reg::RegTensor<uint32_t> gatherIndex;
        Reg::RegTensor<uint32_t> ids;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(gatherIndex),
                    static_cast<int32_t>(offset));
        Reg::Muls(gatherIndex, gatherIndex, 2U, all);
        Reg::Adds(gatherIndex, gatherIndex, 1U, all);
        Reg::Gather(ids, sortedPairs, gatherIndex, all);
        Reg::StoreAlign<uint32_t>(indices + offset, ids, all);
    }
}

__simd_vf__ void ExtractScoreOrderedIds(__ubuf__ uint32_t *indices,
                                        __ubuf__ uint32_t *sortedPairs,
                                        uint32_t retained, uint32_t topK)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> retainedLimit;
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> invalidId;
    Reg::Duplicate(retainedLimit, retained);
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(invalidId, 0xffffffffU);
    for (uint32_t offset = 0U; offset < topK; offset += 64U) {
        Reg::RegTensor<uint32_t> lanes;
        Reg::RegTensor<uint32_t> positions;
        Reg::RegTensor<uint32_t> ids;
        Reg::MaskReg valid;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(lanes),
                    static_cast<int32_t>(offset));
        Reg::Compare<uint32_t, CMPMODE::LT>(valid, lanes, retainedLimit, all);
        Reg::Muls(positions, lanes, 2U, all);
        Reg::Adds(positions, positions, 1U, all);
        Reg::Select(positions, positions, zero, valid);
        if (retained != 0U) {
            Reg::Gather(ids, sortedPairs, positions, all);
        } else {
            Reg::Duplicate(ids, 0U);
        }
        Reg::Select(ids, ids, invalidId, valid);
        Reg::StoreAlign<uint32_t>(indices + offset, ids, all);
    }
}

__simd_vf__ void PackScoreRadixInput(__ubuf__ uint32_t *dstIds,
                                     __ubuf__ uint32_t *dstKeys,
                                     __ubuf__ uint32_t *srcIds,
                                     __ubuf__ uint32_t *srcKeys,
                                     uint32_t retained, uint32_t alignedCount)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> invalidId;
    Reg::RegTensor<uint32_t> retainedLimit;
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(invalidId, 0xffffffffU);
    Reg::Duplicate(retainedLimit, retained);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint32_t offset = 0U; offset < alignedCount; offset += 64U) {
        Reg::RegTensor<uint32_t> positions;
        Reg::RegTensor<uint32_t> keys;
        Reg::RegTensor<uint32_t> ids;
        Reg::MaskReg inRange;
        Reg::MaskReg hasKey;
        Reg::MaskReg keep;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(positions),
                    static_cast<int32_t>(offset));
        Reg::LoadAlign<uint32_t>(keys, srcKeys + offset);
        Reg::LoadAlign<uint32_t>(ids, srcIds + offset);
        Reg::Compare<uint32_t, CMPMODE::LT>(inRange, positions, retainedLimit, all);
        Reg::Compare<uint32_t, CMPMODE::GT>(hasKey, keys, zero, all);
        Reg::And(keep, inRange, hasKey, all);
        Reg::Select(keys, keys, zero, keep);
        Reg::Select(ids, ids, invalidId, keep);
        Reg::StoreAlign<uint32_t>(dstIds + offset, ids, all);
        Reg::StoreAlign<uint32_t>(dstKeys + offset, keys, all);
    }
}

__simd_vf__ void UnpackScoreRadixIds(__ubuf__ uint32_t *dstIds,
                                     __ubuf__ uint32_t *sortedIds,
                                     __ubuf__ uint32_t *sortedKeys,
                                     uint32_t topK)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> invalidId;
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(invalidId, 0xffffffffU);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint32_t offset = 0U; offset < (topK + 63U) / 64U * 64U; offset += 64U) {
        Reg::RegTensor<uint32_t> ids;
        Reg::RegTensor<uint32_t> keys;
        Reg::MaskReg hasKey;
        Reg::LoadAlign<uint32_t>(ids, sortedIds + offset);
        Reg::LoadAlign<uint32_t>(keys, sortedKeys + offset);
        Reg::Compare<uint32_t, CMPMODE::GT>(hasKey, keys, zero, all);
        Reg::Select(ids, ids, invalidId, hasKey);
        Reg::StoreAlign<uint32_t>(dstIds + offset, ids, all);
    }
}

// 从交错分数与编号中生成升序排序输入及无效尾部。
__simd_vf__ void PrepareSelectedIdsSort(__ubuf__ float *values,
                                        __ubuf__ uint32_t *indices,
                                        __ubuf__ uint32_t *retainedPairs,
                                        uint32_t retained, uint32_t alignedCount)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> retainedLimit;
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> invalidId;
    Reg::RegTensor<float> negativeInfinity;
    Reg::Duplicate(retainedLimit, retained);
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(invalidId, 0xffffffffU);
    Reg::Duplicate(negativeInfinity, -__builtin_inff());
    for (uint32_t offset = 0U; offset < alignedCount; offset += 64U) {
        Reg::RegTensor<uint32_t> lanes;
        Reg::RegTensor<uint32_t> pairPositions;
        Reg::RegTensor<uint32_t> safePositions;
        Reg::RegTensor<uint32_t> ids;
        Reg::RegTensor<float> idFloat;
        Reg::RegTensor<float> negativeId;
        Reg::MaskReg valid;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(lanes),
                    static_cast<int32_t>(offset));
        Reg::Compare<uint32_t, CMPMODE::LT>(valid, lanes, retainedLimit, all);
        Reg::Muls(pairPositions, lanes, 2U, all);
        Reg::Adds(pairPositions, pairPositions, 1U, all);
        Reg::Select(safePositions, pairPositions, zero, valid);
        if (retained != 0U) {
            Reg::Gather(ids, retainedPairs, safePositions, all);
        } else {
            Reg::Duplicate(ids, 0U);
        }
        Reg::Cast<float, int32_t, kIdIntToFloat>(idFloat,
            reinterpret_cast<Reg::RegTensor<int32_t> &>(ids), all);
        Reg::Muls(negativeId, idFloat, -1.0F, all);
        Reg::Select(negativeId, negativeId, negativeInfinity, valid);
        Reg::Select(ids, ids, invalidId, valid);
        Reg::StoreAlign<float>(values + offset, negativeId, all);
        Reg::StoreAlign<uint32_t>(indices + offset, ids, all);
    }
}

// 用 Gather 拼接任意长度的候选，超出范围补零键。
__simd_vf__ void BuildCombinedKeys(__ubuf__ uint32_t *combinedKeys,
                                   __ubuf__ uint32_t *previousKeys,
                                   __ubuf__ uint32_t *incomingKeys,
                                   uint32_t previousCount, uint32_t combinedCount,
                                   uint32_t alignedCount)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> previousLimit;
    Reg::RegTensor<uint32_t> combinedLimit;
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(previousLimit, previousCount);
    Reg::Duplicate(combinedLimit, combinedCount);
    for (uint32_t offset = 0U; offset < alignedCount; offset += 64U) {
        Reg::RegTensor<uint32_t> positions;
        Reg::RegTensor<uint32_t> previousPositions;
        Reg::RegTensor<uint32_t> incomingPositions;
        Reg::RegTensor<uint32_t> previousKey;
        Reg::RegTensor<uint32_t> incomingKey;
        Reg::RegTensor<uint32_t> key;
        Reg::MaskReg inCombined;
        Reg::MaskReg isPrevious;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(positions),
                    static_cast<int32_t>(offset));
        Reg::Compare<uint32_t, CMPMODE::LT>(inCombined, positions, combinedLimit, all);
        Reg::Compare<uint32_t, CMPMODE::LT>(isPrevious, positions, previousLimit, all);
        Reg::Select(previousPositions, positions, zero, isPrevious);
        Reg::Sub(incomingPositions, positions, previousLimit, all);
        Reg::Select(incomingPositions, zero, incomingPositions, isPrevious);
        Reg::Select(incomingPositions, incomingPositions, zero, inCombined);
        Reg::Gather(previousKey, previousKeys, previousPositions, all);
        Reg::Gather(incomingKey, incomingKeys, incomingPositions, all);
        Reg::Select(key, previousKey, incomingKey, isPrevious);
        Reg::Select(key, key, zero, inCombined);
        Reg::StoreAlign<uint32_t>(combinedKeys + offset, key, all);
    }
}

// 合并编号写入临时区，所有旧映射读取完成后再回写。
__simd_vf__ void GatherMergedCandidates(__ubuf__ uint32_t *outKeys,
                                        __ubuf__ uint32_t *mergedIds,
                                        __ubuf__ uint32_t *selectedPositions,
                                        __ubuf__ uint32_t *candidateKeys,
                                        __ubuf__ uint32_t *previousIds,
                                        __ubuf__ uint32_t *incomingIds,
                                        uint32_t previousCount, uint32_t topK)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> previousLimit;
    Reg::RegTensor<uint32_t> topKLimit;
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(previousLimit, previousCount);
    Reg::Duplicate(topKLimit, topK);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint32_t offset = 0U; offset < (topK + 63U) / 64U * 64U; offset += 64U) {
        Reg::RegTensor<uint32_t> positions;
        Reg::RegTensor<uint32_t> lanes;
        Reg::RegTensor<uint32_t> safePositions;
        Reg::RegTensor<uint32_t> oldPositions;
        Reg::RegTensor<uint32_t> newPositions;
        Reg::RegTensor<uint32_t> safeOld;
        Reg::RegTensor<uint32_t> safeNew;
        Reg::RegTensor<uint32_t> key;
        Reg::RegTensor<uint32_t> oldId;
        Reg::RegTensor<uint32_t> newId;
        Reg::RegTensor<uint32_t> id;
        Reg::MaskReg inTopK;
        Reg::MaskReg isOld;
        Reg::LoadAlign<uint32_t>(positions, selectedPositions + offset);
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(lanes),
                    static_cast<int32_t>(offset));
        Reg::Compare<uint32_t, CMPMODE::LT>(inTopK, lanes, topKLimit, all);
        Reg::Select(safePositions, positions, zero, inTopK);
        Reg::Compare<uint32_t, CMPMODE::LT>(isOld, safePositions, previousLimit, all);
        Reg::Select(safeOld, safePositions, zero, isOld);
        Reg::Sub(newPositions, safePositions, previousLimit, all);
        Reg::Select(safeNew, zero, newPositions, isOld);
        Reg::Gather(key, candidateKeys, safePositions, all);
        Reg::Gather(oldId, previousIds, safeOld, all);
        Reg::Gather(newId, incomingIds, safeNew, all);
        Reg::Select(id, oldId, newId, isOld);
        Reg::StoreAlign<uint32_t>(outKeys + offset, key, all);
        Reg::StoreAlign<uint32_t>(mergedIds + offset, id, all);
    }
}

__simd_vf__ void CopyMergedIds(__ubuf__ uint32_t *previousIds,
                               __ubuf__ uint32_t *mergedIds, uint32_t topK)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint32_t offset = 0U; offset < (topK + 63U) / 64U * 64U; offset += 64U) {
        Reg::RegTensor<uint32_t> id;
        Reg::LoadAlign<uint32_t>(id, mergedIds + offset);
        Reg::StoreAlign<uint32_t>(previousIds + offset, id, all);
    }
}

// 块内位置转为全局编号，零键候选留至行末统一处理。
__simd_vf__ void GatherChunkCandidates(__ubuf__ uint32_t *candidateKeys,
                                       __ubuf__ uint32_t *candidateIds,
                                       __ubuf__ uint32_t *selectedPositions,
                                       __ubuf__ uint32_t *chunkKeys,
                                       uint32_t count, uint32_t candidateCount,
                                       uint32_t chunkStart, bool useSelected)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> countLimit;
    Reg::RegTensor<uint32_t> candidateLimit;
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> invalidId;
    Reg::Duplicate(countLimit, count);
    Reg::Duplicate(candidateLimit, candidateCount);
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(invalidId, 0xffffffffU);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint32_t offset = 0U; offset < (candidateCount + 63U) / 64U * 64U; offset += 64U) {
        Reg::RegTensor<uint32_t> lanes;
        Reg::RegTensor<uint32_t> positions;
        Reg::RegTensor<uint32_t> safePositions;
        Reg::RegTensor<uint32_t> keys;
        Reg::RegTensor<uint32_t> ids;
        Reg::MaskReg inCandidates;
        Reg::MaskReg inChunk;
        Reg::MaskReg validKey;
        Reg::MaskReg valid;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(lanes),
                    static_cast<int32_t>(offset));
        if (useSelected) {
            Reg::LoadAlign<uint32_t>(positions, selectedPositions + offset);
        } else {
            Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(positions),
                        static_cast<int32_t>(offset));
        }
        Reg::Compare<uint32_t, CMPMODE::LT>(inCandidates, lanes, candidateLimit, all);
        Reg::Compare<uint32_t, CMPMODE::LT>(inChunk, positions, countLimit, all);
        Reg::And(inChunk, inChunk, inCandidates, all);
        Reg::Select(safePositions, positions, zero, inChunk);
        Reg::Gather(keys, chunkKeys, safePositions, all);
        Reg::Compare<uint32_t, CMPMODE::GT>(validKey, keys, zero, all);
        Reg::And(valid, inChunk, validKey, all);
        Reg::Select(keys, keys, zero, valid);
        Reg::Adds(ids, safePositions, chunkStart, all);
        Reg::Select(ids, ids, invalidId, valid);
        Reg::StoreAlign<uint32_t>(candidateKeys + offset, keys, all);
        Reg::StoreAlign<uint32_t>(candidateIds + offset, ids, all);
    }
}

__simd_vf__ void CopyChunkCandidates(__ubuf__ uint32_t *rankedKeys,
                                     __ubuf__ uint32_t *rankedIds,
                                     __ubuf__ uint32_t *candidateKeys,
                                     __ubuf__ uint32_t *candidateIds,
                                     uint32_t count, uint32_t topK)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint32_t offset = 0U; offset < (topK + 63U) / 64U * 64U; offset += 64U) {
        Reg::RegTensor<uint32_t> keys;
        Reg::RegTensor<uint32_t> ids;
        if (offset < count) {
            Reg::LoadAlign<uint32_t>(keys, candidateKeys + offset);
            Reg::LoadAlign<uint32_t>(ids, candidateIds + offset);
        } else {
            Reg::Duplicate(keys, 0U);
            Reg::Duplicate(ids, 0xffffffffU);
        }
        Reg::StoreAlign<uint32_t>(rankedKeys + offset, keys, all);
        Reg::StoreAlign<uint32_t>(rankedIds + offset, ids, all);
    }
}

// 零键候选排序后转为编号 -1 的无效尾部。
__simd_vf__ void PrepareRankedIdsSort(__ubuf__ float *sortValues,
                                      __ubuf__ uint32_t *sortIds,
                                      __ubuf__ uint32_t *rankedKeys,
                                      uint32_t retained, uint32_t alignedCount)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> retainedLimit;
    Reg::RegTensor<uint32_t> zero;
    Reg::RegTensor<uint32_t> invalidId;
    Reg::RegTensor<float> negativeInfinity;
    Reg::Duplicate(retainedLimit, retained);
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(invalidId, 0xffffffffU);
    Reg::Duplicate(negativeInfinity, -__builtin_inff());
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    for (uint32_t offset = 0U; offset < alignedCount; offset += 64U) {
        Reg::RegTensor<uint32_t> lanes;
        Reg::RegTensor<uint32_t> keys;
        Reg::RegTensor<uint32_t> ids;
        Reg::RegTensor<float> idFloat;
        Reg::RegTensor<float> negativeId;
        Reg::MaskReg inRange;
        Reg::MaskReg hasKey;
        Reg::MaskReg valid;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(lanes),
                    static_cast<int32_t>(offset));
        Reg::LoadAlign<uint32_t>(keys, rankedKeys + offset);
        Reg::LoadAlign<uint32_t>(ids, sortIds + offset);
        Reg::Compare<uint32_t, CMPMODE::LT>(inRange, lanes, retainedLimit, all);
        Reg::Compare<uint32_t, CMPMODE::GT>(hasKey, keys, zero, all);
        Reg::And(valid, inRange, hasKey, all);
        Reg::Cast<float, int32_t, kIdIntToFloat>(idFloat,
            reinterpret_cast<Reg::RegTensor<int32_t> &>(ids), all);
        Reg::Muls(negativeId, idFloat, -1.0F, all);
        Reg::Select(negativeId, negativeId, negativeInfinity, valid);
        Reg::Select(ids, ids, invalidId, valid);
        Reg::StoreAlign<float>(sortValues + offset, negativeId, all);
        Reg::StoreAlign<uint32_t>(sortIds + offset, ids, all);
    }
}

// 小分数或同分的大编号更差，平局优先保留小编号。
__aicore__ inline bool ScoreWorse(float a, uint32_t ai, float b, uint32_t bi)
{
    return a < b || (a == b && ai > bi);
}

#if defined(ORIG_DTYPE_QUERY) && defined(DT_FLOAT16) && defined(DT_BF16) && \
    ((ORIG_DTYPE_QUERY == DT_FLOAT16) || (ORIG_DTYPE_QUERY == DT_BF16))
// 短行中的零分与正分候选处理。
namespace UnderKVectorZero {
enum class Result : uint32_t { NumericFallback=0, Ready=1, ImplementationFailure=2 };
__aicore__ inline LocalTensor<float> Merge(LocalTensor<float> ping,LocalTensor<float> pong,uint32_t a){
 auto src=ping,dst=pong;
 for(uint32_t run=32;run<a;run*=4){
  for(uint32_t start=0;start<a;start+=4*run){uint16_t count[4]={0,0,0,0};uint32_t offset[4]={start,start,start,start};uint32_t lists=0,total=0;
   for(uint32_t j=0;j<4;++j)if(start+j*run<a){offset[j]=start+j*run;count[j]=uint16_t(Min(run,a-offset[j]));++lists;total+=count[j];}
   if(lists==1){Adds<int32_t>(dst[start*2].ReinterpretCast<int32_t>(),src[start*2].ReinterpretCast<int32_t>(),0,total*2);}
   else {MrgSortSrcList<float> input(src[offset[0]*2],src[offset[1]*2],src[offset[2]*2],src[offset[3]*2]);uint32_t consumed[4]={0,0,0,0};MrgSort<float,false>(dst[start*2],input,count,consumed,uint16_t((1U<<lists)-1U),1);}
   PipeBarrier<PIPE_V>();
  }
  auto tmp=src;src=dst;dst=tmp;
 }
 return src;
}
// Isolated mechanism only. Numeric fallback is not an implementation-error recovery.
__simd_vf__ void VZCompact(__ubuf__ uint32_t* out,__ubuf__ uint32_t* keys,
 __ubuf__ uint32_t* raw,__ubuf__ uint32_t* meta,uint32_t h,uint32_t kind,bool values){
 auto all=Reg::CreateMask<uint32_t,Reg::MaskPattern::ALL>();uint32_t oneLane=1;auto single=Reg::UpdateMask<uint32_t>(oneLane);
 Reg::RegTensor<uint32_t> zero,one,zkey,limit,count;Reg::Duplicate(zero,0U);Reg::Duplicate(one,1U);Reg::Duplicate(zkey,0x80000000U);Reg::Duplicate(limit,h);Reg::Duplicate(count,0U);
 Reg::ClearSpr<SpecialPurposeReg::AR>();Reg::UnalignRegForStore align;
 for(uint32_t o=0;o < kNativeTopK; o += SparseTopK::kKeyLanes){Reg::RegTensor<uint32_t> k,lane,bits,packed,inc;Reg::MaskReg range,valid,eq,select;
  Reg::LoadAlign(k,keys+o);Reg::LoadAlign(bits,raw+o);Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane),int32_t(o));
  Reg::Compare<uint32_t,CMPMODE::LT>(range,lane,limit,all);Reg::Compare<uint32_t,CMPMODE::GT>(valid,k,zero,all);Reg::Compare<uint32_t,CMPMODE::EQ>(eq,k,zkey,all);
  if(kind==0)select=eq;else {Reg::Compare<uint32_t,CMPMODE::NE>(select,k,zkey,all);}
  Reg::And(select,select,valid,all);Reg::And(select,select,range,all);Reg::Select(inc,one,zero,select);Reg::Add(count,count,inc,all);
  Reg::Squeeze<uint32_t,Reg::GatherMaskMode::STORE_REG>(packed,values?bits:lane,select);Reg::StoreUnAlign<uint32_t,Reg::PostLiteral::POST_MODE_UPDATE>(out,packed,align);
 }Reg::StoreUnAlignPost(out,align);Reg::RegTensor<uint32_t> sum;Reg::Reduce<Reg::ReduceType::SUM>(sum,count,all);Reg::StoreAlign(meta,sum,single);
}
__simd_vf__ void VZOne(__ubuf__ uint32_t* pairs,__ubuf__ uint32_t* values,__ubuf__ uint32_t* ids){
 auto all=Reg::CreateMask<uint32_t,Reg::MaskPattern::ALL>();Reg::RegTensor<uint32_t> zero,one,lane,odd,v,id,result;Reg::Duplicate(zero,0U);Reg::Duplicate(one,1U);Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane),0);Reg::And(odd,lane,one,all);Reg::MaskReg chooseId;Reg::Compare<uint32_t,CMPMODE::NE>(chooseId,odd,zero,all);Reg::Gather(v,values,zero,all);Reg::Gather(id,ids,zero,all);Reg::Select(result,id,v,chooseId);Reg::StoreAlign(pairs,result,all);
}
__simd_vf__ void VZPad(__ubuf__ uint32_t* values,__ubuf__ uint32_t* ids,uint32_t n){
 auto all=Reg::CreateMask<uint32_t,Reg::MaskPattern::ALL>();Reg::RegTensor<uint32_t> limit,negInf,invalid;Reg::Duplicate(limit,n);Reg::Duplicate(negInf,0xff800000U);Reg::Duplicate(invalid,0xffffffffU);
 for(uint32_t o=0;o < kNativeTopK; o += SparseTopK::kKeyLanes){Reg::RegTensor<uint32_t> lane,v,id;Reg::MaskReg keep;Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane),int32_t(o));Reg::Compare<uint32_t,CMPMODE::LT>(keep,lane,limit,all);Reg::LoadAlign(v,values+o);Reg::LoadAlign(id,ids+o);Reg::Select(v,v,negInf,keep);Reg::Select(id,id,invalid,keep);Reg::StoreAlign(values+o,v,all);Reg::StoreAlign(ids+o,id,all);}
}
__simd_vf__ void VZFlags(__ubuf__ uint32_t* raw,__ubuf__ uint32_t* keys,__ubuf__ uint32_t* meta,uint32_t h){
 auto all=Reg::CreateMask<uint32_t,Reg::MaskPattern::ALL>();uint32_t oneLane=1;auto single=Reg::UpdateMask<uint32_t>(oneLane);Reg::RegTensor<uint32_t> zero,one,limit,magmask,expmask,zkey,sub,pos;Reg::Duplicate(zero,0U);Reg::Duplicate(one,1U);Reg::Duplicate(limit,h);Reg::Duplicate(magmask,0x7fffffffU);Reg::Duplicate(expmask,0x7f800000U);Reg::Duplicate(zkey,0x80000000U);Reg::Duplicate(sub,0U);Reg::Duplicate(pos,0U);
 for(uint32_t o=0;o < kNativeTopK; o += SparseTopK::kKeyLanes){Reg::RegTensor<uint32_t> bits,k,lane,mag,exp,inc;Reg::MaskReg range,a,b,c;Reg::LoadAlign(bits,raw+o);Reg::LoadAlign(k,keys+o);Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane),int32_t(o));Reg::Compare<uint32_t,CMPMODE::LT>(range,lane,limit,all);Reg::And(mag,bits,magmask,all);Reg::And(exp,bits,expmask,all);Reg::Compare<uint32_t,CMPMODE::EQ>(a,exp,zero,all);Reg::Compare<uint32_t,CMPMODE::GT>(b,mag,zero,all);Reg::And(c,a,b,all);Reg::And(c,c,range,all);Reg::Select(inc,one,zero,c);Reg::Add(sub,sub,inc,all);Reg::Compare<uint32_t,CMPMODE::GT>(a,k,zkey,all);Reg::And(a,a,range,all);Reg::Select(inc,one,zero,a);Reg::Add(pos,pos,inc,all);}
 Reg::RegTensor<uint32_t> sum;Reg::Reduce<Reg::ReduceType::SUM>(sum,pos,all);Reg::StoreAlign(meta+16,sum,single);Reg::Reduce<Reg::ReduceType::SUM>(sum,sub,all);Reg::StoreAlign(meta+24,sum,single);
}
__simd_vf__ void VZCheck(__ubuf__ uint32_t* pairs,__ubuf__ uint32_t* keys,__ubuf__ uint32_t* raw,__ubuf__ uint32_t* meta,uint32_t n,uint32_t h){
 auto all=Reg::CreateMask<uint32_t,Reg::MaskPattern::ALL>();uint32_t oneLane=1;auto single=Reg::UpdateMask<uint32_t>(oneLane);Reg::RegTensor<uint32_t> zero,one,limit,hlimit,two,bad,ties;Reg::Duplicate(zero,0U);Reg::Duplicate(one,1U);Reg::Duplicate(limit,n);Reg::Duplicate(hlimit,h);Reg::Duplicate(two,2U);Reg::Duplicate(bad,0U);Reg::Duplicate(ties,0U);
 for(uint32_t o=0;o < kNativeTopK; o += SparseTopK::kKeyLanes){Reg::RegTensor<uint32_t> lane,idx,prev,id,pid,key,pkey,safe,inc,pairbits,expectedbits;Reg::MaskReg range,nonfirst,good,error,equal;Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane),int32_t(o));Reg::Compare<uint32_t,CMPMODE::LT>(range,lane,limit,all);Reg::Mul(idx,lane,two,all);Reg::Add(idx,idx,one,all);Reg::Gather(id,pairs,idx,all);Reg::Compare<uint32_t,CMPMODE::LT>(good,id,hlimit,all);Reg::Select(safe,id,zero,good);Reg::Gather(key,keys,safe,all);Reg::Gather(expectedbits,raw,safe,all);Reg::Mul(idx,lane,two,all);Reg::Gather(pairbits,pairs,idx,all);Reg::Compare<uint32_t,CMPMODE::NE>(error,pairbits,expectedbits,all);Reg::And(error,error,range,all);Reg::Select(inc,one,zero,error);Reg::Add(bad,bad,inc,all);Reg::Compare<uint32_t,CMPMODE::GE>(error,id,hlimit,all);Reg::And(error,error,range,all);Reg::Select(inc,one,zero,error);Reg::Add(bad,bad,inc,all);
  Reg::Compare<uint32_t,CMPMODE::GT>(nonfirst,lane,zero,all);Reg::Sub(prev,lane,one,all);Reg::Select(prev,prev,zero,nonfirst);Reg::Mul(idx,prev,two,all);Reg::Add(idx,idx,one,all);Reg::Gather(pid,pairs,idx,all);Reg::Compare<uint32_t,CMPMODE::LT>(good,pid,hlimit,all);Reg::Select(safe,pid,zero,good);Reg::Gather(pkey,keys,safe,all);Reg::And(nonfirst,nonfirst,range,all);Reg::Compare<uint32_t,CMPMODE::EQ>(error,id,pid,all);Reg::And(error,error,nonfirst,all);Reg::Select(inc,one,zero,error);Reg::Add(bad,bad,inc,all);Reg::Compare<uint32_t,CMPMODE::GT>(error,key,pkey,all);Reg::And(error,error,nonfirst,all);Reg::Select(inc,one,zero,error);Reg::Add(bad,bad,inc,all);Reg::Compare<uint32_t,CMPMODE::EQ>(equal,key,pkey,all);Reg::And(equal,equal,nonfirst,all);Reg::Select(inc,one,zero,equal);Reg::Add(ties,ties,inc,all);
 }Reg::RegTensor<uint32_t> sum;Reg::Reduce<Reg::ReduceType::SUM>(sum,bad,all);Reg::StoreAlign(meta,sum,single);Reg::Reduce<Reg::ReduceType::SUM>(sum,ties,all);Reg::StoreAlign(meta+8,sum,single);
}
__simd_vf__ void VZSplice(__ubuf__ uint32_t* output,__ubuf__ uint32_t* pairs,__ubuf__ uint32_t* zeros,uint32_t p,uint32_t z,uint32_t n){
 auto all=Reg::CreateMask<uint32_t,Reg::MaskPattern::ALL>();uint32_t oneLane=1;auto single=Reg::UpdateMask<uint32_t>(oneLane);Reg::RegTensor<uint32_t> zero,one,two,pv,pz,total,invalid,zv,nv;Reg::Duplicate(zero,0U);Reg::Duplicate(one,1U);Reg::Duplicate(two,2U);Reg::Duplicate(pv,p);Reg::Duplicate(zv,z);Reg::Duplicate(nv,n);Reg::Duplicate(pz,p+z);Reg::Duplicate(total,n+z);Reg::Duplicate(invalid,0xffffffffU);
 for(uint32_t o=0;o < kNativeTopK; o += SparseTopK::kKeyLanes){Reg::RegTensor<uint32_t> lane,npos,zpos,idx,nid,zid,result;Reg::MaskReg positive,zeroRange,valid;Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane),int32_t(o));Reg::Compare<uint32_t,CMPMODE::LT>(positive,lane,pv,all);Reg::Compare<uint32_t,CMPMODE::LT>(zeroRange,lane,pz,all);Reg::Compare<uint32_t,CMPMODE::LT>(valid,lane,total,all);Reg::Sub(npos,lane,zv,all);Reg::Select(npos,lane,npos,positive);Reg::Compare<uint32_t,CMPMODE::LT>(positive,npos,nv,all);Reg::Select(npos,npos,zero,positive);Reg::Mul(idx,npos,two,all);Reg::Add(idx,idx,one,all);Reg::Gather(nid,pairs,idx,all);Reg::Sub(zpos,lane,pv,all);Reg::Compare<uint32_t,CMPMODE::LT>(positive,zpos,zv,all);Reg::Select(zpos,zpos,zero,positive);Reg::Gather(zid,zeros,zpos,all);Reg::Compare<uint32_t,CMPMODE::GE>(positive,lane,pv,all);Reg::And(zeroRange,zeroRange,positive,all);Reg::Select(result,zid,nid,zeroRange);Reg::Select(result,result,invalid,valid);Reg::StoreAlign(output+o,result,all);}
}
__aicore__ inline Result Try(LocalTensor<uint8_t> b, GlobalTensor<float>& scores, uint64_t rowOffset, LocalTensor<uint32_t> outputIds, uint32_t h){
    constexpr uint32_t kRowBytes = kNativeTopK * sizeof(uint32_t);
    constexpr uint32_t kRawOffset = 0U;
    constexpr uint32_t kKeysOffset = kRawOffset + kRowBytes;
    constexpr uint32_t kValuesOffset = kKeysOffset + kRowBytes;
    constexpr uint32_t kIdsOffset = kValuesOffset + kRowBytes;
    constexpr uint32_t kZerosOffset = kIdsOffset + kRowBytes;
    constexpr uint32_t kPingOffset = kZerosOffset + kRowBytes;
    constexpr uint32_t kPongOffset = kPingOffset + kPairElements * kRowBytes;
    constexpr uint32_t kMetaOffset = kPongOffset + kPairElements * kRowBytes;
    static_assert(kMetaOffset + 4U * kDmaBlockBytes <= SelectionWorkspace::kOrderIds);
    auto raw = b[kRawOffset].ReinterpretCast<uint32_t>();
    auto keys = b[kKeysOffset].ReinterpretCast<uint32_t>();
    auto values = b[kValuesOffset].ReinterpretCast<float>();
    auto ids = b[kIdsOffset].ReinterpretCast<uint32_t>();
    auto zeros = b[kZerosOffset].ReinterpretCast<uint32_t>();
    auto ping = b[kPingOffset].ReinterpretCast<float>();
    auto pong = b[kPongOffset].ReinterpretCast<float>();
    auto meta = b[kMetaOffset].ReinterpretCast<uint32_t>();
 Duplicate(raw,0U,kNativeTopK);Duplicate(values.ReinterpretCast<uint32_t>(),0xff800000U,kNativeTopK);Duplicate(ids,0xffffffffU,kNativeTopK);Duplicate(zeros,0xffffffffU,kNativeTopK);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);DataCopyExtParams cp{1U,static_cast<uint32_t>(h * sizeof(float)),0U,0U,0U};DataCopyPadExtParams<float> pad{false,0U,0U,0.f};DataCopyPad(raw.ReinterpretCast<float>(),scores[rowOffset],cp,pad);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);Adds<int32_t>(keys.ReinterpretCast<int32_t>(),raw.ReinterpretCast<int32_t>(),0,kNativeTopK);PipeBarrier<PIPE_V>();VZFlags((__ubuf__ uint32_t*)raw.GetPhyAddr(),(__ubuf__ uint32_t*)keys.GetPhyAddr(),(__ubuf__ uint32_t*)meta.GetPhyAddr(),h);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);if(meta.GetValue(24)){return Result::NumericFallback;}SparseTopK::PrepareFp32Keys((__ubuf__ uint32_t*)keys.GetPhyAddr(), h, kNativeTopK, kInvalidScore);PipeBarrier<PIPE_V>();VZFlags((__ubuf__ uint32_t*)raw.GetPhyAddr(),(__ubuf__ uint32_t*)keys.GetPhyAddr(),(__ubuf__ uint32_t*)meta.GetPhyAddr(),h);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
 if(meta.GetValue(24)){return Result::NumericFallback;}
 VZCompact((__ubuf__ uint32_t*)zeros.GetPhyAddr(),(__ubuf__ uint32_t*)keys.GetPhyAddr(),(__ubuf__ uint32_t*)raw.GetPhyAddr(),(__ubuf__ uint32_t*)meta.GetPhyAddr(),h,0,false);PipeBarrier<PIPE_V>();VZCompact((__ubuf__ uint32_t*)ids.GetPhyAddr(),(__ubuf__ uint32_t*)keys.GetPhyAddr(),(__ubuf__ uint32_t*)raw.GetPhyAddr(),(__ubuf__ uint32_t*)meta[8].GetPhyAddr(),h,1,false);PipeBarrier<PIPE_V>();VZCompact((__ubuf__ uint32_t*)values.GetPhyAddr(),(__ubuf__ uint32_t*)keys.GetPhyAddr(),(__ubuf__ uint32_t*)raw.GetPhyAddr(),(__ubuf__ uint32_t*)meta[8].GetPhyAddr(),h,1,true);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);uint32_t z=meta.GetValue(0),n=meta.GetValue(8),p=meta.GetValue(16);uint32_t a=(n+31)/32*32;
 VZPad((__ubuf__ uint32_t*)values.GetPhyAddr(),(__ubuf__ uint32_t*)ids.GetPhyAddr(),n);PipeBarrier<PIPE_V>();
 if(n==1){VZOne((__ubuf__ uint32_t*)ping.GetPhyAddr(),(__ubuf__ uint32_t*)values.GetPhyAddr(),(__ubuf__ uint32_t*)ids.GetPhyAddr());PipeBarrier<PIPE_V>();}else if(n>1){Sort32<float>(ping,values,ids,a/32);PipeBarrier<PIPE_V>();}else{Duplicate(ping.ReinterpretCast<uint32_t>(),0xffffffffU,kNativeTopK * kPairElements);PipeBarrier<PIPE_V>();}auto sorted=n?Merge(ping,pong,a):ping;
 VZCheck((__ubuf__ uint32_t*)sorted.GetPhyAddr(),(__ubuf__ uint32_t*)keys.GetPhyAddr(),(__ubuf__ uint32_t*)raw.GetPhyAddr(),(__ubuf__ uint32_t*)meta.GetPhyAddr(),n,h);PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);if(meta.GetValue(0)){return Result::ImplementationFailure;}if(meta.GetValue(8)){return Result::NumericFallback;}
 VZSplice((__ubuf__ uint32_t*)outputIds.GetPhyAddr(),(__ubuf__ uint32_t*)sorted.GetPhyAddr(),(__ubuf__ uint32_t*)zeros.GetPhyAddr(),p,z,n);PipeBarrier<PIPE_V>();return Result::Ready;
}

}

#endif

// IRS chooses the radix implementation when instantiating its device entry.
enum class Implementation : uint8_t {
    RadixMerge,
    HighPrefixFp32,
};

// Preserve the established route for specialized IRS row policies as well.
inline constexpr Implementation kDefaultImplementation = Implementation::HighPrefixFp32;

class Selector final {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && \
    defined(ORIG_DTYPE_QUERY) && defined(DT_FLOAT16) && defined(DT_BF16) && \
    ((ORIG_DTYPE_QUERY == DT_FLOAT16) || (ORIG_DTYPE_QUERY == DT_BF16))
    friend struct IdCompactBatchPolicy;
    friend struct SingleWindowPolicy;
    friend struct ShortRowPolicy;
    friend struct JointRadixPolicy;
    __aicore__ inline void SelectHistoryTopKJoint(uint32_t group, uint32_t token,
        uint32_t history, uint32_t &retained);
    __aicore__ inline void PrepareJointIdsSort(uint32_t retained, uint32_t aligned);
#endif
public:
    __aicore__ inline void Init(GM_ADDR scores, GM_ADDR actualKvLengths,
                                GM_ADDR cuSeqLens, GM_ADDR selectedIds,
                                uint32_t groups, uint32_t tokens,
                                uint64_t scoreStrideGroup,
                                uint64_t scoreStrideToken, uint32_t rout,
                                uint32_t maxRegionLen, uint32_t topK,
                                uint32_t maskMode, uint32_t regionSize,
                                bool sortIds, uint32_t numRequests,
                                uint32_t usedCubeCores, TPipe *pipe,
                                bool alignedSelection = false,
                                uint32_t batchBegin = 0U, uint32_t batchCount = 0U)
    {
        groups_ = groups;
        totalTokens_ = tokens;
        tokens_ = batchCount != 0U ? batchCount : tokens;
        batchBegin_ = batchBegin;
        batchScratch_ = alignedSelection && batchCount != 0U;
        scoreStrideGroup_ = scoreStrideGroup;
        scoreStrideToken_ = scoreStrideToken;
        rout_ = rout;
        maxRegionLen_ = Min(maxRegionLen, rout);
        topK_ = topK;
        selectedStride_ = alignedSelection ? (topK + 7U) / 8U * 8U : topK;
        maskMode_ = maskMode;
        regionSize_ = regionSize;
        sortIds_ = sortIds;
        numRequests_ = numRequests;
        cachedRequest_ = numRequests;
        usedCubeCores_ = usedCubeCores;
        pipe_ = pipe;
        scores_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scores));
        selectedIds_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(selectedIds));
        if (maskMode_ == 3U) {
            actualKvLengths_ = reinterpret_cast<__gm__ int32_t *>(actualKvLengths);
            cuSeqLens_ = reinterpret_cast<__gm__ int32_t *>(cuSeqLens);
        }
        // 评分释放并重置流水后，为每个 AIV 独立申请工作区。
        pipe_->InitBuffer(ub_, SelectionWorkspace::kBytes);
        LocalTensor<uint8_t> base = ub_.Get<uint8_t>();
        score_ = base[SelectionWorkspace::kScore].ReinterpretCast<float>();
        index_ = base[SelectionWorkspace::kIndex].ReinterpretCast<uint32_t>();
        sortTmp_ = base[SelectionWorkspace::kSort].ReinterpretCast<float>();
        chunkSorted_ = base[SelectionWorkspace::kChunk].ReinterpretCast<float>();
        retained_ = base[SelectionWorkspace::kRetained].ReinterpretCast<float>();
        merge_ = base[SelectionWorkspace::kMerge].ReinterpretCast<float>();
        orderValues_ = base[SelectionWorkspace::kOrderValues].ReinterpretCast<float>();
        orderIndices_ = base[SelectionWorkspace::kOrderIds].ReinterpretCast<uint32_t>();
    }

    template <Implementation implementation>
    struct DefaultRowPolicy {
        __aicore__ inline void ProcessRow(Selector &selector, uint32_t row)
        {
            selector.ProcessRow<implementation>(row);
        }
    };

    template <Implementation implementation = kDefaultImplementation>
    __aicore__ inline void Process()
    {
        DefaultRowPolicy<implementation> policy;
        ProcessRows(policy);
    }

    // All policies preserve the same DMA ownership for adjacent unaligned rows.
    template <class RowPolicy>
    __aicore__ inline void ProcessRows(RowPolicy &policy)
    {
        if (topK_ == 0U || topK_ > kMaxTopK || groups_ == 0U ||
            tokens_ == 0U || usedCubeCores_ == 0U) {
            return;
        }
        // 每个 AIV 独占行条带，使用原始工作核编号。
        const uint32_t workerCount = usedCubeCores_ * GetTaskRation();
        const uint32_t worker = GetBlockIdx();
        if (worker >= workerCount) {
            return;
        }
        const uint64_t rows = static_cast<uint64_t>(groups_) * tokens_;
        // 相邻短行共享搬运单元时，由同一 AIV 写入以避免竞争。
        uint32_t rowsPerUnit = 1U;
        while ((rowsPerUnit * selectedStride_) % 8U != 0U) {
            ++rowsPerUnit;
        }
        const uint64_t rowUnits = (rows + rowsPerUnit - 1U) / rowsPerUnit;
        for (uint64_t unit = worker; unit < rowUnits; unit += workerCount) {
            const uint64_t begin = unit * rowsPerUnit;
            const uint64_t end = Min(static_cast<uint32_t>(rows - begin), rowsPerUnit) + begin;
            for (uint64_t row = begin; row < end; ++row) {
                policy.ProcessRow(*this, static_cast<uint32_t>(row));
            }
        }
    }

private:
    __aicore__ inline uint32_t FindRequest(uint32_t token) const
    {
        // 请求前缀和单调，异常输入由 Host 校验。
        if (numRequests_ <= 1U) {
            return 0U;
        }
        uint32_t lo = 0U;
        uint32_t hi = numRequests_;
        while (lo + 1U < hi) {
            const uint32_t mid = (lo + hi) / 2U;
            if (cuSeqLens_[mid] <= static_cast<int32_t>(token)) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    __aicore__ inline uint32_t HistoryLength(uint32_t token) const
    {
        if (maskMode_ != 3U) {
            return maxRegionLen_;
        }
        if (numRequests_ == 0U) {
            return 0U;
        }
        const uint32_t request = FindRequest(token);
        const int64_t requestStart = cuSeqLens_ == nullptr ? 0 : cuSeqLens_[request];
        const int64_t localToken = static_cast<int64_t>(token) - requestStart;
        // Share scoring's zero-based position derived from the effective KV length.
        if (cachedRequest_ != request) {
            const uint32_t requestEnd = cuSeqLens_ == nullptr
                ? totalTokens_ : static_cast<uint32_t>(cuSeqLens_[request + 1U]);
            cachedFirstPosition_ = IndexerRegionScorePaged::RequestFirstTokenPosition(
                actualKvLengths_[request], static_cast<uint32_t>(requestStart), requestEnd);
            cachedRequest_ = request;
        }
        const int64_t position = cachedFirstPosition_ + (localToken < 0 ? 0 : localToken);
        if (position <= 0 || regionSize_ == 0U) {
            return 0U;
        }
        const uint64_t visible = static_cast<uint64_t>(position) / regionSize_;
        return visible >= maxRegionLen_ ? maxRegionLen_ : static_cast<uint32_t>(visible);
    }

    __aicore__ inline void HeapSwap(uint32_t left, uint32_t right)
    {
        const float score = orderValues_.GetValue(left);
        const uint32_t id = orderIndices_.GetValue(left);
        orderValues_.SetValue(left, orderValues_.GetValue(right));
        orderIndices_.SetValue(left, orderIndices_.GetValue(right));
        orderValues_.SetValue(right, score);
        orderIndices_.SetValue(right, id);
    }

    // 最小堆根保存当前最差候选。
    __aicore__ inline void SiftScoreHeap(uint32_t root, uint32_t count)
    {
        while (2U * root + 1U < count) {
            uint32_t child = 2U * root + 1U;
            if (child + 1U < count &&
                ScoreWorse(orderValues_.GetValue(child + 1U), orderIndices_.GetValue(child + 1U),
                           orderValues_.GetValue(child), orderIndices_.GetValue(child))) {
                ++child;
            }
            if (!ScoreWorse(orderValues_.GetValue(child), orderIndices_.GetValue(child),
                            orderValues_.GetValue(root), orderIndices_.GetValue(root))) {
                return;
            }
            HeapSwap(child, root);
            root = child;
        }
    }

    __aicore__ inline void Consider(float value, uint32_t id, uint32_t &retained)
    {
        // 有序比较同时排除 NaN 和无效分数哨兵。
        if (!(value > kInvalidScore)) {
            return;
        }
        if (retained < topK_) {
            uint32_t pos = retained++;
            while (pos > 0U) {
                const uint32_t parent = (pos - 1U) / 2U;
                if (!ScoreWorse(value, id, orderValues_.GetValue(parent),
                                orderIndices_.GetValue(parent))) {
                    break;
                }
                orderValues_.SetValue(pos, orderValues_.GetValue(parent));
                orderIndices_.SetValue(pos, orderIndices_.GetValue(parent));
                pos = parent;
            }
            orderValues_.SetValue(pos, value);
            orderIndices_.SetValue(pos, id);
        } else if (ScoreWorse(orderValues_.GetValue(0U), orderIndices_.GetValue(0U),
                              value, id)) {
            orderValues_.SetValue(0U, value);
            orderIndices_.SetValue(0U, id);
            SiftScoreHeap(0U, retained);
        }
    }

    template <Implementation implementation = kDefaultImplementation>
    __aicore__ inline void SelectHistoryTopK(uint32_t group, uint32_t token,
                                              uint32_t history, uint32_t &retained,
                                              uint32_t &directMode)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && \
    defined(ORIG_DTYPE_QUERY) && defined(DT_FLOAT16) && defined(DT_BF16) && \
    ((ORIG_DTYPE_QUERY == DT_FLOAT16) || (ORIG_DTYPE_QUERY == DT_BF16))
        terminalIdOnly_ = false;
        if (topK_ > kSmallHeapMaxTopK && topK_ <= SelectionWorkspace::kJointExtra && history > kRadixTile) {
            SelectHistoryTopKJoint(group, token, history, retained);
            return;
        }
#endif
        // IRS owns the row-length choice between independent TopK entries.
        if (topK_ > kSmallHeapMaxTopK) {
            if constexpr (implementation == Implementation::HighPrefixFp32) {
                if (sortIds_ && history > topK_ &&
                    history <= SparseIndexer::Common::Primitive::TopK::kMaxElements &&
                    TryHighPrefixTopK(group, token, history, retained)) {
                    return;
                }
            }
            if (history <= topK_) {
                SelectHistoryTopKShortRow(group, token, history, retained);
            } else if (history <= kRadixTile) {
                SelectHistoryTopKSingleWindow(group, token, history, retained);
            } else {
                SelectHistoryTopKMultiWindow(group, token, history, retained);
            }
        } else if (history > kRadixTile) {
            SelectHistoryTopKRadixHeapSmall(group, token, history, retained);
        } else if (history >= kScoreTile) {
            SelectHistoryTopKRadix(group, token, history, retained, directMode);
        } else {
            SelectHistoryTopKSortHeap(group, token, history, retained);
        }
    }

    __aicore__ inline void SelectHistoryTopKSortHeap(uint32_t group, uint32_t token,
                                                      uint32_t history, uint32_t &retained)
    {
        retained = 0U;
        const uint64_t rowOffset = ScoreOffset(group, token);
        for (uint32_t start = 0U; start < history; start += kScoreTile) {
            const uint32_t count = Min(kScoreTile, history - start);
            DataCopyExtParams params{1U, static_cast<uint32_t>(count * sizeof(float)), 0U, 0U, 0U};
            DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
            DataCopyPad(score_, scores_[rowOffset + start], params, pad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            // 完整块在向量侧排序，边界同分时扫描全块保证小编号优先。
            bool uniform = false;
            bool boundaryTie = true;
            if (count == kScoreTile) {
                SortFullScoreTile(start, count, uniform, boundaryTie);
            } else {
                for (uint32_t i = 0U; i < count; ++i) {
                    const float value = score_.GetValue(i);
                    score_.SetValue(i, value > kInvalidScore ? value : -__builtin_inff());
                }
            }
            CollectSortedTile(start, count, uniform, boundaryTie, retained);
        }
        // 逆序提取最小堆，得到分数降序且同分编号升序。
        DrainScoreHeap(retained);
    }

    // 逆序提取最小堆，保持同分小编号优先。
    __aicore__ inline void DrainScoreHeap(uint32_t retained)
    {
        LocalTensor<uint32_t> retainedIds = retained_.ReinterpretCast<uint32_t>();
        for (uint32_t remaining = retained; remaining > 0U; --remaining) {
            retained_.SetValue(2U * (remaining - 1U), orderValues_.GetValue(0U));
            retainedIds.SetValue(2U * (remaining - 1U) + 1U, orderIndices_.GetValue(0U));
            if (remaining > 1U) {
                orderValues_.SetValue(0U, orderValues_.GetValue(remaining - 1U));
                orderIndices_.SetValue(0U, orderIndices_.GetValue(remaining - 1U));
                SiftScoreHeap(0U, remaining - 1U);
            }
        }
    }

    __aicore__ inline void SortFullScoreTile(uint32_t start, uint32_t count, bool &uniform, bool &boundaryTie)
    {
        LocalTensor<uint8_t> validMask = sortTmp_.ReinterpretCast<uint8_t>();
        CompareScalar(validMask, score_, kInvalidScore, CMPMODE::GT, kScoreTile);
        PipeBarrier<PIPE_V>();
        Select(score_, validMask, score_, -__builtin_inff(),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, kScoreTile);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        ReduceMax(sortTmp_, score_, sortTmp_[128], static_cast<int32_t>(count), false);
        PipeBarrier<PIPE_V>();
        ReduceMin(sortTmp_[64], score_, sortTmp_[128], static_cast<int32_t>(count), false);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        uniform = sortTmp_.GetValue(0U) == sortTmp_.GetValue(64U);
        if (!uniform) {
            ArithProgression<int32_t>(index_.ReinterpretCast<int32_t>(),
                                      static_cast<int32_t>(start), 1, kScoreTile);
            // 编号写入后同步至向量侧，排序结果另行同步至标量侧。
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            Sort<float, true>(chunkSorted_, score_, index_, sortTmp_, kScoreTile / 32U);
            const uint32_t localK = Min(topK_, count);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            boundaryTie = localK < count &&
                chunkSorted_.GetValue(2U * (localK - 1U)) ==
                chunkSorted_.GetValue(2U * localK);
        }
    }

    __aicore__ inline void CollectSortedTile(uint32_t start, uint32_t count, bool uniform, bool boundaryTie,
        uint32_t &retained)
    {
        const uint32_t localK = Min(topK_, count);
        const uint32_t candidates = count == kScoreTile && !uniform && !boundaryTie
            ? localK : count;
        LocalTensor<uint32_t> sortedIds = chunkSorted_.ReinterpretCast<uint32_t>();
        for (uint32_t i = 0U; i < candidates; ++i) {
            const float value = count == kScoreTile && !uniform && !boundaryTie
                ? chunkSorted_.GetValue(2U * i) : score_.GetValue(i);
            const uint32_t id = count == kScoreTile && !uniform && !boundaryTie
                ? sortedIds.GetValue(2U * i + 1U) : start + i;
            Consider(value, id, retained);
        }
    }

    __aicore__ inline void FilterRadixCandidates(LocalTensor<uint32_t> ids,
        LocalTensor<uint32_t> keys, uint32_t aligned)
    {
        LocalTensor<uint32_t> hist = chunkSorted_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> bucket0 = hist[1U * SparseTopK::kHistogramBins];
        LocalTensor<uint32_t> bucket1 = hist[2U * SparseTopK::kHistogramBins];
        LocalTensor<uint32_t> bucket2 = hist[3U * SparseTopK::kHistogramBins];
        LocalTensor<uint32_t> bucket3 = hist[4U * SparseTopK::kHistogramBins];
        LocalTensor<uint32_t> kth = hist[5U * SparseTopK::kHistogramBins];
        SparseTopK::SelectRadixWindow<false>(
            ids, index_, keys, hist, bucket0, bucket1, bucket2, bucket3, kth, topK_, aligned);
    }

    __aicore__ inline void CollectRadixHeapCandidates(LocalTensor<uint32_t> ids, LocalTensor<uint32_t> keys,
        uint32_t count, uint32_t start, uint32_t &retained)
    {
        const uint32_t candidateCount = count > topK_ ? topK_ : count;
        for (uint32_t i = 0U; i < candidateCount; ++i) {
            const uint32_t localId = count > topK_ ? ids.GetValue(i) : i;
            if (localId < count) {
                const uint32_t key = keys.GetValue(localId);
                if (key != 0U) {
                    Consider(DecodeRadixKey(key), start + localId, retained);
                }
            }
        }
    }

    __aicore__ inline void SortFinalRadixScores(uint32_t retained, LocalTensor<uint32_t> rankedKeys)
    {
        const uint32_t sortAligned = Align32(topK_);
        LocalTensor<uint32_t> inputIds = sortTmp_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> inputKeys = inputIds[kMaxTopK];
        LocalTensor<uint32_t> outputIds = retained_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> outputKeys = outputIds[kMaxTopK];
        LocalTensor<uint8_t> radixScratch = merge_.ReinterpretCast<uint8_t>();
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        PackScoreRadixInput(
            reinterpret_cast<__ubuf__ uint32_t *>(inputIds.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(inputKeys.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(orderIndices_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(rankedKeys.GetPhyAddr()),
            retained, sortAligned);
        PipeBarrier<PIPE_V>();
        Sort<uint32_t, uint32_t, false, kSortKeysDescending>(
            outputKeys, outputIds, inputKeys, inputIds, radixScratch, sortAligned);
        PipeBarrier<PIPE_V>();
        UnpackScoreRadixIds(
            reinterpret_cast<__ubuf__ uint32_t *>(orderIndices_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(outputIds.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(outputKeys.GetPhyAddr()), topK_);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        scoreIdsReady_ = true;
    }

    __aicore__ inline void MergeRadixCandidates(LocalTensor<uint32_t> ids, LocalTensor<uint32_t> keys,
        LocalTensor<uint32_t> rankedKeys,
        LocalTensor<uint32_t> incomingKeys, LocalTensor<uint32_t> incomingIds,
        LocalTensor<uint32_t> mergedIds, uint32_t retained, uint32_t combined)
    {
        const uint32_t mergeAligned = AlignUp(combined, kRadixBlockElements);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        BuildCombinedKeys(reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(rankedKeys.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(incomingKeys.GetPhyAddr()),
            retained, combined, mergeAligned);
        PipeBarrier<PIPE_V>();
        FilterRadixCandidates(ids, keys, mergeAligned);
        PipeBarrier<PIPE_V>();
        GatherMergedCandidates(reinterpret_cast<__ubuf__ uint32_t *>(rankedKeys.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(mergedIds.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(ids.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(orderIndices_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(incomingIds.GetPhyAddr()),
            retained, topK_);
        PipeBarrier<PIPE_V>();
        CopyMergedIds(reinterpret_cast<__ubuf__ uint32_t *>(orderIndices_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(mergedIds.GetPhyAddr()), topK_);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        // 所有旧映射读取完成后回写，同分候选保持块间编号升序。
    }

    __aicore__ inline void AppendRadixCandidates(LocalTensor<uint32_t> rankedKeys, LocalTensor<uint32_t> incomingKeys,
        LocalTensor<uint32_t> incomingIds, uint32_t retained, uint32_t incoming)
    {
        if (retained == 0U) {
            CopyChunkCandidates(
                reinterpret_cast<__ubuf__ uint32_t *>(rankedKeys.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(orderIndices_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(incomingKeys.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(incomingIds.GetPhyAddr()),
                incoming, topK_);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
        } else {
            for (uint32_t i = 0U; i < incoming; ++i) {
                rankedKeys.SetValue(retained + i, incomingKeys.GetValue(i));
                orderIndices_.SetValue(retained + i, incomingIds.GetValue(i));
            }
        }
    }

    template <bool WaitForScalar = true>
    __aicore__ inline void PrepareIncomingRadixCandidates(uint64_t rowOffset, uint32_t start, uint32_t history,
        LocalTensor<uint32_t> ids,
        LocalTensor<uint32_t> keys, LocalTensor<uint32_t> incomingKeys,
        LocalTensor<uint32_t> incomingIds, uint32_t &candidateCount)
    {
        const uint32_t count = Min(kRadixTile, history - start);
        const uint32_t aligned = AlignUp(count, kRadixBlockElements);
        DataCopyExtParams copy{1U, static_cast<uint32_t>(count * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
        DataCopyPad(merge_, scores_[rowOffset + start], copy, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        SparseTopK::PrepareFp32Keys(reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr()), count, aligned, kInvalidScore);
        PipeBarrier<PIPE_V>();
        const bool reduceChunk = count > topK_;
        if (reduceChunk) {
            // 上一轮标量写入的索引工作区复用前同步至向量侧。
            if (start != 0U) {
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
            }
            FilterRadixCandidates(ids, keys, aligned);
        }
        candidateCount = reduceChunk ? topK_ : count;
        GatherChunkCandidates(
            reinterpret_cast<__ubuf__ uint32_t *>(incomingKeys.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(incomingIds.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(ids.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr()),
            count, candidateCount, start, reduceChunk);
        if constexpr (WaitForScalar) {
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
        } else {
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CollectShortRadixTile(uint64_t rowOffset, uint32_t start, uint32_t count, uint32_t &retained)
    {
        for (uint32_t offset = 0U; offset < count; offset += kScoreTile) {
            const uint32_t n = Min(kScoreTile, count - offset);
            DataCopyExtParams copy{1U, static_cast<uint32_t>(n * sizeof(float)), 0U, 0U, 0U};
            DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
            DataCopyPad(score_, scores_[rowOffset + start + offset], copy, pad);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            for (uint32_t i = 0U; i < n; ++i) {
                Consider(score_.GetValue(i), start + offset + i, retained);
            }
        }
    }

    __aicore__ inline void CheckUniformRadixTile(uint32_t count, uint32_t &retained, uint32_t &directMode)
    {
        // 最大最小值及求和联合检查同分有效行，拒绝 NaN。
        ReduceMax(score_, sortTmp_, merge_, static_cast<int32_t>(count), false);
        PipeBarrier<PIPE_V>();
        ReduceMin(score_[2], sortTmp_, merge_, static_cast<int32_t>(count), false);
        PipeBarrier<PIPE_V>();
        ReduceSum(score_[4], sortTmp_, merge_, static_cast<int32_t>(count));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float minimum = score_.GetValue(2U);
        const float maximum = score_.GetValue(0U);
        const float total = score_.GetValue(4U);
        if (minimum == maximum && maximum > kInvalidScore &&
            total == maximum * static_cast<float>(count)) {
            retained = topK_;
            directMode = 1U;
            // 单块路径由循环后的公共出口收尾。
            return;
        }
    }

    __aicore__ inline void TryAscendingRadixIds(LocalTensor<uint32_t> ids, LocalTensor<uint32_t> keys, uint32_t count,
        uint32_t &retained, uint32_t &directMode)
    {
        // 全有效候选在向量侧排序，无效候选沿用标量回退。
        const uint32_t alignedTopK = (topK_ + 63U) / 64U * 64U;
        PipeBarrier<PIPE_V>();
        PrepareAscendingIdSort(
            reinterpret_cast<__ubuf__ float *>(orderValues_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(orderIndices_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(score_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(ids.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr()),
            count, topK_, alignedTopK);
        PipeBarrier<PIPE_V>();
        // 有效性结果置于排序输入之外，避免覆盖完整评分块。
        ReduceMin(orderValues_[alignedTopK], score_, chunkSorted_,
                  static_cast<int32_t>(topK_), false);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if (orderValues_.GetValue(alignedTopK) == 1.0F) {
            Sort<float, true>(merge_, orderValues_, orderIndices_,
                              sortTmp_, alignedTopK / 32U);
            PipeBarrier<PIPE_V>();
            ExtractAscendingIds(
                reinterpret_cast<__ubuf__ uint32_t *>(orderIndices_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(merge_.GetPhyAddr()), topK_);
            retained = topK_;
            directMode = 2U;
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            return;
        }
    }

    __aicore__ inline void CollectRadixSelection(LocalTensor<uint32_t> ids, LocalTensor<uint32_t> keys,
        LocalTensor<uint32_t> selected,
        uint32_t count, uint32_t start, bool singleTrunk, uint32_t &retained)
    {
        for (uint32_t i = 0U; i < topK_; ++i) {
            const uint32_t localId = ids.GetValue(i);
            if (localId < count && keys.GetValue(localId) != 0U) {
                if (singleTrunk) {
                    retained_.SetValue(2U * retained, 0.0F);
                    selected.SetValue(2U * retained + 1U, localId);
                    ++retained;
                } else {
                    Consider(DecodeRadixKey(keys.GetValue(localId)), start + localId, retained);
                }
            }
        }
    }

    __aicore__ inline void SortSingleRadixSelection(LocalTensor<uint32_t> keys, LocalTensor<uint32_t> selected,
        uint32_t retained)
    {
        const uint32_t alignedK = Align32(topK_);
        for (uint32_t i = 0U; i < alignedK; ++i) {
            if (i < retained) {
                const uint32_t id = selected.GetValue(2U * i + 1U);
                score_.SetValue(i, DecodeRadixKey(keys.GetValue(id)));
                index_.SetValue(i, id);
            } else {
                score_.SetValue(i, -__builtin_inff());
                index_.SetValue(i, 0xffffffffU);
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Sort<float, true>(chunkSorted_, score_, index_, sortTmp_, alignedK / 32U);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        LocalTensor<uint32_t> sorted = chunkSorted_.ReinterpretCast<uint32_t>();
        RestoreScoreTieOrder(sorted, selected, retained);
    }

    __aicore__ inline void RestoreScoreTieOrder(LocalTensor<uint32_t> sorted, LocalTensor<uint32_t> selected,
        uint32_t retained)
    {
        // 排序无稳定平局保证，单独恢复同分候选的小编号优先。
        uint32_t begin = 0U;
        while (begin < retained) {
            const float value = chunkSorted_.GetValue(2U * begin);
            uint32_t end = begin + 1U;
            while (end < retained && chunkSorted_.GetValue(2U * end) == value) {
                ++end;
            }
            if (end - begin >= 32U) {
                // 长同分段使用负编号向量排序，避免标量插入的平方开销。
                SortLongScoreTie(sorted, selected, begin, end);
            } else {
                SortShortScoreTie(sorted, selected, begin, end);
            }
            begin = end;
        }
    }

    __aicore__ inline void SortLongScoreTie(LocalTensor<uint32_t> sorted, LocalTensor<uint32_t> selected,
        uint32_t begin, uint32_t end)
    {
        const uint32_t tieCount = end - begin;
        const uint32_t tieAligned = Align32(tieCount);
        for (uint32_t i = 0U; i < tieAligned; ++i) {
            const uint32_t id = i < tieCount
                ? sorted.GetValue(2U * (begin + i) + 1U) : 0xffffffffU;
            score_.SetValue(i, i < tieCount
                ? -static_cast<float>(id) : -__builtin_inff());
            index_.SetValue(i, id);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Sort<float, true>(merge_, score_, index_, sortTmp_, tieAligned / 32U);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        LocalTensor<uint32_t> sortedTie = merge_.ReinterpretCast<uint32_t>();
        for (uint32_t i = 0U; i < tieCount; ++i) {
            selected.SetValue(2U * (begin + i) + 1U,
                              sortedTie.GetValue(2U * i + 1U));
        }
    }

    __aicore__ inline void SortShortScoreTie(LocalTensor<uint32_t> sorted, LocalTensor<uint32_t> selected,
        uint32_t begin, uint32_t end)
    {
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t id = sorted.GetValue(2U * i + 1U);
            uint32_t pos = i;
            while (pos > begin && selected.GetValue(2U * (pos - 1U) + 1U) > id) {
                selected.SetValue(2U * pos + 1U,
                                  selected.GetValue(2U * (pos - 1U) + 1U));
                --pos;
            }
            selected.SetValue(2U * pos + 1U, id);
        }
    }

    __aicore__ inline void LoadRadixScoreTile(uint64_t offset, uint32_t count)
    {
        DataCopyExtParams copy{1U, static_cast<uint32_t>(count * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
        DataCopyPad(sortTmp_, scores_[offset], copy, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline float DecodeRadixKey(uint32_t key) const
    {
        const uint32_t bits = (key & 0x80000000U) != 0U
            ? key ^ 0x80000000U : ~key;
        return __builtin_bit_cast(float, bits);
    }

    // 小 K 每块基数筛选后合入小堆，避免重复基数合并。
    __aicore__ inline void SelectHistoryTopKRadixHeapSmall(uint32_t group, uint32_t token,
                                                           uint32_t history, uint32_t &retained)
    {
        const uint64_t rowOffset = ScoreOffset(group, token);
        LocalTensor<uint32_t> ids = sortTmp_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> keys = merge_.ReinterpretCast<uint32_t>();
        retained = 0U;
        for (uint32_t start = 0U; start < history; start += kRadixTile) {
            const uint32_t count = Min(kRadixTile, history - start);
            const uint32_t aligned = AlignUp(count, kRadixBlockElements);
            DataCopyExtParams copy{1U, static_cast<uint32_t>(count * sizeof(float)), 0U, 0U, 0U};
            DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
            DataCopyPad(merge_, scores_[rowOffset + start], copy, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            SparseTopK::PrepareFp32Keys(reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr()), count, aligned, kInvalidScore);
            PipeBarrier<PIPE_V>();
            if (count > topK_) {
                FilterRadixCandidates(ids, keys, aligned);
            }
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            CollectRadixHeapCandidates(ids, keys, count, start, retained);
        }
        DrainScoreHeap(retained);
    }

    __aicore__ inline bool TryHighPrefixTopK(uint32_t group, uint32_t token,
        uint32_t history, uint32_t &retained)
    {
        auto base = ub_.Get<uint8_t>();
        auto keys = base.ReinterpretCast<uint32_t>();
        const uint32_t aligned = AlignUp(history, kRadixBlockElements);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyExtParams copy{1U, static_cast<uint32_t>(history * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
        DataCopyPad(keys.ReinterpretCast<float>(), scores_[ScoreOffset(group, token)], copy, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        SparseTopK::PrepareFp32Keys(reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr()), history, aligned, kInvalidScore);
        using namespace SparseIndexer::Common::Primitive;
        const auto result = TopK::SelectHighPrefixIds(base, orderIndices_, history, topK_);
        if (result != TopK::SelectionStatus::Success) {
            return false;
        }
        retained = topK_;
        prefixIdsReady_ = true;
        return true;
    }

    // A row with at most K candidates only needs preparation and final ordering.
    __aicore__ inline void SelectHistoryTopKShortRow(uint32_t group, uint32_t token,
        uint32_t history, uint32_t &retained)
    {
        auto ids = sortTmp_.ReinterpretCast<uint32_t>();
        auto keys = merge_.ReinterpretCast<uint32_t>();
        auto rankedKeys = orderValues_.ReinterpretCast<uint32_t>();
        auto incomingKeys = retained_.ReinterpretCast<uint32_t>();
        auto incomingIds = incomingKeys[kMaxTopK];
        PrepareIncomingRadixCandidates(ScoreOffset(group, token), 0U, history,
            ids, keys, incomingKeys, incomingIds, retained);
        AppendRadixCandidates(rankedKeys, incomingKeys, incomingIds, 0U, retained);
        FinishRankedCandidates(retained, rankedKeys);
    }

    // Select one window directly; no window-candidate merge is needed.
    __aicore__ inline void SelectHistoryTopKSingleWindow(uint32_t group, uint32_t token,
        uint32_t history, uint32_t &retained)
    {
        auto ids = sortTmp_.ReinterpretCast<uint32_t>();
        auto keys = merge_.ReinterpretCast<uint32_t>();
        auto rankedKeys = orderValues_.ReinterpretCast<uint32_t>();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && \
    defined(ORIG_DTYPE_QUERY) && defined(DT_FLOAT8_E4M3FN) && (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN)
        if (topK_ == kNativeTopK) {
            PrepareIncomingRadixCandidates<false>(ScoreOffset(group, token), 0U, history,
                ids, keys, rankedKeys, orderIndices_, retained);
        } else
#endif
        {
            auto incomingKeys = retained_.ReinterpretCast<uint32_t>();
            auto incomingIds = incomingKeys[kMaxTopK];
            PrepareIncomingRadixCandidates(ScoreOffset(group, token), 0U, history,
                ids, keys, incomingKeys, incomingIds, retained);
            AppendRadixCandidates(rankedKeys, incomingKeys, incomingIds, 0U, retained);
        }
        FinishRankedCandidates(retained, rankedKeys);
    }

    __aicore__ inline void FinishRankedCandidates(uint32_t retained,
        LocalTensor<uint32_t> rankedKeys)
    {
        if (sortIds_) {
            rankedIdsReady_ = true;
        } else {
            SortFinalRadixScores(retained, rankedKeys);
        }
    }

    // The caller chooses this path only when more than one window is needed.
    __aicore__ inline void SelectHistoryTopKMultiWindow(uint32_t group, uint32_t token,
        uint32_t history, uint32_t &retained)
    {
        const uint64_t rowOffset = ScoreOffset(group, token);
        // 每块缩减至 K 个候选，复用工作区进行第二阶段合并。
        LocalTensor<uint32_t> ids = sortTmp_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> keys = merge_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> rankedKeys = orderValues_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> incomingKeys = retained_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> incomingIds = incomingKeys[kMaxTopK];
        LocalTensor<uint32_t> mergedIds = ids[kMaxTopK];
        retained = 0U;
        uint32_t firstPendingWindow = 0U;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && \
    defined(ORIG_DTYPE_QUERY) && defined(DT_FLOAT8_E4M3FN) && (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN)
        if (topK_ == kNativeTopK && history > topK_) {
            // Finish reading the first window before the next MTE2 overwrites it.
            PrepareIncomingRadixCandidates(rowOffset, 0U, history, ids, keys,
                rankedKeys, orderIndices_, retained);
            firstPendingWindow = kRadixTile;
        }
#endif
        for (uint32_t start = firstPendingWindow; start < history; start += kRadixTile) {
            uint32_t candidateCount;
            PrepareIncomingRadixCandidates(rowOffset, start, history, ids, keys,
                incomingKeys, incomingIds, candidateCount);
            const uint32_t incoming = candidateCount;
            const uint32_t combined = retained + incoming;
            if (combined <= topK_) {
                AppendRadixCandidates(rankedKeys, incomingKeys, incomingIds, retained, incoming);
                retained = combined;
                continue;
            }
            MergeRadixCandidates(ids, keys, rankedKeys, incomingKeys, incomingIds,
                mergedIds, retained, combined);
            retained = topK_;
        }
        FinishRankedCandidates(retained, rankedKeys);
    }

    __aicore__ inline void SelectHistoryTopKRadix(uint32_t group, uint32_t token,
                                                   uint32_t history, uint32_t &retained,
                                                   uint32_t &directMode)
    {
        const uint64_t rowOffset = ScoreOffset(group, token);
        LocalTensor<uint32_t> keys = sortTmp_.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> ids = merge_.ReinterpretCast<uint32_t>();
        retained = 0U;
        const bool singleTrunk = history <= kRadixTile;
        LocalTensor<uint32_t> selected = retained_.ReinterpretCast<uint32_t>();
        for (uint32_t start = 0U; start < history; start += kRadixTile) {
            const uint32_t count = Min(kRadixTile, history - start);
            // 尾块不足 K 时，全部元素参与全局候选竞争。
            if (count < topK_) {
                CollectShortRadixTile(rowOffset, start, count, retained);
                continue;
            }
            const uint32_t aligned = AlignUp(count, kRadixBlockElements);
            LoadRadixScoreTile(rowOffset + start, count);
            if (singleTrunk) {
                CheckUniformRadixTile(count, retained, directMode);
                if (directMode == 1U) {
                    continue;
                }
            }
            SparseTopK::PrepareFp32Keys(reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr()), count, aligned, kInvalidScore);
            PipeBarrier<PIPE_V>();
            FilterRadixCandidates(ids, keys, aligned);
            if (singleTrunk && sortIds_) {
                TryAscendingRadixIds(ids, keys, count, retained, directMode);
                if (directMode == 2U) {
                    continue;
                }
            }
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            CollectRadixSelection(ids, keys, selected, count, start, singleTrunk, retained);
            if (singleTrunk && !sortIds_ && retained != 0U) {
                SortSingleRadixSelection(keys, selected, retained);
            }
        }
        if (singleTrunk) {
            return;
        }
        DrainScoreHeap(retained);
    }

    __aicore__ inline void SortSelectedIds(uint32_t retained)
    {
        LocalTensor<uint32_t> retainedIds = retained_.ReinterpretCast<uint32_t>();
        const uint32_t aligned = Align32(topK_);
        // 负编号降序对应逻辑编号升序。
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        PrepareSelectedIdsSort(
            reinterpret_cast<__ubuf__ float *>(orderValues_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(orderIndices_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(retainedIds.GetPhyAddr()),
            retained, aligned);
        PipeBarrier<PIPE_V>();
        Sort<float, true>(merge_, orderValues_, orderIndices_, sortTmp_, aligned / 32U);
        PipeBarrier<PIPE_V>();
        ExtractAscendingIds(
            reinterpret_cast<__ubuf__ uint32_t *>(orderIndices_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(merge_.GetPhyAddr()), topK_);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    __aicore__ inline void SortRankedIds(uint32_t retained)
    {
        const uint32_t aligned = Align32(topK_);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && \
    defined(ORIG_DTYPE_QUERY) && defined(DT_FLOAT16) && defined(DT_BF16) && \
    ((ORIG_DTYPE_QUERY == DT_FLOAT16) || (ORIG_DTYPE_QUERY == DT_BF16))
        if (terminalIdOnly_) {
            PrepareJointIdsSort(retained, aligned);
        } else
#endif
        {
            PrepareRankedIdsSort(
                reinterpret_cast<__ubuf__ float *>(orderValues_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(orderIndices_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(orderValues_.GetPhyAddr()),
                retained, aligned);
        }
        PipeBarrier<PIPE_V>();
        Sort<float, true>(merge_, orderValues_, orderIndices_, sortTmp_, aligned / 32U);
        PipeBarrier<PIPE_V>();
        ExtractAscendingIds(
            reinterpret_cast<__ubuf__ uint32_t *>(orderIndices_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(merge_.GetPhyAddr()), topK_);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    __aicore__ inline void Store(uint32_t row, uint32_t retained, uint32_t directMode)
    {
        if (directMode != 0U || prefixIdsReady_) {
            if (directMode == 1U) {
                ArithProgression<int32_t>(orderIndices_.ReinterpretCast<int32_t>(),
                                          0, 1, topK_);
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyExtParams directCopy{1U, static_cast<uint32_t>(topK_ * sizeof(int32_t)),
                                         0U, 0U, 0U};
            DataCopyPad(selectedIds_[SelectedOffset(row)],
                        orderIndices_.ReinterpretCast<int32_t>(), directCopy);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        } else {
            LocalTensor<uint32_t> retainedIds = retained_.ReinterpretCast<uint32_t>();
            if (sortIds_) {
                if (rankedIdsReady_) {
                    SortRankedIds(retained);
                } else {
                    SortSelectedIds(retained);
                }
            } else if (!scoreIdsReady_) {
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                ExtractScoreOrderedIds(
                    reinterpret_cast<__ubuf__ uint32_t *>(orderIndices_.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t *>(retainedIds.GetPhyAddr()),
                    retained, topK_);
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
            }
            DataCopyExtParams copy{1U, static_cast<uint32_t>(topK_ * sizeof(int32_t)), 0U, 0U, 0U};
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            DataCopyPad(selectedIds_[SelectedOffset(row)],
                        orderIndices_.ReinterpretCast<int32_t>(), copy);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }

    template <Implementation implementation = kDefaultImplementation>
    __aicore__ inline void ProcessRow(uint32_t row)
    {
        rankedIdsReady_ = false;
        scoreIdsReady_ = false;
        prefixIdsReady_ = false;
        const uint32_t group = row / tokens_;
        const uint32_t token = row - group * tokens_ + batchBegin_;
        const uint32_t history = HistoryLength(token);
#if defined(ORIG_DTYPE_QUERY) && defined(DT_FLOAT16) && defined(DT_BF16) && \
    ((ORIG_DTYPE_QUERY == DT_FLOAT16) || (ORIG_DTYPE_QUERY == DT_BF16))
        if (!sortIds_ && topK_ == kNativeTopK && history > 0U && history <= topK_) {
            const uint64_t rowOffset = ScoreOffset(group, token);
            const auto result = UnderKVectorZero::Try(ub_.Get<uint8_t>(), scores_,
                rowOffset, orderIndices_, history);
            if (result == UnderKVectorZero::Result::Ready) {
                Store(row, 0U, 2U);
                return;
            }
            if (result == UnderKVectorZero::Result::ImplementationFailure) {
                Trap();
                return;
            }
            // Discard every fast view before original SelectHistory reloads GM.
            PipeBarrier<PIPE_ALL>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            rankedIdsReady_ = false;
            scoreIdsReady_ = false;
        }
#endif
        uint32_t retained = 0U;
        uint32_t directMode = 0U;
        if (history != 0U) {
            SelectHistoryTopK<implementation>(group, token, history, retained, directMode);
        }
        Store(row, retained, directMode);
    }

    __aicore__ inline uint64_t ScoreOffset(uint32_t group, uint32_t globalToken) const
    {
        return static_cast<uint64_t>(group) * scoreStrideGroup_ +
            static_cast<uint64_t>(globalToken - batchBegin_) * scoreStrideToken_;
    }

    __aicore__ inline uint64_t SelectedOffset(uint32_t batchRow) const
    {
        if (batchScratch_ || (batchBegin_ == 0U && tokens_ == totalTokens_)) {
            return static_cast<uint64_t>(batchRow) * selectedStride_;
        }
        const uint32_t group = batchRow / tokens_;
        const uint32_t token = batchRow - group * tokens_ + batchBegin_;
        return (static_cast<uint64_t>(group) * totalTokens_ + token) * selectedStride_;
    }

    uint32_t totalTokens_ = 0U;
    uint32_t batchBegin_ = 0U;
    bool batchScratch_ = false;
    uint32_t groups_ = 0U;
    uint32_t tokens_ = 0U;
    uint64_t scoreStrideGroup_ = 0U;
    uint64_t scoreStrideToken_ = 0U;
    uint32_t rout_ = 0U;
    uint32_t maxRegionLen_ = 0U;
    uint32_t topK_ = 0U;
    uint32_t selectedStride_ = 0U;
    uint32_t maskMode_ = 0U;
    uint32_t regionSize_ = 0U;
    bool sortIds_ = false;
    bool rankedIdsReady_ = false;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && \
    defined(ORIG_DTYPE_QUERY) && defined(DT_FLOAT16) && defined(DT_BF16) && \
    ((ORIG_DTYPE_QUERY == DT_FLOAT16) || (ORIG_DTYPE_QUERY == DT_BF16))
    bool terminalIdOnly_ = false;
#endif
    bool prefixIdsReady_ = false;
    bool scoreIdsReady_ = false;
    uint32_t numRequests_ = 0U;
    uint32_t usedCubeCores_ = 0U;
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> ub_;
    GlobalTensor<float> scores_;
    GlobalTensor<int32_t> selectedIds_;
    __gm__ int32_t *actualKvLengths_ = nullptr;
    __gm__ int32_t *cuSeqLens_ = nullptr;
    mutable uint32_t cachedRequest_ = 0U;
    mutable int64_t cachedFirstPosition_ = 0;
    LocalTensor<float> score_, sortTmp_, chunkSorted_, retained_, merge_, orderValues_;
    LocalTensor<uint32_t> index_, orderIndices_;
};

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && \
    defined(ORIG_DTYPE_QUERY) && defined(DT_FLOAT16) && defined(DT_BF16) && \
    ((ORIG_DTYPE_QUERY == DT_FLOAT16) || (ORIG_DTYPE_QUERY == DT_BF16))
// 版权所有华为技术有限公司 2026，遵循 CANN 开源软件许可协议 2.0。

// 原生低精度长行选择。
struct JointRadixPolicy {
    // 所有 Gather 索引先夹到安全位置；旧映射全部读完后才能回写。
    template <bool ID_ONLY>
    __simd_vf__ static void GatherJoint(__ubuf__ uint32_t *outKeys, __ubuf__ uint32_t *outIds,
        __ubuf__ uint32_t *positions, __ubuf__ uint32_t *keys, __ubuf__ uint32_t *oldIds,
        uint32_t retained, uint32_t count, uint32_t start, uint32_t topK)
    {
        Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint32_t> zero, invalid, oldLimit, countLimit, kLimit;
        Reg::Duplicate(zero, 0U); Reg::Duplicate(invalid, 0xffffffffU);
        Reg::Duplicate(oldLimit, retained); Reg::Duplicate(countLimit, count);
        Reg::Duplicate(kLimit, topK);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        for (uint32_t offset = 0U; offset < (topK + 63U) / 64U * 64U; offset += 64U) {
            Reg::RegTensor<uint32_t> pos, lane, safe, safeOld, delta, key, id, newId;
            Reg::MaskReg inK, inCount, isOld, hasKey, valid;
            Reg::LoadAlign<uint32_t>(pos, positions + offset);
            Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(lane), static_cast<int32_t>(offset));
            Reg::Compare<uint32_t, CMPMODE::LT>(inK, lane, kLimit, all);
            Reg::Compare<uint32_t, CMPMODE::LT>(inCount, pos, countLimit, all);
            Reg::And(valid, inK, inCount, all);
            Reg::Select(safe, pos, zero, valid);
            Reg::Compare<uint32_t, CMPMODE::LT>(isOld, safe, oldLimit, all);
            Reg::Select(safeOld, safe, zero, isOld);
            Reg::Gather(key, keys, safe, all);
            Reg::Gather(id, oldIds, safeOld, all);
            Reg::Sub(delta, safe, oldLimit, all);
            Reg::Adds(newId, delta, start, all);
            Reg::Select(id, id, newId, isOld);
            Reg::Compare<uint32_t, CMPMODE::GT>(hasKey, key, zero, all);
            Reg::And(valid, valid, hasKey, all);
            Reg::Select(id, id, invalid, valid);
            Reg::StoreAlign<uint32_t>(outIds + offset, id, all);
            if constexpr (!ID_ONLY) {
                Reg::Select(key, key, zero, valid);
                Reg::StoreAlign<uint32_t>(outKeys + offset, key, all);
            }
        }
    }

    // 已归一编号独立准备排序输入，末窗死亡键不再读取。
    __simd_vf__ static void PrepareIds(__ubuf__ float *values, __ubuf__ uint32_t *ids,
        uint32_t retained, uint32_t aligned)
    {
        Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint32_t> limit, invalid;
        Reg::RegTensor<float> negativeInfinity;
        Reg::Duplicate(limit, retained); Reg::Duplicate(invalid, 0xffffffffU);
        Reg::Duplicate(negativeInfinity, -__builtin_inff());
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        for (uint32_t offset = 0U; offset < aligned; offset += 64U) {
            Reg::RegTensor<uint32_t> lane, id;
            Reg::RegTensor<float> value;
            Reg::MaskReg inRange, hasId, valid;
            Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t> &>(lane), static_cast<int32_t>(offset));
            Reg::LoadAlign<uint32_t>(id, ids + offset);
            Reg::Compare<uint32_t, CMPMODE::LT>(inRange, lane, limit, all);
            Reg::Compare<uint32_t, CMPMODE::NE>(hasId, id, invalid, all);
            Reg::And(valid, inRange, hasId, all);
            Reg::Cast<float, int32_t, kIdIntToFloat>(value, reinterpret_cast<Reg::RegTensor<int32_t> &>(id), all);
            Reg::Muls(value, value, -1.0F, all);
            Reg::Select(value, value, negativeInfinity, valid);
            Reg::Select(id, id, invalid, valid);
            Reg::StoreAlign<float>(values + offset, value, all);
            Reg::StoreAlign<uint32_t>(ids + offset, id, all);
        }
    }

    // 整行能力已在入口判定，固定首窗大于 K，因此每窗必需联合筛选。
    __aicore__ static inline void Select(Selector &s, uint32_t group, uint32_t token,
        uint32_t history, uint32_t &retained)
    {
        const uint64_t row = s.ScoreOffset(group, token);
        auto positions = s.sortTmp_.ReinterpretCast<uint32_t>();
        auto raw = s.retained_.ReinterpretCast<uint32_t>();
        auto keys = s.merge_.ReinterpretCast<uint32_t>();
        auto ranked = s.orderValues_.ReinterpretCast<uint32_t>();
        auto temporaryIds = positions[4096U];
        retained = 0U;
        for (uint32_t start = 0U; start < history; start += kRadixTile) {
            const uint32_t n = Min(kRadixTile, history - start);
            const uint32_t combined = retained + n;
            const uint32_t aligned = AlignUp(combined, kRadixBlockElements);
            DataCopyExtParams copy{1U, static_cast<uint32_t>(n * sizeof(float)), 0U, 0U, 0U};
            DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            DataCopyPad(s.retained_, s.scores_[row + start], copy, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            SparseTopK::PrepareFp32Keys(reinterpret_cast<__ubuf__ uint32_t *>(raw.GetPhyAddr()), n, AlignUp(n, kRadixBlockElements), kInvalidScore);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::S_V>(EVENT_ID0); WaitFlag<HardEvent::S_V>(EVENT_ID0);
            BuildCombinedKeys(reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(ranked.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(raw.GetPhyAddr()), retained, combined, aligned);
            PipeBarrier<PIPE_V>();
            s.FilterRadixCandidates(positions, keys, aligned);
            PipeBarrier<PIPE_V>();
            const bool terminal = s.sortIds_ && start + n == history;
            GatherWindow(s, terminal, positions, keys, ranked, temporaryIds, retained, combined, start);
            PipeBarrier<PIPE_V>();
            CopyMergedIds(reinterpret_cast<__ubuf__ uint32_t *>(s.orderIndices_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t *>(temporaryIds.GetPhyAddr()), s.topK_);
            SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);
            retained = s.topK_;
            s.terminalIdOnly_ = terminal;
        }
        if (s.sortIds_) { s.rankedIdsReady_ = true; } else { s.SortFinalRadixScores(retained, ranked); }
    }

    __aicore__ static inline void GatherWindow(Selector &s, bool terminal,
        LocalTensor<uint32_t> positions, LocalTensor<uint32_t> keys, LocalTensor<uint32_t> ranked,
        LocalTensor<uint32_t> temporaryIds, uint32_t retained, uint32_t combined, uint32_t start)
    {
        auto outKeys = reinterpret_cast<__ubuf__ uint32_t *>(ranked.GetPhyAddr());
        auto outIds = reinterpret_cast<__ubuf__ uint32_t *>(temporaryIds.GetPhyAddr());
        auto pos = reinterpret_cast<__ubuf__ uint32_t *>(positions.GetPhyAddr());
        auto input = reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr());
        auto oldIds = reinterpret_cast<__ubuf__ uint32_t *>(s.orderIndices_.GetPhyAddr());
        if (terminal) { GatherJoint<true>(outKeys, outIds, pos, input, oldIds, retained, combined, start, s.topK_); }
        else { GatherJoint<false>(outKeys, outIds, pos, input, oldIds, retained, combined, start, s.topK_); }
    }
};

__aicore__ inline void Selector::SelectHistoryTopKJoint(uint32_t group, uint32_t token,
    uint32_t history, uint32_t &retained)
{
    JointRadixPolicy::Select(*this, group, token, history, retained);
}

__aicore__ inline void Selector::PrepareJointIdsSort(uint32_t retained, uint32_t aligned)
{
    JointRadixPolicy::PrepareIds(reinterpret_cast<__ubuf__ float *>(orderValues_.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t *>(orderIndices_.GetPhyAddr()), retained, aligned);
}

// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
// Shares Selector state and general selection primitives.
__simd_vf__ void CompactUnderKValidIds(__ubuf__ uint32_t *outputIds,
    __ubuf__ float *scores, uint32_t history)
{
    Reg::MaskReg all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<float> threshold;
    Reg::RegTensor<int32_t> limit;
    Reg::Duplicate(threshold, kInvalidScore);
    Reg::Duplicate(limit, static_cast<int32_t>(history));
    Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
    Reg::UnalignRegForStore alignIds;
    for (uint32_t offset = 0U; offset < history; offset += 64U) {
        Reg::RegTensor<float> values;
        Reg::RegTensor<int32_t> positions;
        Reg::RegTensor<uint32_t> packedIds;
        Reg::MaskReg inRange;
        Reg::MaskReg scoreValid;
        Reg::MaskReg keep;
        Reg::LoadAlign<float>(values, scores + offset);
        Reg::Arange(positions, static_cast<int32_t>(offset));
        Reg::Compare<int32_t, CMPMODE::LT>(inRange, positions, limit, all);
        Reg::Compare<float, CMPMODE::GT>(scoreValid, values, threshold, all);
        Reg::And(keep, inRange, scoreValid, all);
        Reg::Squeeze<uint32_t, Reg::GatherMaskMode::STORE_REG>(packedIds,
            reinterpret_cast<Reg::RegTensor<uint32_t> &>(positions), keep);
        Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(outputIds, packedIds, alignIds);
    }
    Reg::StoreUnAlignPost(outputIds, alignIds);
}

struct IdCompactBatchPolicy {
    static constexpr uint32_t kUnderKBatchCapacity = 4U;
    static constexpr uint32_t kUnderKSlotElements = kNativeTopK;
    struct UnderKBatchRow {
        uint64_t scoreOffset;
        uint32_t row;
        uint32_t history;
    };
    struct UnderKBatch {
        UnderKBatchRow rows[kUnderKBatchCapacity];
        uint32_t count;
        uint32_t group;
        uint32_t request;
    };
    static_assert(sizeof(UnderKBatch) <= 128U, "underK descriptors exceed frozen budget");

    __aicore__ static inline void Process(Selector &selector)
    {
        if (selector.topK_ == 0U || selector.topK_ > kMaxTopK || selector.groups_ == 0U ||
            selector.tokens_ == 0U || selector.usedCubeCores_ == 0U) {
            return;
        }
        // 每个 AIV 独占行条带，使用原始工作核编号。
        const uint32_t workerCount = selector.usedCubeCores_ * GetTaskRation();
        const uint32_t worker = GetBlockIdx();
        if (worker >= workerCount) {
            return;
        }
        const uint64_t rows = static_cast<uint64_t>(selector.groups_) * selector.tokens_;
        // 相邻短行共享搬运单元时，由同一 AIV 写入以避免竞争。
        uint32_t rowsPerUnit = 1U;
        while ((rowsPerUnit * selector.selectedStride_) % 8U != 0U) {
            ++rowsPerUnit;
        }
        const uint64_t rowUnits = (rows + rowsPerUnit - 1U) / rowsPerUnit;
        const uint64_t ownedUnits = worker < rowUnits
            ? 1U + (rowUnits - 1U - worker) / workerCount : 0U;
        const uint32_t savedRequest = selector.cachedRequest_;
        const int64_t savedFirstPosition = selector.cachedFirstPosition_;
        bool allLegacy = false;
        if (selector.batchBegin_ == 0U && selector.tokens_ == selector.totalTokens_ &&
            selector.maskMode_ == 3U && selector.numRequests_ != 0U &&
            static_cast<uint64_t>(selector.numRequests_) < ownedUnits) {
            if (selector.cuSeqLens_ == nullptr) {
                allLegacy = selector.numRequests_ == 1U && selector.HistoryLength(0U) > selector.topK_;
            } else {
                bool complete = true;
                bool hasRows = false;
                uint32_t previousEnd = 0U;
                for (uint32_t request = 0U; request < selector.numRequests_; ++request) {
                    const int32_t requestBegin = selector.cuSeqLens_[request];
                    const int32_t requestEnd = selector.cuSeqLens_[request + 1U];
                    if (requestBegin < 0 || requestEnd < requestBegin ||
                        static_cast<uint32_t>(requestBegin) != previousEnd ||
                        static_cast<uint32_t>(requestEnd) > selector.tokens_) {
                        complete = false;
                        break;
                    }
                    previousEnd = static_cast<uint32_t>(requestEnd);
                    if (requestBegin == requestEnd) {
                        continue;
                    }
                    hasRows = true;
                    // Every later history in this request is at least its first history.
                    if (selector.HistoryLength(static_cast<uint32_t>(requestBegin)) <= selector.topK_) {
                        complete = false;
                        break;
                    }
                }
                allLegacy = complete && hasRows && previousEnd == selector.tokens_;
            }
        }
        selector.cachedRequest_ = savedRequest;
        selector.cachedFirstPosition_ = savedFirstPosition;
        if (allLegacy) {
            for (uint64_t unit = worker; unit < rowUnits; unit += workerCount) {
                const uint64_t begin = unit * rowsPerUnit;
                const uint64_t end = Min(static_cast<uint32_t>(rows - begin), rowsPerUnit) + begin;
                for (uint64_t row = begin; row < end; ++row) {
                    ProcessRow(selector, static_cast<uint32_t>(row));
                }
            }
            return;
        }
        UnderKBatch batch{};
        for (uint64_t unit = worker; unit < rowUnits; unit += workerCount) {
            const uint64_t begin = unit * rowsPerUnit;
            const uint64_t end = Min(static_cast<uint32_t>(rows - begin), rowsPerUnit) + begin;
            for (uint64_t row = begin; row < end; ++row) {
                const uint32_t group = static_cast<uint32_t>(row) / selector.tokens_;
                const uint32_t token = static_cast<uint32_t>(row) - group * selector.tokens_ + selector.batchBegin_;
                const uint32_t request = selector.FindRequest(token);
                const uint32_t history = selector.HistoryLength(token);
                if (batch.count != 0U &&
                    (batch.group != group || batch.request != request || history > selector.topK_)) {
                    Flush(selector, batch);
                }
                if (history > selector.topK_) {
                    ProcessRow(selector, static_cast<uint32_t>(row));
                    continue;
                }
                batch.group = group;
                batch.request = request;
                batch.rows[batch.count++] = {
                    selector.ScoreOffset(group, token),
                    static_cast<uint32_t>(row), history};
                if (batch.count == kUnderKBatchCapacity) {
                    Flush(selector, batch);
                }
            }
        }
        Flush(selector, batch);

    }

    __aicore__ static inline void Flush(Selector &selector, UnderKBatch &batch)
    {
        if (batch.count == 0U) {
            return;
        }
        LocalTensor<uint8_t> base = selector.ub_.Get<uint8_t>();
        for (uint32_t slot = 0U; slot < batch.count; ++slot) {
            const uint32_t history = batch.rows[slot].history;
            if (history != 0U) {
                LocalTensor<float> scores = base[SelectionWorkspace::kSort + slot * kUnderKSlotElements * sizeof(float)].ReinterpretCast<float>();
                DataCopyExtParams copy{1U, static_cast<uint32_t>(history * sizeof(float)), 0U, 0U, 0U};
                DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
                DataCopyPad(scores, selector.scores_[batch.rows[slot].scoreOffset], copy, pad);
            }
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (uint32_t slot = 0U; slot < batch.count; ++slot) {
            LocalTensor<uint32_t> ids = base[SelectionWorkspace::kOrderIds + slot * kUnderKSlotElements * sizeof(uint32_t)].ReinterpretCast<uint32_t>();
            Duplicate(ids, 0xffffffffU, kUnderKSlotElements);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t slot = 0U; slot < batch.count; ++slot) {
            if (batch.rows[slot].history != 0U) {
                CompactUnderKValidIds(
                    reinterpret_cast<__ubuf__ uint32_t *>(base[SelectionWorkspace::kOrderIds + slot * kUnderKSlotElements * sizeof(uint32_t)].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(base[SelectionWorkspace::kSort + slot * kUnderKSlotElements * sizeof(float)].GetPhyAddr()),
                    batch.rows[slot].history);
            }
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams copy{1U, static_cast<uint32_t>(selector.topK_ * sizeof(int32_t)), 0U, 0U, 0U};
        for (uint32_t slot = 0U; slot < batch.count; ++slot) {
            LocalTensor<int32_t> ids = base[SelectionWorkspace::kOrderIds + slot * kUnderKSlotElements * sizeof(uint32_t)].ReinterpretCast<int32_t>();
            DataCopyPad(selector.selectedIds_[selector.SelectedOffset(batch.rows[slot].row)],
                        ids, copy);
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        batch.count = 0U;

    }

    __aicore__ static inline void ProcessRow(Selector &selector, uint32_t row)
    {
        selector.rankedIdsReady_ = false;
        selector.scoreIdsReady_ = false;
        selector.prefixIdsReady_ = false;
        const uint32_t group = row / selector.tokens_;
        const uint32_t token = row - group * selector.tokens_ + selector.batchBegin_;
        const uint32_t history = selector.HistoryLength(token);
        if (history <= kNativeTopK) {
            // Retain the same row ownership, buffers, DMA helpers and Store events.
            Duplicate(selector.orderIndices_, 0xffffffffU, selector.topK_);
            PipeBarrier<PIPE_V>();
            if (history != 0U) {
                const uint64_t offset = selector.ScoreOffset(group, token);
                selector.LoadRadixScoreTile(offset, history);
                CompactUnderKValidIds(
                    reinterpret_cast<__ubuf__ uint32_t *>(selector.orderIndices_.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(selector.sortTmp_.GetPhyAddr()), history);
            }
            selector.Store(row, 0U, 2U);
            return;
        }
        uint32_t retained = 0U;
        uint32_t directMode = 0U;
        if (history != 0U) {
            selector.SelectHistoryTopK(group, token, history, retained, directMode);
        }
        selector.Store(row, retained, directMode);

    }
};

__aicore__ __attribute__((noinline)) static void RunIdCompactBatch(GM_ADDR scores, GM_ADDR actualKvLengths,
                            GM_ADDR cuSeqLens, GM_ADDR selectedIds,
                            uint32_t groups, uint32_t tokens,
                            uint64_t scoreStrideGroup, uint64_t scoreStrideToken,
                            uint32_t rout, uint32_t maxRegionLen, uint32_t topK,
                            uint32_t maskMode, uint32_t regionSize, bool sortIds,
                            uint32_t numRequests, uint32_t usedCubeCores,
                            TPipe *pipe, bool alignedSelection = false,
                            uint32_t batchBegin = 0U, uint32_t batchCount = 0U)
{
    if ASCEND_IS_AIC {
        return;
    }
    Selector selector;
    selector.Init(scores, actualKvLengths, cuSeqLens, selectedIds,
                  groups, tokens, scoreStrideGroup, scoreStrideToken, rout,
                  maxRegionLen, topK, maskMode, regionSize, sortIds,
                  numRequests, usedCubeCores, pipe, alignedSelection, batchBegin, batchCount);
    IdCompactBatchPolicy::Process(selector);
}

// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
// Native resource-bounded score selector.
struct SingleWindowPolicy final {
    __aicore__ static inline void PrepareSingleWindowCandidates(Selector &selector, uint32_t group, uint32_t token,
                                                         uint32_t history, uint32_t &retained)
    {
        const uint32_t aligned = AlignUp(history, kRadixBlockElements);
        LocalTensor<uint8_t> base = selector.ub_.Get<uint8_t>();
        LocalTensor<uint32_t> keys = base[SelectionWorkspace::kScore].ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> positions = base[SelectionWorkspace::kRetained].ReinterpretCast<uint32_t>();
        const uint64_t rowOffset = selector.ScoreOffset(group, token);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyExtParams copy{1U, static_cast<uint32_t>(history * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
        DataCopyPad(keys.ReinterpretCast<float>(), selector.scores_[rowOffset], copy, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        SparseTopK::PrepareFp32Keys(reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr()), history, aligned, kInvalidScore);
        PipeBarrier<PIPE_V>();
        selector.FilterRadixCandidates(positions, keys, aligned);
        PipeBarrier<PIPE_V>();
        GatherChunkCandidates(
            reinterpret_cast<__ubuf__ uint32_t *>(selector.orderValues_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(selector.orderIndices_.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(positions.GetPhyAddr()),
            reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr()), history, selector.topK_, 0U, true);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        retained = selector.topK_;
    }

    __aicore__ static inline void SelectHistoryTopKSingleWindow(Selector &selector, uint32_t group, uint32_t token,
                                                        uint32_t history, uint32_t &retained)
    {
        PrepareSingleWindowCandidates(selector, group, token, history, retained);
        // Raw keys, positions and histogram are dead; original Pack/Sort may reuse them.
        selector.SortFinalRadixScores(retained, selector.orderValues_.ReinterpretCast<uint32_t>());
    }

    __aicore__ static inline void ProcessRow(Selector &selector, uint32_t row)
    {
        const uint32_t group = row / selector.tokens_;
        const uint32_t token = row - group * selector.tokens_ + selector.batchBegin_;
        const uint32_t history = selector.HistoryLength(token);
        if (!selector.sortIds_ && selector.topK_ == kNativeTopK && history > kRadixTile &&
            history <= SelectionWorkspace::kSingleWindowCapacity) {
            selector.rankedIdsReady_ = false;
            selector.scoreIdsReady_ = false;
            uint32_t retained = 0U;
            SelectHistoryTopKSingleWindow(selector, group, token, history, retained);
            selector.Store(row, retained, 0U);
            return;
        }
        selector.ProcessRow(row);
    }

    __aicore__ static inline void Process(Selector &selector)
    {
        // HistoryLength is bounded by maxRegionLen_: no window can match this arena.
        if (selector.maxRegionLen_ <= kRadixTile || selector.sortIds_ || selector.topK_ != kNativeTopK) {
            selector.Process();
            return;
        }
        SingleWindowPolicy policy;
        selector.ProcessRows(policy);
    }

};

__aicore__ __attribute__((noinline)) static void RunSingleWindow(GM_ADDR scores, GM_ADDR actualKvLengths,
                            GM_ADDR cuSeqLens, GM_ADDR selectedIds,
                            uint32_t groups, uint32_t tokens,
                            uint64_t scoreStrideGroup, uint64_t scoreStrideToken,
                            uint32_t rout, uint32_t maxRegionLen, uint32_t topK,
                            uint32_t maskMode, uint32_t regionSize, bool sortIds,
                            uint32_t numRequests, uint32_t usedCubeCores,
                            TPipe *pipe, bool alignedSelection = false,
                            uint32_t batchBegin = 0U, uint32_t batchCount = 0U)
{
    if ASCEND_IS_AIC {
        return;
    }
    Selector selector;
    selector.Init(scores, actualKvLengths, cuSeqLens, selectedIds,
                  groups, tokens, scoreStrideGroup, scoreStrideToken, rout,
                  maxRegionLen, topK, maskMode, regionSize, sortIds,
                  numRequests, usedCubeCores, pipe, alignedSelection, batchBegin, batchCount);
    SingleWindowPolicy::Process(selector);
}

#endif

// 评分与寻址资源释放并全核同步后，运行独立 AIV 选择流水。
template <Implementation implementation = kDefaultImplementation>
__aicore__ inline void Run(GM_ADDR scores, GM_ADDR actualKvLengths,
                            GM_ADDR cuSeqLens, GM_ADDR selectedIds,
                            uint32_t groups, uint32_t tokens,
                            uint64_t scoreStrideGroup, uint64_t scoreStrideToken,
                            uint32_t rout, uint32_t maxRegionLen, uint32_t topK,
                            uint32_t maskMode, uint32_t regionSize, bool sortIds,
                            uint32_t numRequests, uint32_t usedCubeCores,
                            TPipe *pipe, bool alignedSelection = false,
                            uint32_t batchBegin = 0U, uint32_t batchCount = 0U)
{
    if ASCEND_IS_AIC {
        return;
    }
    Selector selector;
    selector.Init(scores, actualKvLengths, cuSeqLens, selectedIds,
                  groups, tokens, scoreStrideGroup, scoreStrideToken, rout,
                  maxRegionLen, topK, maskMode, regionSize, sortIds,
                  numRequests, usedCubeCores, pipe, alignedSelection, batchBegin, batchCount);
    selector.Process<implementation>();
}

// 二次全核同步后压紧私有对齐行，共享搬运单元按原分组处理。
__aicore__ inline void CompactActiveRows(GM_ADDR scratch, GM_ADDR selectedIds,
                                         uint32_t groups, uint32_t tokens, uint32_t topK,
                                         uint32_t worker, uint32_t workerCount, TPipe *pipe,
                                         uint32_t batchBegin = 0U, uint32_t batchCount = 0U)
{
    const uint32_t paddedK = (topK + 7U) / 8U * 8U;
    uint32_t rowsPerUnit = 1U;
    while ((rowsPerUnit * topK) % 8U != 0U) {
        ++rowsPerUnit;
    }
    const uint32_t activeTokens = batchCount != 0U ? batchCount : tokens;
    // Assign ownership using GLOBAL output DMA units. A batch or group boundary
    // may cut an unaligned K row unit; local-row striping would let two AIVs
    // update the same 32-byte output unit when T0 is not a multiple of 8.

    GlobalTensor<int32_t> source, destination;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(scratch));
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(selectedIds));
    TBuf<TPosition::VECCALC> ub;
    pipe->InitBuffer(ub, paddedK * sizeof(int32_t));
    LocalTensor<int32_t> values = ub.Get<int32_t>();
    DataCopyExtParams copy{1U, static_cast<uint32_t>(topK * sizeof(int32_t)),
                           0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> pad{false, 0U, 0U, 0};
    if (batchCount == 0U) {
        // Preserve the full-Score compact traversal and DMA ownership exactly.
        const uint64_t rows = static_cast<uint64_t>(groups) * tokens;
        const uint64_t units = (rows + rowsPerUnit - 1U) / rowsPerUnit;
        for (uint64_t unit = worker; unit < units; unit += workerCount) {
            const uint64_t begin = unit * rowsPerUnit;
            const uint64_t end = begin + Min(static_cast<uint32_t>(rows - begin), rowsPerUnit);
            for (uint64_t row = begin; row < end; ++row) {
                DataCopyPad(values, source[row * paddedK], copy, pad);
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                DataCopyPad(destination[row * topK], values, copy);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
        return;
    }
    for (uint32_t group = 0U; group < groups; ++group) {
        const uint64_t firstRow = static_cast<uint64_t>(group) * tokens + batchBegin;
        const uint64_t lastRow = firstRow + activeTokens;
        const uint64_t firstUnit = firstRow / rowsPerUnit;
        const uint64_t endUnit = (lastRow + rowsPerUnit - 1U) / rowsPerUnit;
        const uint64_t ownedFirstUnit = firstUnit +
            (worker + workerCount - firstUnit % workerCount) % workerCount;
        for (uint64_t unit = ownedFirstUnit; unit < endUnit; unit += workerCount) {
            const uint64_t unitBegin = unit * rowsPerUnit;
            const uint64_t begin = unitBegin < firstRow ? firstRow : unitBegin;
            const uint64_t unitEnd = unitBegin + rowsPerUnit;
            const uint64_t end = unitEnd < lastRow ? unitEnd : lastRow;
            for (uint64_t row = begin; row < end; ++row) {
                const uint64_t batchRow = static_cast<uint64_t>(group) * activeTokens + row - firstRow;
                DataCopyPad(values, source[batchRow * paddedK], copy, pad);
                SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
                DataCopyPad(destination[row * topK], values, copy);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
    }
}

__aicore__ inline void Compact(GM_ADDR scratch, GM_ADDR selectedIds,
                               uint32_t groups, uint32_t tokens, uint32_t topK,
                               uint32_t usedCubeCores, TPipe *pipe,
                               uint32_t batchBegin = 0U, uint32_t batchCount = 0U)
{
    // 仅活动 AIV 复制，每轮双向同步后复用缓冲区。
    if ASCEND_IS_AIC {
        return;
    }
    if (topK == 0U || usedCubeCores == 0U) {
        return;
    }
    const uint32_t workerCount = usedCubeCores * GetTaskRation();
    const uint32_t worker = GetBlockIdx();
    if (worker >= workerCount) {
        return;
    }
    CompactActiveRows(scratch, selectedIds, groups, tokens, topK,
                      worker, workerCount, pipe, batchBegin, batchCount);
}

} // 命名空间结束。
} // 命名空间结束。

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) && \
    defined(ORIG_DTYPE_QUERY) && defined(DT_FLOAT16) && defined(DT_BF16) && \
    ((ORIG_DTYPE_QUERY == DT_FLOAT16) || (ORIG_DTYPE_QUERY == DT_BF16))
namespace IndexerRegionScore {
namespace FusedSelect {
// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-Agreement-2.0
// Native-only short-row backend.
#ifndef IRS_D2_PREPARED_KEY_RANK_H
#define IRS_D2_PREPARED_KEY_RANK_H
// Prepared-key ranking for the short-row selector.
// Preconditions: h<=128, keys point to ceil(h/64)*64 initialized uint32s,
// out points to 512 uint32s, aligned UB, nonoverlapping, original contiguous IDs.
// The caller prepares the keys using the shared FP32 conversion.
__simd_vf__ void ShortRowPreparedKeyRank(__ubuf__ uint32_t* out,
                                       __ubuf__ uint32_t* keys, uint32_t h)
{
    auto all = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> zero, one, invalid;
    Reg::Duplicate(zero, 0U);
    Reg::Duplicate(one, 1U);
    Reg::Duplicate(invalid, 0xffffffffU);
    for (uint32_t o = 0; o < kNativeTopK; o += SparseTopK::kKeyLanes) {
        Reg::StoreAlign(out + o, invalid, all);
    }
    // Finish caller's prepared-key stores before the first vector/broadcast load.
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    // A late fill must not overwrite Scatter. Scatter indices are ELEMENTS.
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();
    for (uint32_t o = 0; o < h; o += 64) {
        Reg::RegTensor<uint32_t> keyI, ids, rank;
        Reg::MaskReg inRange, positive, valid;
        Reg::LoadAlign(keyI, keys + o);
        // SDK Arange supports signed int32; IDs 0..127 have identical bits.
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(ids), static_cast<int32_t>(o));
        Reg::Duplicate(rank, 0U);
        Reg::Compares<uint32_t, CMPMODE::LT>(inRange, ids, h, all);
        Reg::Compare<uint32_t, CMPMODE::GT>(positive, keyI, zero, all);
        Reg::And(valid, inRange, positive, all);
        // All candidates before this register win ties. All candidates
        // after it lose ties, so those chunks need only one comparison.
        for (uint32_t j = 0; j < o; ++j) {
            Reg::RegTensor<uint32_t> keyJ, inc;
            Reg::MaskReg wins;
            Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(keyJ, keys + j);
            Reg::Compare<uint32_t, CMPMODE::GE>(wins, keyJ, keyI, all);
            Reg::Select(inc, one, zero, wins);
            Reg::Add(rank, rank, inc, all);
        }
        const uint32_t groupEnd = h < o + 64U ? h : o + 64U;
        for (uint32_t j = o; j < groupEnd; ++j) {
            Reg::RegTensor<uint32_t> keyJ, inc;
            Reg::MaskReg greater, equal, earlier, tie, wins;
            Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(keyJ, keys + j);
            Reg::Compare<uint32_t, CMPMODE::GT>(greater, keyJ, keyI, all);
            Reg::Compare<uint32_t, CMPMODE::EQ>(equal, keyJ, keyI, all);
            Reg::Compares<uint32_t, CMPMODE::GT>(earlier, ids, j, all);
            Reg::And(tie, equal, earlier, all);
            Reg::Or(wins, greater, tie, all);
            Reg::Select(inc, one, zero, wins);
            Reg::Add(rank, rank, inc, all);
        }
        for (uint32_t j = groupEnd; j < h; ++j) {
            Reg::RegTensor<uint32_t> keyJ, inc;
            Reg::MaskReg wins;
            Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(keyJ, keys + j);
            Reg::Compare<uint32_t, CMPMODE::GT>(wins, keyJ, keyI, all);
            Reg::Select(inc, one, zero, wins);
            Reg::Add(rank, rank, inc, all);
        }
        Reg::Scatter<uint32_t, uint32_t>(out, ids, rank, valid);
    }
}

#endif

#ifndef IRS_D2_PREPARED_KEY_NETWORK_UNROLLED_H
#define IRS_D2_PREPARED_KEY_NETWORK_UNROLLED_H
// No runtime K/J/group loops. All stages and group identity are constants.
template<uint32_t K,uint32_t J,uint32_t BASE>
__simd_callee__ inline void NetworkStage(Reg::RegTensor<uint32_t>& keys,
                                        Reg::RegTensor<uint32_t>& ids,
                                        Reg::RegTensor<uint32_t>& lane,
                                        Reg::MaskReg& all)
{
    Reg::RegTensor<uint32_t> index,tmp,constant,partnerK,partnerI,zero;
    Reg::Duplicate(zero,0U);
    Reg::Duplicate(constant,J);Reg::Xor(index,lane,constant,all);
    Reg::Gather(partnerK,keys,index);Reg::Gather(partnerI,ids,index);
    Reg::MaskReg gt,eq,lt,tie,better,kZero,jNonZero,wantBest,take;
    Reg::Compare<uint32_t,CMPMODE::GT>(gt,keys,partnerK,all);
    Reg::Compare<uint32_t,CMPMODE::EQ>(eq,keys,partnerK,all);
    Reg::Compare<uint32_t,CMPMODE::LT>(lt,ids,partnerI,all);
    Reg::And(tie,eq,lt,all);Reg::Or(better,gt,tie,all);
    if constexpr(K<64) {
        Reg::Duplicate(constant,K);Reg::And(tmp,lane,constant,all);
        Reg::Compare<uint32_t,CMPMODE::EQ>(kZero,tmp,zero,all);
    } else {
        // Global position's K bit is constant within each 64-lane group.
        if constexpr((BASE&K)==0) Reg::Compare<uint32_t,CMPMODE::EQ>(kZero,zero,zero,all);
        else Reg::Compare<uint32_t,CMPMODE::GT>(kZero,zero,zero,all);
    }
    Reg::Duplicate(constant,J);Reg::And(tmp,lane,constant,all);
    Reg::Compare<uint32_t,CMPMODE::GT>(jNonZero,tmp,zero,all);
    Reg::Xor(wantBest,kZero,jNonZero,all);Reg::Xor(take,better,wantBest,all);
    Reg::Select(keys,partnerK,keys,take);Reg::Select(ids,partnerI,ids,take);
}

template<uint32_t C,uint32_t K,uint32_t J>
__simd_callee__ inline void NetworkStages(Reg::RegTensor<uint32_t>& k0,
                                         Reg::RegTensor<uint32_t>& i0,
                                         Reg::RegTensor<uint32_t>& k1,
                                         Reg::RegTensor<uint32_t>& i1,
                                         Reg::RegTensor<uint32_t>& lane,
                                         Reg::MaskReg& all)
{
    if constexpr(J==64) {
        Reg::MaskReg gt,eq,lt,tie,better;
        Reg::Compare<uint32_t,CMPMODE::GT>(gt,k0,k1,all);
        Reg::Compare<uint32_t,CMPMODE::EQ>(eq,k0,k1,all);
        Reg::Compare<uint32_t,CMPMODE::LT>(lt,i0,i1,all);
        Reg::And(tie,eq,lt,all);Reg::Or(better,gt,tie,all);
        Reg::RegTensor<uint32_t> lowK,lowI,highK,highI;
        Reg::Select(lowK,k0,k1,better);Reg::Select(lowI,i0,i1,better);
        Reg::Select(highK,k1,k0,better);Reg::Select(highI,i1,i0,better);
        Reg::Move(k0,lowK);Reg::Move(i0,lowI);
        Reg::Move(k1,highK);Reg::Move(i1,highI);
    } else {
        NetworkStage<K,J,0>(k0,i0,lane,all);
        if constexpr(C==128) NetworkStage<K,J,64>(k1,i1,lane,all);
    }
    if constexpr(J>1) NetworkStages<C,K,J/2>(k0,i0,k1,i1,lane,all);
    else if constexpr(K<C) NetworkStages<C,K*2,K>(k0,i0,k1,i1,lane,all);
}

template<uint32_t C>
__simd_vf__ void ShortRowPreparedKeyNetworkFixed(__ubuf__ uint32_t* out,
                                               __ubuf__ uint32_t* keys,uint32_t h)
{
    auto all=Reg::CreateMask<uint32_t,Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint32_t> invalid,zero;
    Reg::Duplicate(invalid,0xffffffffU);Reg::Duplicate(zero,0U);
    for(uint32_t o=0;o < kNativeTopK; o += SparseTopK::kKeyLanes) Reg::StoreAlign(out+o,invalid,all);
    if(h==0) return;
    Reg::LocalMemBar<Reg::MemType::VEC_STORE,Reg::MemType::VEC_LOAD>();
    Reg::LocalMemBar<Reg::MemType::VEC_STORE,Reg::MemType::VEC_STORE>();
    Reg::RegTensor<uint32_t> k0,i0,k1,i1,lane;
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lane),0);
    Reg::Move(i0,lane);
    Reg::LoadAlign(k0,keys);
    Reg::MaskReg valid;
    Reg::Compares<uint32_t,CMPMODE::LT>(valid,i0,h,all);
    Reg::Select(k0,k0,zero,valid);
    if constexpr(C==128) {
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(i1),64);
        Reg::LoadAlign(k1,keys+64);
        Reg::Compares<uint32_t,CMPMODE::LT>(valid,i1,h,all);
        Reg::Select(k1,k1,zero,valid);
    }
    NetworkStages<C,2,1>(k0,i0,k1,i1,lane,all);
    Reg::Compare<uint32_t,CMPMODE::GT>(valid,k0,zero,all);
    if constexpr(C==32) {
        Reg::MaskReg low;
        Reg::Compares<uint32_t,CMPMODE::LT>(low,lane,32U,all);
        Reg::And(valid,valid,low,all);
    }
    Reg::Select(i0,i0,invalid,valid);Reg::StoreAlign(out,i0,all);
    if constexpr(C==128) {
        Reg::Compare<uint32_t,CMPMODE::GT>(valid,k1,zero,all);
        Reg::Select(i1,i1,invalid,valid);Reg::StoreAlign(out+64,i1,all);
    }
}

#endif

// Measured conservative switch point; not a claim of the exact crossover in81..95.
constexpr uint32_t kShortRowNetworkBegin = 96U;
struct ShortRowPolicy final {
    // Only scalar descriptors persist between phases (8 * (4 + 4 + 8) bytes).
    __aicore__ static inline void Flush(Selector &selector, const uint32_t *rows,
                                      const uint32_t *histories, const uint64_t *scoreOffsets,
                                      uint32_t &count)
    {
        if (count == 0U) {
            return;
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        const uint32_t rowStep = selector.usedCubeCores_ * GetTaskRation();
        const bool stridedBatch = count >= 4U && selector.groups_ == 1U &&
            selector.scoreStrideToken_ == kNativeShortRowCapacity;
        DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
        if (stridedBatch) {
            constexpr uint32_t rowBytes = kNativeShortRowCapacity * sizeof(float);
            DataCopyExtParams copy{static_cast<uint16_t>(count), rowBytes, (rowStep - 1U) * rowBytes, 0U, 0U};
            DataCopyPad(selector.score_, selector.scores_[scoreOffsets[0]], copy, pad);
        } else {
            for (uint32_t slot = 0U; slot < count; ++slot) {
                if (histories[slot] == 0U) {
                    continue;
                }
                LocalTensor<uint32_t> keys = selector.score_.ReinterpretCast<uint32_t>()[slot * kNativeShortRowCapacity];
                keys.SetSize(kNativeShortRowCapacity);
                DataCopyExtParams copy{1U, static_cast<uint32_t>(histories[slot] * sizeof(float)),
                                       0U, 0U, 0U};
                DataCopyPad(keys.ReinterpretCast<float>(), selector.scores_[scoreOffsets[slot]], copy, pad);
            }
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (uint32_t slot = 0U; slot < count; ++slot) {
            if (histories[slot] == 0U) {
                continue;
            }
            LocalTensor<uint32_t> keys = selector.score_.ReinterpretCast<uint32_t>()[slot * kNativeShortRowCapacity];
            keys.SetSize(kNativeShortRowCapacity);
            const uint32_t history = histories[slot];
            const uint32_t aligned = (history + 63U) / 64U * 64U;
            SparseTopK::PrepareFp32Keys(reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr()), history, aligned, kInvalidScore);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t slot = 0U; slot < count; ++slot) {
            LocalTensor<uint32_t> keys = selector.score_.ReinterpretCast<uint32_t>()[slot * kNativeShortRowCapacity];
            keys.SetSize(kNativeShortRowCapacity);
            LocalTensor<uint32_t> outputIds = selector.index_[slot * kNativeTopK];
            outputIds.SetSize(kNativeTopK);
            const uint32_t history = histories[slot];
            if (history < kShortRowNetworkBegin) {
                ShortRowPreparedKeyRank(
                    reinterpret_cast<__ubuf__ uint32_t *>(outputIds.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr()), history);
            } else {
                ShortRowPreparedKeyNetworkFixed<kNativeShortRowCapacity>(
                    reinterpret_cast<__ubuf__ uint32_t *>(outputIds.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t *>(keys.GetPhyAddr()), history);
            }
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        constexpr uint32_t idsRowBytes = kNativeTopK * sizeof(int32_t);
        if (stridedBatch) {
            DataCopyExtParams copy{static_cast<uint16_t>(count), idsRowBytes, 0U, (rowStep - 1U) * idsRowBytes, 0U};
            DataCopyPad(selector.selectedIds_[selector.SelectedOffset(rows[0])],
                        selector.index_.ReinterpretCast<int32_t>(), copy);
        } else {
            for (uint32_t slot = 0U; slot < count; ++slot) {
                LocalTensor<uint32_t> outputIds = selector.index_[slot * kNativeTopK];
                outputIds.SetSize(kNativeTopK);
                DataCopyExtParams copy{1U, idsRowBytes, 0U, 0U, 0U};
                DataCopyPad(selector.selectedIds_[selector.SelectedOffset(rows[slot])],
                            outputIds.ReinterpretCast<int32_t>(), copy);
            }
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        count = 0U;
    }

    struct RowBatch {
        uint32_t rows[kNativeShortRowBatch];
        uint32_t histories[kNativeShortRowBatch];
        uint64_t scoreOffsets[kNativeShortRowBatch];
        uint32_t count = 0U;

        __aicore__ inline void ProcessRow(Selector &selector, uint32_t row)
        {
            const uint32_t group = row / selector.tokens_;
            const uint32_t token = row - group * selector.tokens_ + selector.batchBegin_;
            // The unchanged outer dispatch guarantees history <=128.
            rows[count] = row;
            histories[count] = selector.HistoryLength(token);
            scoreOffsets[count] = selector.ScoreOffset(group, token);
            ++count;
            if (count == kNativeShortRowBatch) {
                Flush(selector, rows, histories, scoreOffsets, count);
            }
        }
    };

    __aicore__ static inline void Process(Selector &selector)
    {
        RowBatch batch;
        selector.ProcessRows(batch);
        Flush(selector, batch.rows, batch.histories, batch.scoreOffsets, batch.count);
    }

};

__aicore__ __attribute__((noinline)) static void RunShortRows(GM_ADDR scores, GM_ADDR actualKvLengths,
                            GM_ADDR cuSeqLens, GM_ADDR selectedIds,
                            uint32_t groups, uint32_t tokens,
                            uint64_t scoreStrideGroup, uint64_t scoreStrideToken,
                            uint32_t rout, uint32_t maxRegionLen, uint32_t topK,
                            uint32_t maskMode, uint32_t regionSize, bool sortIds,
                            uint32_t numRequests, uint32_t usedCubeCores,
                            TPipe *pipe, bool alignedSelection = false,
                            uint32_t batchBegin = 0U, uint32_t batchCount = 0U)
{
    if ASCEND_IS_AIC {
        return;
    }
    Selector selector;
    selector.Init(scores, actualKvLengths, cuSeqLens, selectedIds,
                  groups, tokens, scoreStrideGroup, scoreStrideToken, rout,
                  maxRegionLen, topK, maskMode, regionSize, sortIds,
                  numRequests, usedCubeCores, pipe, alignedSelection, batchBegin, batchCount);
    ShortRowPolicy::Process(selector);
}
}
}
#endif

#endif // 头文件保护结束。
