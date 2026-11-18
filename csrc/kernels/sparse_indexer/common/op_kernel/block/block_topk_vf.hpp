/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * Vector radix-select TopK ported from ops-transformer QLIv2. The kernel emits
 * QLIv2's threshold/filter order, not a score-descending order.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_BLOCK_TOPK_VF_HPP
#define FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_BLOCK_TOPK_VF_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "kernel_operator.h"

#include "mask_utils.hpp"
#include "sparse_indexer_common.hpp"
#include "topk_vf/vf_topk_16_gather_quant_v2.h"

namespace SparseIndexer::Common::Block {

template <class DtypeScore_, class DtypeOut_>
class BlockTopKVector {
public:
    struct Params {
        uint32_t topK{0};
        uint32_t s2Stride{0};
        uint32_t s2Tile{0};
        bool hasOutputIdxOffset{false};
    };

    using DtypeScore = DtypeScore_;
    using DtypeOut = DtypeOut_;
    using ArchTag = Catlass::Arch::Ascend950;

    static constexpr uint32_t kSplitS1 = 2;
    static constexpr uint32_t kFloatBlock = Catlass::BYTE_PER_BLK / sizeof(DtypeScore);

    CATLASS_DEVICE BlockTopKVector(Catlass::Arch::Resource<ArchTag>& resource,
        AscendC::GlobalTensor<DtypeScore> const& workspace, AscendC::GlobalTensor<DtypeOut> const& outIndices,
        Params const& params, AscendC::GlobalTensor<int32_t> const& outputIdxOffset,
        AscendC::GlobalTensor<DtypeScore> const& outValues)
        : gmScores_(workspace),
          gmOut_(outIndices),
          gmValues_(outValues),
          gmOutputIdxOffset_(outputIdxOffset),
          topK_(params.topK),
          s2Stride_(params.s2Stride),
          s2Tile_(params.s2Tile),
          hasOutputIdxOffset_(params.hasOutputIdxOffset),
          tileColAlign_(SparseIndexer::Common::AlignUp(params.s2Tile, 256U)),
          topKAlign_(SparseIndexer::Common::AlignUp(params.topK, 256U))
    {
        const uint32_t outWords = topKAlign_ + 64U;
        const uint32_t tmpWords = SparseIndexer::Common::AlignUp(sharedBufferBytes(params.topK, tileColAlign_), 4U) / 4U;
        uint32_t byteOffset = 0;
        scoreUb_ = resource.ubBuf.template GetBufferByByte<DtypeScore>(byteOffset);
        byteOffset += SparseIndexer::Common::AlignUp(tileColAlign_ * sizeof(DtypeScore), 256U);
        inputUb_ = resource.ubBuf.template GetBufferByByte<uint16_t>(byteOffset);
        byteOffset += SparseIndexer::Common::AlignUp(tileColAlign_ * sizeof(uint16_t), 256U);
        outIdxUb_ = resource.ubBuf.template GetBufferByByte<uint32_t>(byteOffset);
        byteOffset += outWords * sizeof(uint32_t);
        outValueUb_ = resource.ubBuf.template GetBufferByByte<uint16_t>(byteOffset);
        byteOffset += SparseIndexer::Common::AlignUp((topKAlign_ + 64U) * sizeof(uint16_t), 256U);
        topkTmpUb_ = resource.ubBuf.template GetBufferByByte<uint32_t>(byteOffset);
        (void)byteOffset;

        hisIndex1_ = outIdxUb_;
        hisIndex2_ = topkTmpUb_[0];
        hist_ = hisIndex2_[topKAlign_];
        idxHigh_ = hist_[256];
        idxLow_ = idxHigh_[256];
        nk_ = idxLow_[256];
        tmpIndex_ = nk_[64].template ReinterpretCast<uint16_t>();
    }

    CATLASS_DEVICE void operator()(uint32_t s1Start, uint32_t actualS1, uint32_t kStart, uint32_t kLen)
    {
        if (actualS1 == 0) {
            return;
        }
        const uint32_t rowsPerSub = SparseIndexer::Common::CeilDiv(actualS1, kSplitS1);
        const uint32_t sub = AscendC::GetSubBlockIdx();
        const uint32_t rowBegin = sub * rowsPerSub;
        if (rowBegin >= actualS1) {
            return;
        }
        const uint32_t rowCount = AscendC::Std::min(rowsPerSub, actualS1 - rowBegin);
        const uint32_t alignedLen = SparseIndexer::Common::AlignUp(kLen, 256U);

        AscendC::DataCopyExtParams copyIn{1, 0, 0, 0, 0};
        AscendC::DataCopyExtParams copyOut{1, static_cast<uint32_t>(topK_ * sizeof(DtypeOut)), 0, 0, 0};
        AscendC::LocalTensor<DtypeOut> outIdx = outIdxUb_.template ReinterpretCast<DtypeOut>();

        for (uint32_t ri = 0; ri < rowCount; ++ri) {
            const uint32_t s1Row = s1Start + rowBegin + ri;
            if (kLen == 0) {
                for (uint32_t i = 0; i < topK_; ++i) {
                    outIdx.SetValue(i, static_cast<DtypeOut>(-1));
                }
                storeRow(outIdx, s1Row, copyOut);
                continue;
            }

            copyIn.blockLen = kLen * sizeof(DtypeScore);
            AscendC::DataCopyPadExtParams<DtypeScore> padParams{false, 0, 0, static_cast<DtypeScore>(0)};
            AscendC::DataCopyPad(scoreUb_, gmScores_[s1Row * s2Stride_ + kStart], copyIn, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::PipeBarrier<PIPE_V>();

            uint32_t finiteLen = 0;
            for (uint32_t i = 0; i < kLen; ++i) {
                const float value = scoreUb_.GetValue(i);
                if (!IsNegInfFloat(value)) {
                    ++finiteLen;
                    inputUb_.SetValue(i, FloatToSortableU16(value));
                } else {
                    inputUb_.SetValue(i, 0U);
                }
            }
            for (uint32_t i = kLen; i < alignedLen; ++i) {
                inputUb_.SetValue(i, 0U);
            }
            AscendC::PipeBarrier<PIPE_V>();

            if (finiteLen < topK_) {
                for (uint32_t i = 0; i < topK_; ++i) {
                    outIdx.SetValue(i, i < finiteLen ? static_cast<DtypeOut>(kStart + i) : static_cast<DtypeOut>(-1));
                }
            } else {
                topkb16gather::LiTopKVF<false>(tmpIndex_, outValueUb_, inputUb_, hist_, idxHigh_, idxLow_, nk_, topK_,
                    alignedLen);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t i = 0; i < topK_; ++i) {
                    outIdx.SetValue(i, static_cast<DtypeOut>(tmpIndex_.GetValue(i)));
                }
                liV2TopkCommon::IndicesAddOffset(outIdxUb_, kStart, topK_);
            }

            int32_t rowOff = hasOutputIdxOffset_ ? gmOutputIdxOffset_.GetValue(s1Row) : 0;
            if (rowOff != 0) {
                for (uint32_t i = 0; i < topK_; ++i) {
                    if (outIdx.GetValue(i) >= 0) {
                        outIdx.SetValue(i, outIdx.GetValue(i) + rowOff);
                    }
                }
            }
            storeRow(outIdx, s1Row, copyOut);
        }
    }

private:
    CATLASS_DEVICE static uint32_t sharedBufferBytes(uint32_t topK, uint32_t trunkLen)
    {
        const uint32_t alignedTopK = SparseIndexer::Common::AlignUp(topK, 256U);
        const uint32_t bytes1 = (2U * alignedTopK + 3U * 256U + 64U) * sizeof(uint32_t);
        const uint32_t bytes2 = (alignedTopK + trunkLen) * sizeof(uint16_t);
        const uint32_t reuse = alignedTopK * sizeof(uint32_t);
        return bytes1 + bytes2 - reuse;
    }

    CATLASS_DEVICE static uint16_t FloatToSortableU16(float value)
    {
        union {
            float f;
            uint32_t u;
        } bits;
        bits.f = value;
        const uint32_t lsb = (bits.u >> 16U) & 1U;
        uint16_t bf = static_cast<uint16_t>((bits.u + 0x7fffu + lsb) >> 16U);
        return (bf & 0x8000U) ? static_cast<uint16_t>(~bf) : static_cast<uint16_t>(bf ^ 0x8000U);
    }

    CATLASS_DEVICE void storeRow(AscendC::LocalTensor<DtypeOut> const& outIdx,
        uint32_t s1Row, AscendC::DataCopyExtParams const& copyOut)
    {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(gmOut_[s1Row * topK_], outIdx, copyOut);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    AscendC::GlobalTensor<DtypeScore> const& gmScores_;
    AscendC::GlobalTensor<DtypeOut> const& gmOut_;
    AscendC::GlobalTensor<DtypeScore> gmValues_;
    AscendC::GlobalTensor<int32_t> const& gmOutputIdxOffset_;
    AscendC::LocalTensor<DtypeScore> scoreUb_;
    AscendC::LocalTensor<uint16_t> inputUb_;
    AscendC::LocalTensor<uint32_t> outIdxUb_;
    AscendC::LocalTensor<uint16_t> outValueUb_;
    AscendC::LocalTensor<uint32_t> topkTmpUb_;
    AscendC::LocalTensor<uint32_t> hisIndex1_;
    AscendC::LocalTensor<uint32_t> hisIndex2_;
    AscendC::LocalTensor<uint32_t> hist_;
    AscendC::LocalTensor<uint32_t> idxHigh_;
    AscendC::LocalTensor<uint32_t> idxLow_;
    AscendC::LocalTensor<uint32_t> nk_;
    AscendC::LocalTensor<uint16_t> tmpIndex_;
    uint32_t topK_{0};
    uint32_t s2Stride_{0};
    uint32_t s2Tile_{0};
    bool hasOutputIdxOffset_{false};
    uint32_t tileColAlign_{0};
    uint32_t topKAlign_{0};
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_BLOCK_TOPK_VF_HPP
