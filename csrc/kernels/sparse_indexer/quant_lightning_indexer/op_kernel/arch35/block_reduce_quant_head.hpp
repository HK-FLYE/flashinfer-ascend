/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * Vector reduce for QLI quant_mode=1:
 *   score[s2] = k_scale[s2] * sum_h (w[h] * q_scale[h]) * ReLU(Q @ K^T)[h, s2]
 * Cube produces raw FP8 matmul into shared UB; this block owns scales + ReLU +
 * head reduce + mask3, matching QLIv2's vector ownership of float scales.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_ARCH35_BLOCK_REDUCE_QUANT_HEAD_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_ARCH35_BLOCK_REDUCE_QUANT_HEAD_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "kernel_operator.h"

#include "block/block_reduce_head.hpp"
#include "block/mask_policy.hpp"
#include "sparse_indexer_common.hpp"

namespace QuantLightningIndexerKernelDetail {

template <class MaskPolicy_ = SparseIndexer::Common::Block::LiV2MaskPolicy<float>>
class BlockReduceQuantHead {
public:
    using DtypeScore = float;
    using DtypeW = float;
    using DtypeDst = float;
    using ArchTag = Catlass::Arch::Ascend950;
    using MaskPolicy = MaskPolicy_;

    static constexpr uint32_t kFloatBlock = Catlass::BYTE_PER_BLK / sizeof(float);
    static constexpr uint32_t kFloatVec = 256 / sizeof(float);
    static constexpr uint32_t kMaxRepeat = 255;

    struct Params {
        uint32_t n1{0};
        uint32_t s2{0};
        uint32_t s1Tile{0};
        uint32_t s2Tile{0};
        int32_t maskMode{3};
        int32_t cmpRatio{4};
        uint32_t blockSize{0};
        uint32_t maxBlocksPerBatch{0};
        uint32_t numPages{0};
        uint64_t kScaleStride0{0};
    };

    CATLASS_DEVICE
    BlockReduceQuantHead(
        AscendC::LocalTensor<float>& ubScore, Catlass::Arch::Resource<ArchTag>& resource,
        AscendC::GlobalTensor<float> const& gmW, AscendC::GlobalTensor<float> const& gmQScale,
        AscendC::GlobalTensor<float> const& gmKScale, AscendC::GlobalTensor<int32_t> const& gmBlockTable,
        AscendC::GlobalTensor<float> const& gmDst, Params const& params)
        : ubScore_(ubScore),
          gmW_(gmW),
          gmQScale_(gmQScale),
          gmKScale_(gmKScale),
          gmBlockTable_(gmBlockTable),
          gmDst_(gmDst),
          params_(params),
          n1_(params.n1),
          s2_(params.s2),
          s1Tile_(params.s1Tile),
          s2Tile_(params.s2Tile),
          maskPolicy_(ubOut_, ubWBrcb_, ubTmp_)
    {
        wStride_ = SparseIndexer::Common::AlignUp(n1_, kFloatBlock);
        const uint32_t aivPerAic = AscendC::GetSubBlockNum();
        const uint32_t colPerAiv =
            SparseIndexer::Common::AlignUp(s2Tile_, 16U * aivPerAic) / aivPerAic;
        uint32_t offset = s1Tile_ * n1_ * colPerAiv * sizeof(float);
        ubOut_ = resource.ubBuf.template GetBufferByByte<float>(offset);
        offset += s1Tile_ * colPerAiv * sizeof(float);
        ubW_ = resource.ubBuf.template GetBufferByByte<float>(offset);
        offset += s1Tile_ * wStride_ * sizeof(float);
        ubWIn_ = resource.ubBuf.template GetBufferByByte<float>(offset);
        offset += s1Tile_ * wStride_ * sizeof(float);
        ubQScaleIn_ = resource.ubBuf.template GetBufferByByte<float>(offset);
        offset += s1Tile_ * wStride_ * sizeof(float);
        ubWCol_ = resource.ubBuf.template GetBufferByByte<float>(offset);
        offset += s1Tile_ * sizeof(float);
        ubWBrcb_ = resource.ubBuf.template GetBufferByByte<float>(offset);
        offset += s1Tile_ * kFloatBlock * sizeof(float);
        // ScaleHead's non-first-head path writes up to s1Tile * kFloatVec temps.
        ubTmp_ = resource.ubBuf.template GetBufferByByte<float>(offset);
        offset += s1Tile_ * kFloatVec * sizeof(float);
        ubKScale_ = resource.ubBuf.template GetBufferByByte<float>(offset);
    }

    CATLASS_DEVICE
    void ScaleHead(uint32_t head, uint32_t actualS1, uint32_t colPerAiv, uint32_t cols, bool first)
    {
        uint32_t dstRep = colPerAiv / kFloatBlock;
        uint32_t srcRep = n1_ * colPerAiv / kFloatBlock;
        // srcRep = n1*colPerAiv/8; when it exceeds 255 the multi-row vector
        // path is illegal. Prefer per-token Muls/Add over Axpy (Axpy on this
        // path diverged from CPU golden for n1=64 / colPerAiv=32).
        if (dstRep > kMaxRepeat || srcRep > kMaxRepeat) {
            for (uint32_t token = 0; token < actualS1; ++token) {
                auto score = ubScore_[(token * n1_ + head) * colPerAiv];
                auto out = ubOut_[token * colPerAiv];
                float weight = ubW_.GetValue(token * wStride_ + head);
                if (first) {
                    AscendC::Muls(out, score, weight, cols);
                } else {
                    AscendC::Muls(ubTmp_, score, weight, cols);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(out, out, ubTmp_, cols);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }

        uint32_t packed = SparseIndexer::Common::AlignUp(actualS1, kFloatBlock);
        for (uint32_t token = 0; token < actualS1; ++token) {
            ubWCol_.SetValue(token, ubW_.GetValue(token * wStride_ + head));
        }
        for (uint32_t token = actualS1; token < packed; ++token) {
            ubWCol_.SetValue(token, 0.0f);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(
            ubWBrcb_.template ReinterpretCast<uint32_t>(), ubWCol_.template ReinterpretCast<uint32_t>(),
            packed / kFloatBlock, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        for (uint32_t rowBase = 0; rowBase < actualS1; rowBase += kMaxRepeat) {
            uint32_t rows = AscendC::Std::min(actualS1 - rowBase, kMaxRepeat);
            auto out = ubOut_[rowBase * colPerAiv];
            auto score = ubScore_[(rowBase * n1_ + head) * colPerAiv];
            auto scale = ubWBrcb_[rowBase * kFloatBlock];
            uint32_t full = cols / kFloatVec;
            uint32_t remain = cols % kFloatVec;
            for (uint32_t chunk = 0; chunk < full; ++chunk) {
                uint32_t off = chunk * kFloatVec;
                if (first) {
                    AscendC::Mul<float, false>(
                        out[off], score[off], scale, (uint64_t)0, rows,
                        AscendC::BinaryRepeatParams(1, 1, 0, dstRep, srcRep, 1));
                } else {
                    AscendC::Mul<float, false>(
                        ubTmp_, score[off], scale, (uint64_t)0, rows,
                        AscendC::BinaryRepeatParams(1, 1, 0, kFloatVec / kFloatBlock, srcRep, 1));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<float, false>(
                        out[off], out[off], ubTmp_, (uint64_t)0, rows,
                        AscendC::BinaryRepeatParams(1, 1, 1, dstRep, dstRep, kFloatVec / kFloatBlock));
                }
            }
            if (remain != 0) {
                uint64_t mask = 0;
                for (uint32_t bit = 0; bit < remain; ++bit) {
                    mask |= (uint64_t)1 << bit;
                }
                AscendC::SetVectorMask<int8_t>(0x0, mask);
                uint32_t off = full * kFloatVec;
                if (first) {
                    AscendC::Mul<float, false>(
                        out[off], score[off], scale, (uint64_t)0, rows,
                        AscendC::BinaryRepeatParams(1, 1, 0, dstRep, srcRep, 1));
                } else {
                    AscendC::Mul<float, false>(
                        ubTmp_, score[off], scale, (uint64_t)0, rows,
                        AscendC::BinaryRepeatParams(1, 1, 0, kFloatVec / kFloatBlock, srcRep, 1));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<float, false>(
                        out[off], out[off], ubTmp_, (uint64_t)0, rows,
                        AscendC::BinaryRepeatParams(1, 1, 1, dstRep, dstRep, kFloatVec / kFloatBlock));
                }
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
    }

    CATLASS_DEVICE void operator()(
        uint32_t s1Start, uint32_t actualS1, uint32_t s2Start, uint32_t actualS2, uint32_t localQStart,
        uint32_t qAct, uint32_t kAct, uint32_t reqKStart, int32_t residual, bool maskActive, uint32_t batchIdx)
    {
        const uint32_t aivPerAic = AscendC::GetSubBlockNum();
        const uint32_t colPerAiv =
            SparseIndexer::Common::AlignUp(actualS2, 16U * aivPerAic) / aivPerAic;
        const uint32_t colBegin = AscendC::GetSubBlockIdx() * colPerAiv;
        if (colBegin >= actualS2) {
            return;
        }
        const uint32_t cols = AscendC::Std::min(colPerAiv, actualS2 - colBegin);
        const uint32_t alignedS1 = SparseIndexer::Common::AlignUp(actualS1, kFloatBlock);

        AscendC::Relu(ubScore_, ubScore_, alignedS1 * n1_ * colPerAiv);
        AscendC::PipeBarrier<PIPE_V>();

        const uint32_t rightPad = wStride_ - n1_;
        AscendC::DataCopyExtParams load{
            static_cast<uint16_t>(actualS1), static_cast<uint32_t>(n1_ * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{rightPad != 0, 0, static_cast<uint8_t>(rightPad), 0.0f};
        AscendC::DataCopyPad(ubWIn_, gmW_[s1Start * n1_], load, pad);
        AscendC::DataCopyPad(ubQScaleIn_, gmQScale_[s1Start * n1_], load, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        if (alignedS1 > actualS1) {
            AscendC::Duplicate(ubWIn_[actualS1 * wStride_], 0.0f, (alignedS1 - actualS1) * wStride_);
            AscendC::Duplicate(ubQScaleIn_[actualS1 * wStride_], 0.0f, (alignedS1 - actualS1) * wStride_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Mul(ubW_, ubWIn_, ubQScaleIn_, alignedS1 * wStride_);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t h = 0; h < n1_; ++h) {
            ScaleHead(h, actualS1, colPerAiv, cols, h == 0);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Gather PA key scales for this vector's S2 slice, then Mul into scores.
        for (uint32_t col = 0; col < cols; ++col) {
            const uint32_t tok = s2Start + colBegin + col;
            const uint32_t block = params_.blockSize == 0 ? 0U : (tok / params_.blockSize);
            const uint32_t off = params_.blockSize == 0 ? 0U : (tok % params_.blockSize);
            int32_t page = -1;
            if (block < params_.maxBlocksPerBatch) {
                page = gmBlockTable_.GetValue(
                    static_cast<int64_t>(batchIdx) * params_.maxBlocksPerBatch + block);
            }
            const float kScale =
                (page >= 0 && static_cast<uint32_t>(page) < params_.numPages)
                    ? gmKScale_.GetValue(
                          static_cast<uint32_t>(page) * params_.kScaleStride0 + off)
                    : 0.0f;
            ubKScale_.SetValue(col, kScale);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < actualS1; ++row) {
            AscendC::Mul(ubOut_[row * colPerAiv], ubOut_[row * colPerAiv], ubKScale_, cols);
        }
        AscendC::PipeBarrier<PIPE_V>();

        maskPolicy_.Apply(
            actualS1, cols, colPerAiv, colBegin, s2Start, localQStart, qAct, kAct, reqKStart, residual,
            maskActive, params_.maskMode, params_.cmpRatio);
        SparseIndexer::Common::Block::DirectWorkspaceStore<float, float>::StoreTile(
            gmDst_[s1Start * s2_ + s2Start + colBegin], ubOut_, actualS1, cols, colPerAiv, s2_);
    }

private:
    AscendC::LocalTensor<float>& ubScore_;
    AscendC::LocalTensor<float> ubOut_;
    AscendC::LocalTensor<float> ubW_;
    AscendC::LocalTensor<float> ubWIn_;
    AscendC::LocalTensor<float> ubQScaleIn_;
    AscendC::LocalTensor<float> ubWCol_;
    AscendC::LocalTensor<float> ubWBrcb_;
    AscendC::LocalTensor<float> ubTmp_;
    AscendC::LocalTensor<float> ubKScale_;
    AscendC::GlobalTensor<float> const& gmW_;
    AscendC::GlobalTensor<float> const& gmQScale_;
    AscendC::GlobalTensor<float> const& gmKScale_;
    AscendC::GlobalTensor<int32_t> const& gmBlockTable_;
    AscendC::GlobalTensor<float> const& gmDst_;
    Params params_{};
    uint32_t n1_{0};
    uint32_t s2_{0};
    uint32_t s1Tile_{0};
    uint32_t s2Tile_{0};
    uint32_t wStride_{0};
    MaskPolicy maskPolicy_;
};

} // namespace QuantLightningIndexerKernelDetail

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_ARCH35_BLOCK_REDUCE_QUANT_HEAD_HPP
