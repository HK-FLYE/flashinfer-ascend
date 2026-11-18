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

namespace Reg = AscendC::Reg;

__simd_vf__ void ScaleHeadRowVF(__ubuf__ float *out, __ubuf__ float *score, __ubuf__ float *weight,
                                __ubuf__ float *kScale, uint32_t n1, uint32_t headStride, uint32_t cols)
{
    Reg::MaskReg mask = Reg::UpdateMask<float>(cols);
    Reg::RegTensor<float> sum;
    Reg::RegTensor<float> qk;
    Reg::RegTensor<float> w;
    Reg::RegTensor<float> scale;

    Reg::Duplicate(sum, 0.0f, mask);
    for (uint32_t head = 0; head < n1; ++head) {
        Reg::LoadAlign<float>(qk, score + head * headStride);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(w, weight + head);
        Reg::Relu(qk, qk, mask);
        Reg::MulAddDst(sum, qk, w, mask);
    }
    Reg::LoadAlign<float>(scale, kScale);
    Reg::Mul(sum, sum, scale, mask);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(out, sum, mask);
}

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
        offset = SparseIndexer::Common::AlignUp(offset, 32U);
        ubWBrcb_ = resource.ubBuf.template GetBufferByByte<float>(offset);
        offset += s1Tile_ * kFloatBlock * sizeof(float);
        // MaskPolicy uses up to s1Tile * kFloatVec temporary floats.
        offset = SparseIndexer::Common::AlignUp(offset, 32U);
        ubTmp_ = resource.ubBuf.template GetBufferByByte<float>(offset);
        offset += s1Tile_ * kFloatVec * sizeof(float);
        offset = SparseIndexer::Common::AlignUp(offset, 256U);
        ubKScale_ = resource.ubBuf.template GetBufferByByte<float>(offset);
    }

    CATLASS_DEVICE
    void ScaleHeadVF(uint32_t actualS1, uint32_t colPerAiv, uint32_t cols)
    {
        if (colPerAiv % 64U != 0U) {
            // Unaligned tail tiles cannot use Reg::LoadAlign. Keep an explicit
            // scalar fallback so these small S2 slices preserve FP32 semantics.
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
            for (uint32_t row = 0; row < actualS1; ++row) {
                for (uint32_t col = 0; col < cols; ++col) {
                    float sum = 0.0f;
                    for (uint32_t head = 0; head < n1_; ++head) {
                        const float value = ubScore_.GetValue((row * n1_ + head) * colPerAiv + col);
                        sum += (value > 0.0f ? value : 0.0f) * ubW_.GetValue(row * wStride_ + head);
                    }
                    ubOut_.SetValue(row * colPerAiv + col, sum * ubKScale_.GetValue(col));
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        constexpr uint32_t kBlockFloats = 64;
        for (uint32_t row = 0; row < actualS1; ++row) {
            for (uint32_t off = 0; off < cols; off += kBlockFloats) {
                const uint32_t blockCols = AscendC::Std::min(kBlockFloats, cols - off);
                ScaleHeadRowVF((__ubuf__ float *)ubOut_[row * colPerAiv + off].GetPhyAddr(),
                    (__ubuf__ float *)ubScore_[(row * n1_) * colPerAiv + off].GetPhyAddr(),
                    (__ubuf__ float *)ubW_[row * wStride_].GetPhyAddr(),
                    (__ubuf__ float *)ubKScale_[off].GetPhyAddr(), n1_, colPerAiv, blockCols);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
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
        ScaleHeadVF(actualS1, colPerAiv, cols);

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
