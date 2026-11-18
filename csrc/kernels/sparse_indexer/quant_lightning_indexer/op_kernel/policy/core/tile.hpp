/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_TILE_HPP
#define FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_TILE_HPP

#include <cstdint>

namespace SparseIndexer::QuantLightningIndexer::Policy {

template <class Element_>
struct TileView {
    using Element = Element_;
    Element* data{nullptr};
    uint32_t rows{0};
    uint32_t cols{0};
    uint32_t rowStride{0};
    uint32_t validRows{0};
    uint32_t validCols{0};
};

using QkTileView = TileView<float>;
using ScoreTileView = TileView<float>;
using IndexTileView = TileView<int32_t>;
using KeyTileView = TileView<uint8_t>;
using QueryTileView = TileView<uint8_t>;

struct ReduceParams {
    uint32_t n1{0};
    uint32_t qScaleStride{0};
    uint32_t kScaleStride{0};
    uint32_t blockSize{0};
    uint32_t maxBlocksPerBatch{0};
    uint32_t numPages{0};
    uint32_t maskMode{3};
    uint32_t cmpRatio{4};
};

} // namespace SparseIndexer::QuantLightningIndexer::Policy

#endif // FLASHINFER_QUANT_LIGHTNING_INDEXER_POLICY_TILE_HPP
