/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "quant_lightning_indexer_host.h"
#include "register/op_def_registry.h"

#define FI_STRING_INNER(value) #value
#define FI_STRING(value) FI_STRING_INNER(value)
#define FI_REGISTER_OP(op) OP_ADD(op)

namespace ops {
class FI_PACKAGE_OP_TYPE : public OpDef {
public:
    explicit FI_PACKAGE_OP_TYPE(const char* name) : OpDef(name)
    {
        // First-grid dtype signature: FP8 Q/K, FP32 weights/scales, INT32 meta,
        // INT32 indices, BF16 values. Keep one combination to match NativeBinding.
        Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND});
        Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .IgnoreContiguous();
        Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        Input("query_dequant_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        Input("key_dequant_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .IgnoreContiguous();
        Input("block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        for (const char* tensorName :
             {"cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k", "cmp_residual_k"}) {
            Input(tensorName)
                .ParamType(OPTIONAL)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND});
        }
        Output("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        Output("values")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND});

        Attr("topk").AttrType(REQUIRED).Int(0);
        Attr("quant_mode").AttrType(OPTIONAL).Int(1);
        Attr("mask_mode").AttrType(OPTIONAL).Int(3);
        Attr("cmp_ratio").AttrType(OPTIONAL).Int(4);
        Attr("max_seq_len_k").AttrType(OPTIONAL).Int(0);
        Attr("max_seqlen_q").AttrType(OPTIONAL).Int(-1);

        SetInferShape(InferQuantLightningIndexerShape)
            .SetInferDataType(InferQuantLightningIndexerType);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        AICore().SetTiling(FiQuantLightningIndexerTiling).AddConfig(FI_STRING(FI_COMPUTE_UNIT), config);
    }
};
FI_REGISTER_OP(FI_PACKAGE_OP_TYPE);
} // namespace ops
