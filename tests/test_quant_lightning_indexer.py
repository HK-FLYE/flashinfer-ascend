"""L0 public contract tests for sparse_indexer.quant_lightning_indexer.

First grid: Ascend950 DeepSeek V4 DSA hot path
(quant_mode=1, TND/PA_BBND, mask_mode=3, cmp_ratio=4, D=128, return_value=0).
"""

import pytest

from flashinfer.operators.sparse_indexer.quant_lightning_indexer.operator import OPERATOR


class Tensor:
    def __init__(self, shape, dtype="float8_e4m3fn", device="npu:0"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device


def args(**overrides):
    base = dict(
        query=Tensor((8, 64, 128)),
        key=Tensor((4, 16, 1, 128)),
        weights=Tensor((8, 64), "float32"),
        query_dequant_scale=Tensor((8, 64), "float32"),
        key_dequant_scale=Tensor((4, 16, 1), "float32"),
        topk=4,
        quant_mode=1,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        max_seq_len_k=64,
        max_seqlen_q=-1,
        cu_seqlens_q=Tensor((2,), "int32"),
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=Tensor((1,), "int32"),
        cmp_residual_k=Tensor((1,), "int32"),
        block_table=Tensor((1, 8), "int32"),
        return_value=0,
        validate_metadata=False,
    )
    base.update(overrides)
    return base


def infer(**kwargs):
    return OPERATOR.infer(args(**kwargs), None)


def test_hot_path_contract_keeps_two_outputs():
    invocation = infer()
    assert [o.shape for o in invocation.outputs] == [(8, 1, 4), (0,)]
    assert [o.dtype for o in invocation.outputs] == ["int32", "bfloat16"]
    assert invocation.attributes["quant_mode"] == 1
    assert invocation.attributes["mask_mode"] == 3
    assert invocation.attributes["cmp_ratio"] == 4
    assert invocation.inference.kv_cache_type == "paged"
    assert invocation.inference.causal is True


def test_return_value_one_allocates_values_shape():
    invocation = infer(return_value=1)
    assert [o.shape for o in invocation.outputs] == [(8, 1, 4), (8, 1, 4)]


def test_topk_larger_than_visible_tokens_keeps_output_width():
    invocation = infer(topk=2048, max_seq_len_k=16)
    assert invocation.outputs[0].shape == (8, 1, 2048)


@pytest.mark.parametrize(
    "change,match",
    [
        (dict(layout_q="BSND"), "layout_q"),
        (dict(layout_k="TND"), "layout_k"),
        (dict(quant_mode=2), "quant_mode"),
        (dict(mask_mode=0), "mask_mode"),
        (dict(cmp_ratio=1), "cmp_ratio"),
        (dict(topk=0), "topk"),
        (dict(return_value=2), "return_value"),
        (dict(query=Tensor((8, 64, 64))), "head dim"),
        (dict(query=Tensor((8, 65, 128)), weights=Tensor((8, 65), "float32"),
              query_dequant_scale=Tensor((8, 65), "float32")), "N1"),
        (dict(cmp_residual_k=None), "cmp_residual_k"),
        (dict(block_table=None), "block_table"),
        (dict(seqused_k=None), "seqused_k"),
        (dict(cu_seqlens_q=None), "cu_seqlens_q"),
        (dict(max_seq_len_k=0), "max_seq_len_k"),
        (dict(weights=Tensor((8, 64), "float16")), "weights"),
        (dict(query=Tensor((8, 64, 128), "float16")), "query"),
    ],
)
def test_first_grid_rejects_out_of_scope_or_invalid(change, match):
    with pytest.raises(ValueError, match=match):
        infer(**change)


def test_top_level_export_is_installed():
    import flashinfer

    assert callable(flashinfer.quant_lightning_indexer)


def test_native_recipe_declares_ordered_abi():
    from flashinfer.ascendc.sparse_indexer.quant_lightning_indexer.implementation import (
        IMPLEMENTATION,
        NATIVE,
    )

    assert IMPLEMENTATION.kernel_id == (
        "ascendc/sparse_indexer/quant_lightning_indexer/catlass/op_package/Ascend950"
    )
    assert [item.name for item in NATIVE.tensors] == [
        "query",
        "key",
        "weights",
        "query_dequant_scale",
        "key_dequant_scale",
        "block_table",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "seqused_q",
        "seqused_k",
        "cmp_residual_k",
        "indices",
        "values",
    ]
    assert NATIVE.allow_jit_fallback is False
    assert NATIVE.build.device_entry == "fi_quant_lightning_indexer"
    assert NATIVE.build.op_type == "FiQuantLightningIndexer"
    assert NATIVE.build.recipe_id == "quant_lightning_indexer"
    assert NATIVE.dtype_signatures == (
        (
            "float8_e4m3fn",
            "float8_e4m3fn",
            "float32",
            "float32",
            "float32",
            "int32",
            "int32",
            "int32",
            "int32",
            "int32",
            "int32",
            "int32",
            "bfloat16",
        ),
    )
