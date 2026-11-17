"""L0/L1 tests for sparse_indexer.quant_lightning_indexer.

First grid: Ascend950 DeepSeek V4 DSA hot path
(quant_mode=1, TND/PA_BBND, mask_mode=3, cmp_ratio=4, D=128, return_value=0).

L0 = public schema / NativeBinding contract.
L1 = CPU golden semantic boundaries (device kernel still unverified).
"""

import math

import pytest
import torch

from flashinfer.operators.sparse_indexer.quant_lightning_indexer.cases import (
    CASE_PARAMETERS,
    _mask3_visible_len,
    golden,
    sample,
)
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


@pytest.mark.parametrize("case", CASE_PARAMETERS, ids=lambda c: c["name"])
def test_cpu_golden_indices_shape_and_padding(case):
    tensors = sample(torch, case)
    indices, values = golden(torch, **tensors)
    tq = case["tq"]
    topk = case["topk"]
    assert indices.shape == (tq, 1, topk)
    assert indices.dtype == torch.int32
    assert values.shape == (0,)
    assert values.dtype == torch.bfloat16
    assert ((indices >= 0) | (indices == -1)).all()
    # Padding slots after exhausted valid keys must be -1.
    for row in indices[:, 0, :]:
        valid = row[row >= 0]
        assert valid.numel() == valid.unique().numel()
        if valid.numel() < topk:
            assert (row[valid.numel() :] == -1).all()


def test_cpu_golden_mask3_respects_visibility_window():
    case = next(c for c in CASE_PARAMETERS if c["name"] == "hot_path_anchor")
    tensors = sample(torch, case)
    # Force residual=0 so the closed-form window is easy to check.
    tensors["cmp_residual_k"] = torch.zeros_like(tensors["cmp_residual_k"])
    indices, _ = golden(torch, **tensors)
    q_len = case["tq"]
    k_act = int(tensors["seqused_k"][0])
    cmp_ratio = case["cmp_ratio"]
    for local_q in range(q_len):
        visible = _mask3_visible_len(local_q, q_len, k_act, cmp_ratio, 0)
        chosen = indices[local_q, 0]
        chosen = chosen[chosen >= 0]
        if visible > 0:
            assert (chosen < visible).all()
        else:
            assert chosen.numel() == 0


def test_cpu_golden_return_value_matches_score_gather():
    case = next(c for c in CASE_PARAMETERS if c["name"] == "short_query_topk_overflow")
    tensors = sample(torch, case)
    tensors["return_value"] = 1
    indices, values = golden(torch, **tensors)
    assert values.shape == indices.shape
    assert values.dtype == torch.bfloat16
    # Invalid slots stay -inf after bf16 cast of -inf scores.
    invalid = indices < 0
    assert torch.isneginf(values[invalid]).all()


def test_cpu_golden_is_deterministic():
    case = CASE_PARAMETERS[0]
    a = sample(torch, case)
    b = sample(torch, case)
    ia, _ = golden(torch, **a)
    ib, _ = golden(torch, **b)
    assert torch.equal(ia, ib)


def test_mask3_visible_len_formula():
    # q_len=8, k_act=48, cmp=4, residual=0, local_q=7 → floor((192-8+7+1)/4)=48
    assert _mask3_visible_len(7, 8, 48, 4, 0) == 48
    # early query sees fewer keys
    assert _mask3_visible_len(0, 8, 48, 4, 0) == math.floor((192 - 8 + 0 + 1) / 4)
    assert _mask3_visible_len(0, 8, 8, 4, 0) < 8
