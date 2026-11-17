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
    make_cpu_inputs,
    qli_v2_oracle,
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
    tensors = make_cpu_inputs(torch, case)
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
    case = next(c for c in CASE_PARAMETERS if c["name"] == "hot_path_lite")
    tensors = make_cpu_inputs(torch, case)
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
    tensors = make_cpu_inputs(torch, case)
    tensors["return_value"] = 1
    indices, values = golden(torch, **tensors)
    assert values.shape == indices.shape
    assert values.dtype == torch.bfloat16
    # Invalid slots stay -inf after bf16 cast of -inf scores.
    invalid = indices < 0
    assert torch.isneginf(values[invalid]).all()


def test_cpu_golden_is_deterministic():
    case = CASE_PARAMETERS[0]
    a = make_cpu_inputs(torch, case)
    b = make_cpu_inputs(torch, case)
    ia, _ = golden(torch, **a)
    ib, _ = golden(torch, **b)
    assert torch.equal(ia, ib)


def test_mask3_visible_len_formula():
    # q_len=8, k_act=48, cmp=4, residual=0, local_q=7 → floor((192-8+7+1)/4)=48
    assert _mask3_visible_len(7, 8, 48, 4, 0) == 48
    # early query sees fewer keys
    assert _mask3_visible_len(0, 8, 48, 4, 0) == math.floor((192 - 8 + 0 + 1) / 4)
    assert _mask3_visible_len(0, 8, 8, 4, 0) < 8
    # residual expands the compressed window
    assert _mask3_visible_len(0, 4, 16, 4, 3) > _mask3_visible_len(0, 4, 16, 4, 0)


def test_cpu_golden_all_masked_fills_minus_one():
    case = next(c for c in CASE_PARAMETERS if c["name"] == "all_masked_empty_visible")
    tensors = make_cpu_inputs(torch, case)
    indices, _ = golden(torch, **tensors)
    assert (indices == -1).all()


def test_cpu_golden_residual_changes_selected_prefix():
    case = next(c for c in CASE_PARAMETERS if c["name"] == "residual_boundary")
    base = make_cpu_inputs(torch, case)
    with_res = dict(base)
    with_res["cmp_residual_k"] = torch.tensor([3], dtype=torch.int32)
    without = dict(base)
    without["cmp_residual_k"] = torch.tensor([0], dtype=torch.int32)
    idx_r, _ = golden(torch, **with_res)
    idx_0, _ = golden(torch, **without)
    # Larger residual must not shrink the visible prefix of any row.
    for local_q in range(case["tq"]):
        vis_r = _mask3_visible_len(local_q, case["tq"], 16, 4, 3)
        vis_0 = _mask3_visible_len(local_q, case["tq"], 16, 4, 0)
        assert vis_r >= vis_0
        chosen_r = idx_r[local_q, 0]
        chosen_0 = idx_0[local_q, 0]
        assert (chosen_r[chosen_r >= 0] < vis_r).all()
        assert (chosen_0[chosen_0 >= 0] < vis_0).all()


def test_cpu_golden_noncontig_page_axis_matches_contiguous():
    case = next(c for c in CASE_PARAMETERS if c["name"] == "pa_page_axis_noncontig")
    noncontig = make_cpu_inputs(torch, case)
    assert not noncontig["key"].is_contiguous()
    assert not noncontig["key_dequant_scale"].is_contiguous()
    contig = make_cpu_inputs(torch, dict(case, noncontig_page_axis=False))
    assert torch.equal(noncontig["key"].contiguous(), contig["key"].contiguous())
    ia, _ = golden(torch, **noncontig)
    ib, _ = golden(torch, **contig)
    assert torch.equal(ia, ib)


def test_cpu_golden_invalid_used_page_treated_as_zero():
    case = next(c for c in CASE_PARAMETERS if c["name"] == "oob_page_in_used_slot")
    tensors = make_cpu_inputs(torch, case)
    assert int(tensors["block_table"][0, 0]) < 0
    indices, _ = golden(torch, **tensors)
    assert indices.shape == (case["tq"], 1, case["topk"])
    assert ((indices >= 0) | (indices == -1)).all()
    fixed = dict(tensors)
    fixed["block_table"] = tensors["block_table"].clone()
    fixed["block_table"][0, 0] = 0
    idx_fixed, _ = golden(torch, **fixed)
    assert (idx_fixed[idx_fixed >= 0] < case["max_seq_len_k"]).all()


def _hand_built_inputs(*, weights=None, q_scale=None, k_scale=None, key_vals=None):
    tq, n1, d, topk = 1, 2, 128, 3
    query = torch.ones((tq, n1, d), dtype=torch.float8_e4m3fn)
    if key_vals is None:
        key = torch.ones((1, 16, 1, d), dtype=torch.float8_e4m3fn)
    else:
        key = key_vals
    if weights is None:
        weights = torch.ones((tq, n1), dtype=torch.float32)
    if q_scale is None:
        q_scale = torch.ones((tq, n1), dtype=torch.float32)
    if k_scale is None:
        k_scale = torch.ones((1, 16, 1), dtype=torch.float32)
    return dict(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=q_scale,
        key_dequant_scale=k_scale,
        topk=topk,
        quant_mode=1,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
        max_seq_len_k=4,
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        seqused_k=torch.tensor([4], dtype=torch.int32),
        cmp_residual_k=torch.tensor([0], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        return_value=0,
    )


def test_cpu_golden_stable_topk_prefers_lower_index_on_ties():
    indices, _ = golden(torch, **_hand_built_inputs())
    # With equal scores, stable TopK must pick the lowest indices first.
    assert indices[0, 0].tolist() == [0, 1, 2]


def test_cpu_golden_zero_weights_still_topk_by_index():
    weights = torch.zeros((1, 2), dtype=torch.float32)
    indices, _ = golden(torch, **_hand_built_inputs(weights=weights))
    # All scores zero → still stable by index among visible keys.
    assert indices[0, 0].tolist() == [0, 1, 2]


def test_cpu_golden_extreme_scales_finite_indices():
    q_scale = torch.full((1, 2), 1e-6, dtype=torch.float32)
    k_scale = torch.full((1, 16, 1), 1e4, dtype=torch.float32)
    indices, values = golden(
        torch, **{**_hand_built_inputs(q_scale=q_scale, k_scale=k_scale), "return_value": 1}
    )
    assert ((indices >= 0) | (indices == -1)).all()
    assert torch.isfinite(values[indices >= 0]).all()


def test_cpu_golden_negative_qk_masked_by_relu_still_fills():
    # Opposite-sign Q/K → ReLU zeros all scores; TopK still fills by stable order.
    key = (-torch.ones((1, 16, 1, 128))).to(torch.float8_e4m3fn)
    indices, _ = golden(torch, **_hand_built_inputs(key_vals=key))
    assert indices[0, 0].tolist() == [0, 1, 2]


def _require_npu():
    import torch_npu  # noqa: F401

    assert torch.npu.is_available() and torch.npu.device_count() > 0, (
        "NPU required; do not treat skip as pass"
    )
    torch.npu.set_device(0)


def _qli_v2_indices_sorted_by_values(torch, tensors):
    """Normalize QLIv2 device TopK the same way ops-transformer pytest does.

    QLIv2 NPU emits the TopK *set* but not score-descending order in the index
    slots. With ``return_value=1``, their tests re-sort indices by the returned
    values before comparing. That is the semantic order we match.
    """
    ref, values = qli_v2_oracle(torch, {**tensors, "return_value": 1})
    assert values is not None and values.numel() == ref.numel(), (ref.shape, getattr(values, "shape", None))
    _, order = values.float().sort(dim=-1, descending=True)
    return torch.gather(ref, dim=-1, index=order)


@pytest.mark.real_compile
def test_smoke_tiny_matches_qli_v2_score_order():
    """NPU + CPU golden vs QLIv2 after value-resort (ops-transformer protocol).

    Single process: run our kernel *before* cann_ops_transformer. Their JIT
    extension links with our AOT ``op_api`` search path and can break a later
    ``FiQuantLightningIndexer`` GetWorkspaceSize (161001).
    """
    import os

    _require_npu()
    if not os.environ.get("FLASHINFER_AOT_PATHS"):
        raise AssertionError(
            "FLASHINFER_AOT_PATHS is required for NPU vs QLIv2; missing package is a failure"
        )
    import torch_npu
    import flashinfer
    from flashinfer.operators.sparse_indexer.quant_lightning_indexer.cases import sample

    case = next(c for c in CASE_PARAMETERS if c["name"] == "smoke_tiny")
    args, options, _ = sample(torch, torch_npu, **case)
    out_idx, _ = flashinfer.quant_lightning_indexer(*args, **options)
    tensors = make_cpu_inputs(torch, case)
    expected, _ = golden(torch, **tensors)
    assert torch.equal(out_idx.cpu(), expected), (out_idx.cpu(), expected)

    raw, _ = qli_v2_oracle(torch, tensors)
    # Raw device slot order is not score-sorted (documented, not a pass criterion).
    assert not torch.equal(raw, expected), (raw, expected)
    ordered = _qli_v2_indices_sorted_by_values(torch, tensors)
    assert torch.equal(ordered.to(expected.dtype), expected), (ordered, expected)
    assert torch.equal(out_idx.cpu().to(ordered.dtype), ordered.cpu()), (
        out_idx.cpu(),
        ordered.cpu(),
    )
