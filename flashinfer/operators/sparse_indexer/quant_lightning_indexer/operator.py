"""Public semantics for the Ascend950 quant lightning indexer (first grid).

Hot-path formula (quant_mode=1)::

    score = sum_h weights[row, h] * relu(
        (q_scale[row, h] * k_scale[:]) * (query[row, h] @ key.T)
    )
    indices = stable_topk(score, topk)

Scheduling is folded into host tiling / the device kernel. There is no
separate public metadata operator in this phase.
"""

from types import SimpleNamespace

from flashinfer.operators import Invocation, OperatorDefinition, OutputSpec
from flashinfer.runtime import InferenceSpec

OP = "sparse_indexer.quant_lightning_indexer"
MAX_TOPK = 2048
MAX_TOTAL = 16384
MAX_HEADS = 64
HEAD_DIM = 128
MAX_BATCH = 16
FIRST_QUANT_MODE = 1
FIRST_MASK_MODE = 3
FIRST_CMP_RATIO = 4


def _dtype(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def _layout(value, name, allowed):
    value = str(value).upper()
    if value not in allowed:
        raise ValueError("%s must be one of %s" % (name, "/".join(allowed)))
    return value


def _positive(name, value):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(name + " must be a positive integer")


def _meta(arguments):
    """Validate first-grid shape/metadata facts without reading Tensor values."""
    query = arguments["query"]
    key = arguments["key"]
    weights = arguments["weights"]
    q_scale = arguments["query_dequant_scale"]
    k_scale = arguments["key_dequant_scale"]
    topk = arguments["topk"]
    quant_mode = arguments["quant_mode"]
    layout_q = _layout(arguments["layout_q"], "layout_q", ("TND",))
    layout_k = _layout(arguments["layout_k"], "layout_k", ("PA_BBND",))
    mask_mode = arguments["mask_mode"]
    cmp_ratio = arguments["cmp_ratio"]
    return_value = arguments["return_value"]
    max_seq_len_k = arguments["max_seq_len_k"]

    _positive("topk", topk)
    if topk > MAX_TOPK:
        raise ValueError("topk must be <= 2048")
    if quant_mode != FIRST_QUANT_MODE:
        raise ValueError("quant_mode must be 1 in the first grid")
    if mask_mode != FIRST_MASK_MODE:
        raise ValueError("mask_mode must be 3 in the first grid")
    if cmp_ratio != FIRST_CMP_RATIO:
        raise ValueError("cmp_ratio must be 4 in the first grid")
    if return_value not in (0, 1):
        raise ValueError("return_value must be 0 or 1")
    if not isinstance(max_seq_len_k, int) or max_seq_len_k <= 0:
        raise ValueError("max_seq_len_k must be a positive integer for PA_BBND")

    qshape = tuple(int(v) for v in query.shape)
    wshape = tuple(int(v) for v in weights.shape)
    kshape = tuple(int(v) for v in key.shape)
    qs_shape = tuple(int(v) for v in q_scale.shape)
    ks_shape = tuple(int(v) for v in k_scale.shape)

    if len(qshape) != 3 or len(wshape) != 2:
        raise ValueError("layout_q=TND expects query [Tq,N1,D] and weights [Tq,N1]")
    tq, n1, d = qshape
    if wshape != (tq, n1):
        raise ValueError("weights shape must match query [Tq,N1]")
    if qs_shape != (tq, n1):
        raise ValueError("query_dequant_scale shape must match weights [Tq,N1]")
    _positive("Tq", tq)
    if tq > MAX_TOTAL:
        raise ValueError("Tq must be <= 16384")
    if not 1 <= n1 <= MAX_HEADS:
        raise ValueError("N1 must be in [1, 64]")
    if d != HEAD_DIM:
        raise ValueError("head dim must be 128 in the first grid")

    if len(kshape) == 4:
        pages, block, nk, kd = kshape
    elif len(kshape) == 3:
        pages, block, kd = kshape
        nk = 1
    else:
        raise ValueError("layout_k=PA_BBND expects key [PAGES,BLOCK,Nk,D] or [PAGES,BLOCK,D]")
    if kd != d:
        raise ValueError("key head dim must match query")
    if nk != 1:
        raise ValueError("first grid requires Nk=1")
    if pages < 1 or block < 1:
        raise ValueError("paged key requires positive PAGES and BLOCK")
    expected_ks = (pages, block, nk) if len(kshape) == 4 else (pages, block)
    if ks_shape != expected_ks and ks_shape != (pages, block, 1) and ks_shape != (pages, block):
        raise ValueError("key_dequant_scale shape must match key without the D axis")

    if _dtype(query) != "float8_e4m3fn":
        raise ValueError("query requires float8_e4m3fn in the first grid")
    if _dtype(key) != "float8_e4m3fn":
        raise ValueError("key requires float8_e4m3fn in the first grid")
    if _dtype(weights) != "float32":
        raise ValueError("weights requires float32 in the first grid")
    if _dtype(q_scale) != "float32" or _dtype(k_scale) != "float32":
        raise ValueError("dequant scales require float32 in the first grid")

    cu_q = arguments["cu_seqlens_q"]
    cu_k = arguments["cu_seqlens_k"]
    seqused_q = arguments["seqused_q"]
    seqused_k = arguments["seqused_k"]
    residual = arguments["cmp_residual_k"]
    table = arguments["block_table"]

    if cu_q is None:
        raise ValueError("cu_seqlens_q is required for layout_q=TND")
    if len(tuple(cu_q.shape)) != 1 or tuple(cu_q.shape)[0] < 2:
        raise ValueError("cu_seqlens_q must have shape [B+1]")
    if _dtype(cu_q) != "int32":
        raise ValueError("cu_seqlens_q must be int32")
    batch = int(cu_q.shape[0]) - 1
    if batch < 1 or batch > MAX_BATCH:
        raise ValueError("batch must be in [1, 16]")
    if cu_k is not None:
        raise ValueError("cu_seqlens_k is not valid with layout_k=PA_BBND")
    if seqused_k is None:
        raise ValueError("seqused_k is required for layout_k=PA_BBND")
    if tuple(seqused_k.shape) != (batch,) or _dtype(seqused_k) != "int32":
        raise ValueError("seqused_k must be int32 [B]")
    if residual is None:
        raise ValueError("cmp_residual_k is required when cmp_ratio=4")
    if tuple(residual.shape) != (batch,) or _dtype(residual) != "int32":
        raise ValueError("cmp_residual_k must be int32 [B]")
    if table is None:
        raise ValueError("block_table is required for layout_k=PA_BBND")
    if len(tuple(table.shape)) != 2 or table.shape[0] != batch or table.shape[1] < 1:
        raise ValueError("block_table must be int32 [B,max_blocks]")
    if _dtype(table) != "int32":
        raise ValueError("block_table must be int32")
    if seqused_q is not None:
        if tuple(seqused_q.shape) != (batch,) or _dtype(seqused_q) != "int32":
            raise ValueError("seqused_q must be int32 [B]")

    indices_shape = (tq, nk, topk)
    values_shape = indices_shape if return_value else (0,)
    return SimpleNamespace(
        layout_q=layout_q,
        layout_k=layout_k,
        tq=tq,
        n1=n1,
        d=d,
        nk=nk,
        batch=batch,
        pages=pages,
        block=block,
        topk=topk,
        indices_shape=indices_shape,
        values_shape=values_shape,
        dtype=_dtype(query),
        kv_cache_type="paged",
    )


def infer(arguments, layouts):
    meta = _meta(arguments)
    # Page-axis tensors may arrive as a non-contiguous view (production PA).
    # Runner contig-copies them before the kernel; other operands stay ND-only.
    if layouts is not None:
        allow_strided = {"key", "key_dequant_scale"}
        bad = [
            name
            for name, layout in layouts.items()
            if layout != "ND" and name not in allow_strided
        ]
        if bad:
            raise ValueError(
                "quant_lightning_indexer requires contiguous ND tensors"
                + f" (non-ND: {bad})"
            )
    device = arguments["query"].device
    # Contig-copy page tensors here so Ascend C dispatch (which requires
    # contiguous operands) accepts production non-contiguous PA views.
    key = arguments["key"]
    k_scale = arguments["key_dequant_scale"]
    if hasattr(key, "is_contiguous") and not key.is_contiguous():
        key = key.contiguous()
    if hasattr(k_scale, "is_contiguous") and not k_scale.is_contiguous():
        k_scale = k_scale.contiguous()
    inputs = (
        arguments["query"],
        key,
        arguments["weights"],
        arguments["query_dequant_scale"],
        k_scale,
        arguments["block_table"],
        arguments["cu_seqlens_q"],
        arguments["cu_seqlens_k"],
        arguments["seqused_q"],
        arguments["seqused_k"],
        arguments["cmp_residual_k"],
    )
    return Invocation(
        inputs,
        (
            OutputSpec(meta.indices_shape, "int32", device),
            OutputSpec(meta.values_shape, "bfloat16", device),
        ),
        {
            "layout_q": meta.layout_q,
            "layout_k": meta.layout_k,
            "mask_mode": arguments["mask_mode"],
            "cmp_ratio": arguments["cmp_ratio"],
            "quant_mode": arguments["quant_mode"],
            "max_seq_len_k": arguments["max_seq_len_k"],
            "max_seqlen_q": arguments["max_seqlen_q"],
            "return_value": arguments["return_value"],
        },
        InferenceSpec(
            compute_dtype=meta.dtype,
            accumulator_dtype="float32",
            attention_variant="quant_lightning_indexer",
            kv_cache_type=meta.kv_cache_type,
            causal=True,
        ),
    )


_METADATA_TENSOR_NAMES = (
    "cu_seqlens_q",
    "cu_seqlens_k",
    "seqused_q",
    "seqused_k",
    "cmp_residual_k",
    "block_table",
)


def _strict_metadata(arguments):
    if arguments.get("validate_metadata", False):
        return True
    for name in _METADATA_TENSOR_NAMES:
        tensor = arguments.get(name)
        if tensor is not None and getattr(tensor.device, "type", None) != "cpu":
            return False
    return True


def validate_values(arguments):
    if not _strict_metadata(arguments):
        return
    meta = _meta(arguments)
    cu_q = arguments["cu_seqlens_q"].cpu().tolist()
    if cu_q[0] != 0 or cu_q[-1] != meta.tq or any(a > b for a, b in zip(cu_q, cu_q[1:])):
        raise ValueError("cu_seqlens_q must be nondecreasing, start at 0 and end at Tq")
    ratio = arguments["cmp_ratio"]
    if any(v < 0 or v >= ratio for v in arguments["cmp_residual_k"].cpu().tolist()):
        raise ValueError("cmp_residual_k must satisfy 0 <= residual < cmp_ratio")
    used = arguments["seqused_k"].cpu().tolist()
    width = int(arguments["block_table"].shape[1])
    if any((v + meta.block - 1) // meta.block > width for v in used):
        raise ValueError("block_table width is smaller than seqused_k requires")


OPERATOR = OperatorDefinition(
    name="quant_lightning_indexer",
    domain="sparse_indexer",
    op=OP,
    schema=(
        "(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, "
        "Tensor key_dequant_scale, int topk, int quant_mode=1, *, "
        "str layout_q='TND', str layout_k='PA_BBND', int mask_mode=3, int cmp_ratio=4, "
        "int max_seq_len_k=0, int max_seqlen_q=-1, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, "
        "Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? cmp_residual_k=None, "
        "Tensor? block_table=None, int return_value=0, "
        "bool validate_metadata=False) -> (Tensor, Tensor)"
    ),
    output_names=("indices", "values"),
    infer=infer,
    validate_values=validate_values,
    description=(
        "Ascend950 quant lightning indexer first grid. FP8 query/key with float32 "
        "scales/weights, TND/PA_BBND, causal compressed masking (mask_mode=3, "
        "cmp_ratio=4). Eager AOT-only and not graph capturable."
    ),
)
