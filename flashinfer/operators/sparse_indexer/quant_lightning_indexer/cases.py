"""CPU golden and first-grid samples for quant_lightning_indexer.

quant_mode=1 reference matches the QLIv2 FP8 path used in production DSA:
dequant Q/K with float32 scales, matmul, ReLU, weighted head reduce, stable TopK.
"""

import math


CASE_PARAMETERS = (
    {
        # First acceptance case: tiny so the scalar AIV path finishes quickly.
        "name": "smoke_tiny",
        "tq": 1,
        "n1": 4,
        "d": 128,
        "topk": 4,
        "pages": 1,
        "block": 16,
        "batch": 1,
        "max_seq_len_k": 8,
        "seqused_k": 8,
        "residual": 0,
        "quant_mode": 1,
        "mask_mode": 3,
        "cmp_ratio": 4,
        "seed": 3,
    },
    {
        # N1=8 cube-path matrix member (kept for a smaller-head regression).
        "name": "hot_path_lite",
        "tq": 8,
        "n1": 8,
        "d": 128,
        "topk": 4,
        "pages": 4,
        "block": 16,
        "batch": 1,
        "max_seq_len_k": 64,
        "seqused_k": 48,
        "residual": 0,
        "quant_mode": 1,
        "mask_mode": 3,
        "cmp_ratio": 4,
        "seed": 7,
    },
    {
        # Production-ish N1=64 anchor. Fixed by capping s2Tile so ScaleHead
        # stays on the multi-row vector path (srcRep<=255) plus Muls/Add fallback.
        "name": "hot_path_anchor",
        "tq": 8,
        "n1": 64,
        "d": 128,
        "topk": 4,
        "pages": 4,
        "block": 16,
        "batch": 1,
        "max_seq_len_k": 64,
        "seqused_k": 48,
        "residual": 0,
        "quant_mode": 1,
        "mask_mode": 3,
        "cmp_ratio": 4,
        "seed": 7,
    },
    {
        "name": "short_query_topk_overflow",
        "tq": 1,
        "n1": 4,
        "d": 128,
        "topk": 16,
        "pages": 2,
        "block": 16,
        "batch": 1,
        "max_seq_len_k": 8,
        "seqused_k": 8,
        "residual": 3,
        "quant_mode": 1,
        "mask_mode": 3,
        "cmp_ratio": 4,
        "seed": 11,
    },
    {
        "name": "multi_batch_ragged",
        "tq": 6,
        "n1": 8,
        "d": 128,
        "topk": 4,
        "pages": 6,
        "block": 16,
        "batch": 2,
        "cu_q": (0, 2, 6),
        "max_seq_len_k": 32,
        "seqused_k": (10, 24),
        "residual": (1, 2),
        "quant_mode": 1,
        "mask_mode": 3,
        "cmp_ratio": 4,
        "seed": 13,
    },
    {
        "name": "topk_one",
        "tq": 4,
        "n1": 4,
        "d": 128,
        "topk": 1,
        "pages": 2,
        "block": 16,
        "batch": 1,
        "max_seq_len_k": 16,
        "seqused_k": 12,
        "residual": 0,
        "quant_mode": 1,
        "mask_mode": 3,
        "cmp_ratio": 4,
        "seed": 17,
    },
    {
        "name": "all_masked_empty_visible",
        "tq": 2,
        "n1": 4,
        "d": 128,
        "topk": 4,
        "pages": 1,
        "block": 16,
        "batch": 1,
        "max_seq_len_k": 4,
        "seqused_k": 0,
        "residual": 0,
        "quant_mode": 1,
        "mask_mode": 3,
        "cmp_ratio": 4,
        "seed": 19,
    },
    {
        "name": "residual_boundary",
        "tq": 4,
        "n1": 4,
        "d": 128,
        "topk": 8,
        "pages": 2,
        "block": 16,
        "batch": 1,
        "max_seq_len_k": 16,
        "seqused_k": 16,
        "residual": 3,
        "quant_mode": 1,
        "mask_mode": 3,
        "cmp_ratio": 4,
        "seed": 23,
    },
)


def _stable_topk(torch, scores, k):
    rows = []
    for row in scores:
        finite = torch.isfinite(row)
        ids = torch.arange(row.numel(), dtype=torch.int64)[finite]
        if ids.numel() == 0:
            rows.append(torch.full((k,), -1, dtype=torch.int32))
            continue
        values = row[ids]
        order = sorted(
            range(ids.numel()),
            key=lambda i: (-float(values[i]), int(ids[i])),
        )[:k]
        result = torch.full((k,), -1, dtype=torch.int32)
        if order:
            result[: len(order)] = ids[torch.tensor(order, dtype=torch.int64)].to(
                torch.int32
            )
        rows.append(result)
    return torch.stack(rows)


def _mask3_visible_len(local_q, q_len, k_act, cmp_ratio=1, residual=0):
    raw = (k_act * cmp_ratio + residual - q_len + local_q + 1) / float(cmp_ratio)
    if raw < 0:
        return 0
    return min(k_act, math.floor(raw))


def _apply_mask(torch, scores, q_act, k_act, *, mask_mode, cmp_ratio, residual):
    q_len, k_len = scores.shape
    if q_act < q_len:
        scores[q_act:q_len, :] = float("-inf")
    if k_act < k_len:
        scores[:, k_act:k_len] = float("-inf")
    if mask_mode == 3:
        for local_q in range(min(q_act, q_len)):
            visible = _mask3_visible_len(local_q, q_act, k_act, cmp_ratio, residual)
            if visible < k_act:
                scores[local_q, visible:k_act] = float("-inf")


def _gather_paged(torch, pages, scales, table, used, max_seq_len_k):
    if pages.dim() == 4:
        pages = pages.squeeze(2)
    if scales.dim() == 3 and scales.shape[-1] == 1:
        scales = scales.squeeze(-1)
    batch = table.shape[0]
    block = pages.shape[1]
    logical_k = []
    logical_s = []
    for b in range(batch):
        rows = torch.zeros((max_seq_len_k, pages.shape[-1]), dtype=torch.float32)
        scale_rows = torch.zeros((max_seq_len_k,), dtype=torch.float32)
        count = int(used[b])
        for token in range(min(count, max_seq_len_k)):
            page = int(table[b, token // block])
            if 0 <= page < pages.shape[0]:
                rows[token] = pages[page, token % block].float()
                scale_rows[token] = scales[page, token % block].float()
        logical_k.append(rows)
        logical_s.append(scale_rows)
    return logical_k, logical_s


def _reduce_quant(torch, query, key, weights, q_scale, k_scale):
    # quant_mode=1: dequant then matmul (equivalent to outer-scale ⊙ Q@K for per-token scales).
    q = query.float() * q_scale.float().unsqueeze(-1)
    k = key.float() * k_scale.float().unsqueeze(-1)
    scores = torch.relu(torch.einsum("tnd,sd->tns", q, k))
    return (scores * weights.float().unsqueeze(-1)).sum(dim=1)


def golden(
    torch,
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    topk,
    *,
    quant_mode=1,
    layout_q="TND",
    layout_k="PA_BBND",
    mask_mode=3,
    cmp_ratio=4,
    max_seq_len_k=0,
    cu_seqlens_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    return_value=0,
):
    if quant_mode != 1 or layout_q != "TND" or layout_k != "PA_BBND":
        raise NotImplementedError("CPU golden currently covers the first grid only")
    if block_table is None or seqused_k is None or cu_seqlens_q is None:
        raise ValueError("first-grid golden requires cu_seqlens_q, seqused_k and block_table")
    if cmp_ratio != 1 and cmp_residual_k is None:
        raise ValueError("cmp_residual_k is required when cmp_ratio!=1")

    q = query.cpu()
    w = weights.float().cpu()
    qs = query_dequant_scale.float().cpu()
    pages = key.cpu()
    ks = key_dequant_scale.cpu()
    table = block_table.cpu()
    used = seqused_k.cpu()
    residual = (
        cmp_residual_k.cpu().tolist()
        if cmp_residual_k is not None
        else [0] * table.shape[0]
    )
    ql = cu_seqlens_q.cpu().tolist()
    logical_k, logical_s = _gather_paged(torch, pages, ks, table, used, max_seq_len_k)

    rows_i = []
    rows_v = []
    for b in range(len(ql) - 1):
        q0, q1 = ql[b], ql[b + 1]
        q_len = q1 - q0
        score = _reduce_quant(
            torch, q[q0:q1], logical_k[b], w[q0:q1], qs[q0:q1], logical_s[b]
        )
        _apply_mask(
            torch,
            score,
            q_len,
            int(used[b]),
            mask_mode=mask_mode,
            cmp_ratio=cmp_ratio,
            residual=residual[b],
        )
        idx = _stable_topk(torch, score, topk)
        rows_i.append(idx)
        if return_value:
            safe = idx.long().clamp_min(0)
            vals = torch.gather(score, 1, safe)
            vals = vals.masked_fill(idx < 0, float("-inf"))
            rows_v.append(vals)

    indices = torch.cat(rows_i).unsqueeze(1).to(torch.int32)
    if return_value:
        values = torch.cat(rows_v).unsqueeze(1).to(torch.bfloat16)
    else:
        values = torch.empty((0,), dtype=torch.bfloat16)
    return indices, values


def make_cpu_inputs(torch, case=None):
    case = dict(CASE_PARAMETERS[0] if case is None else case)
    case.pop("name", None)
    rng = torch.Generator().manual_seed(int(case.get("seed", 7)))
    tq = case["tq"]
    n1 = case["n1"]
    d = case["d"]
    pages = case["pages"]
    block = case["block"]
    batch = case["batch"]
    max_seq_len_k = case["max_seq_len_k"]

    # Materialize FP8 via float32 then cast; keeps values in a mild range.
    q_f = torch.randn((tq, n1, d), generator=rng) * 0.1
    k_f = torch.randn((pages, block, 1, d), generator=rng) * 0.1
    query = q_f.to(torch.float8_e4m3fn)
    key = k_f.to(torch.float8_e4m3fn)
    weights = (torch.randn((tq, n1), generator=rng).abs() * 0.25 + 0.05).to(torch.float32)
    q_scale = (torch.randn((tq, n1), generator=rng).abs() * 0.05 + 0.01).to(torch.float32)
    k_scale = (torch.randn((pages, block, 1), generator=rng).abs() * 0.05 + 0.01).to(
        torch.float32
    )

    if "cu_q" in case:
        cu = torch.tensor(list(case["cu_q"]), dtype=torch.int32)
    else:
        cu = torch.tensor([0, tq], dtype=torch.int32)
        if batch != 1:
            raise ValueError("batch>1 sample requires cu_q")
    if isinstance(case.get("seqused_k"), (tuple, list)):
        seqused_k = torch.tensor(list(case["seqused_k"]), dtype=torch.int32)
    else:
        seqused_k = torch.tensor([int(case.get("seqused_k", max_seq_len_k))], dtype=torch.int32)
    if isinstance(case.get("residual"), (tuple, list)):
        residual = torch.tensor(list(case["residual"]), dtype=torch.int32)
    else:
        residual = torch.tensor([int(case.get("residual", 0))], dtype=torch.int32)

    max_blocks = max(1, (max_seq_len_k + block - 1) // block)
    table = torch.full((batch, max_blocks), -1, dtype=torch.int32)
    cursor = 0
    for b in range(batch):
        need = (int(seqused_k[b]) + block - 1) // block
        for i in range(need):
            table[b, i] = cursor
            cursor += 1
            if cursor >= pages:
                cursor = pages - 1

    return {
        "query": query,
        "key": key,
        "weights": weights,
        "query_dequant_scale": q_scale,
        "key_dequant_scale": k_scale,
        "topk": case["topk"],
        "quant_mode": case["quant_mode"],
        "layout_q": "TND",
        "layout_k": "PA_BBND",
        "mask_mode": case["mask_mode"],
        "cmp_ratio": case["cmp_ratio"],
        "max_seq_len_k": max_seq_len_k,
        "cu_seqlens_q": cu,
        "seqused_k": seqused_k,
        "cmp_residual_k": residual,
        "block_table": table,
        "return_value": int(case.get("return_value", 0)),
    }


def sample(torch, torch_npu, **configuration):
    """Framework acceptance sample: device args/options + CPU golden expected."""
    tensors = make_cpu_inputs(torch, configuration or None)
    expected = golden(torch, **tensors)

    def to_device(value):
        if not hasattr(value, "device"):
            return value
        tensor = value.contiguous().npu()
        # Keep ND storage for ND-declared tensors (avoid NCHW default on 4D).
        if tensor.dim() >= 4:
            return torch_npu.npu_format_cast(tensor, 2)
        return tensor

    args = (
        to_device(tensors["query"]),
        to_device(tensors["key"]),
        to_device(tensors["weights"]),
        to_device(tensors["query_dequant_scale"]),
        to_device(tensors["key_dequant_scale"]),
        int(tensors["topk"]),
        int(tensors["quant_mode"]),
    )
    options = dict(
        layout_q=tensors["layout_q"],
        layout_k=tensors["layout_k"],
        mask_mode=int(tensors["mask_mode"]),
        cmp_ratio=int(tensors["cmp_ratio"]),
        max_seq_len_k=int(tensors["max_seq_len_k"]),
        cu_seqlens_q=to_device(tensors["cu_seqlens_q"]),
        seqused_k=to_device(tensors["seqused_k"]),
        cmp_residual_k=to_device(tensors["cmp_residual_k"]),
        block_table=to_device(tensors["block_table"]),
        return_value=int(tensors["return_value"]),
        backend="ascendc",
    )
    return args, options, expected


# Exact indices match for the correctness-first kernel; empty values shape is trivial.
TOLERANCES = ((0.0, 0.0), (0.0, 0.0))


def qli_v2_oracle(torch, tensors):
    """Run ops-transformer QLIv2 on the same semantic inputs (NPU).

    Requires ``cann_ops_transformer`` and a live NPU. Builds the companion
    metadata tensor on the QLIv2 side; this repo does not expose metadata.
    """
    import cann_ops_transformer as cot

    def to_npu(value):
        if not hasattr(value, "device"):
            return value
        return value.contiguous().npu()

    query = to_npu(tensors["query"])
    key = to_npu(tensors["key"])
    weights = to_npu(tensors["weights"])
    q_scale = to_npu(tensors["query_dequant_scale"])
    k_scale = to_npu(tensors["key_dequant_scale"])
    cu = to_npu(tensors["cu_seqlens_q"])
    seqused = to_npu(tensors["seqused_k"])
    residual = to_npu(tensors["cmp_residual_k"])
    table = to_npu(tensors["block_table"])
    topk = int(tensors["topk"])
    n1 = int(query.shape[1])
    d = int(query.shape[2])
    tq = int(query.shape[0])
    batch = int(seqused.shape[0])
    max_seq = int(tensors["max_seq_len_k"])
    metadata = cot.quant_lightning_indexer_metadata(
        n1,
        1,
        d,
        topk,
        int(tensors["quant_mode"]),
        cu_seqlens_q=cu,
        seqused_k=seqused,
        cmp_residual_k=residual,
        batch_size=batch,
        max_seqlen_q=tq,
        max_seqlen_k=max_seq,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=int(tensors["mask_mode"]),
        cmp_ratio=int(tensors["cmp_ratio"]),
    )
    indices, values = cot.quant_lightning_indexer(
        query,
        key,
        weights,
        q_scale,
        k_scale,
        topk,
        int(tensors["quant_mode"]),
        cu_seqlens_q=cu,
        seqused_k=seqused,
        cmp_residual_k=residual,
        block_table=table,
        metadata=metadata,
        max_seqlen_q=tq,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=int(tensors["mask_mode"]),
        cmp_ratio=int(tensors["cmp_ratio"]),
        return_value=int(tensors.get("return_value", 0)),
    )
    return indices.cpu(), values.cpu()
