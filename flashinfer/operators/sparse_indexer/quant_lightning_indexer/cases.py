"""CPU golden stubs and first-grid sample matrix for quant_lightning_indexer.

P1 expands golden to full FP8 dequant + ReLU + stable TopK. P0 only needs the
schema export surface (sample / CASE_PARAMETERS / golden placeholders).
"""


CASE_PARAMETERS = (
    {
        "name": "hot_path_anchor",
        "tq": 8,
        "n1": 64,
        "d": 128,
        "topk": 4,
        "pages": 4,
        "block": 16,
        "batch": 1,
        "max_seq_len_k": 64,
        "quant_mode": 1,
        "mask_mode": 3,
        "cmp_ratio": 4,
    },
)


def sample(torch, case=None):
    case = dict(CASE_PARAMETERS[0] if case is None else case)
    tq = case["tq"]
    n1 = case["n1"]
    d = case["d"]
    pages = case["pages"]
    block = case["block"]
    batch = case["batch"]
    query = torch.zeros((tq, n1, d), dtype=torch.float8_e4m3fn)
    key = torch.zeros((pages, block, 1, d), dtype=torch.float8_e4m3fn)
    weights = torch.ones((tq, n1), dtype=torch.float32)
    q_scale = torch.ones((tq, n1), dtype=torch.float32)
    k_scale = torch.ones((pages, block, 1), dtype=torch.float32)
    cu = torch.tensor([0, tq], dtype=torch.int32)
    if batch != 1:
        raise ValueError("P0 sample only covers batch=1")
    seqused_k = torch.tensor([min(case["max_seq_len_k"], pages * block)], dtype=torch.int32)
    residual = torch.tensor([0], dtype=torch.int32)
    table = torch.zeros((1, max(1, (int(seqused_k[0]) + block - 1) // block)), dtype=torch.int32)
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
        "max_seq_len_k": case["max_seq_len_k"],
        "cu_seqlens_q": cu,
        "seqused_k": seqused_k,
        "cmp_residual_k": residual,
        "block_table": table,
        "return_value": 0,
    }


def golden(torch, **kwargs):
    raise NotImplementedError("CPU golden lands in P1")
