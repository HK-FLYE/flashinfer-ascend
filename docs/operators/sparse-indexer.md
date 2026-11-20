# Sparse Indexer 接口与实现范围

面向集成者：本页说明 `sparse_indexer` 逻辑 domain 的当前成员、公共接口，以及与 sparse attention 的组合边界。

## 当前 family

公共 domain 为 `sparse_indexer`。当前登记一个成员：

| logical op | public API | 状态 |
| --- | --- | --- |
| `sparse_indexer.quant_lightning_indexer` | `flashinfer.quant_lightning_indexer` | Ascend950 首期热点格：CPU golden + cube/vector 路径已对拍当前 sample 矩阵 indices（含 N1=64 锚点）；与 QLIv2 在 value-resort 后严次序一致 |

`sparse_indexer.lightning_indexer` / `general_indexer` 未在本分支创建。独立 metadata 产品算子首期不做；分核调度内嵌 tiling/kernel。

## QuantLightningIndexer（首期热点格）

公共入口为 `flashinfer.quant_lightning_indexer`，返回 `(indices, values)`。首期生产语义固定只消费 indices，`values` 始终为空形状 `bfloat16` Tensor；不提供 `return_value` 开关。计算语义对齐 DeepSeek V4 DSA 生产热点：

```text
score = sum_h W * ReLU((Sq @ Sk^T) ⊙ (Qq @ Kq^T))
indices = stable_topk(score, topk)
```

首期只声明下列组合（其余组合拒绝或未实现）：

| 项 | 值 |
| --- | --- |
| SoC | Ascend950 |
| `quant_mode` | 1（FP8 e4m3） |
| Q/K | `float8_e4m3fn` |
| scale / W | `float32` |
| `layout_q` / `layout_k` | `TND` / `PA_BBND` |
| `mask_mode` / `cmp_ratio` | 3 / 4（必传 `cmp_residual_k`） |
| `D` / `Nk` | 128 / 1 |
| `topk` | 1..2048 |
| 图能力 | `NOT_CAPTURABLE` |
| 后端 | Eager Ascend C AOT-only（`allow_jit_fallback=False`） |

必选运行时输入：`cu_seqlens_q`、`seqused_k`、`block_table`、`cmp_residual_k`、`max_seq_len_k>0`。TND 下 `indices` 形状为 `[Tq, Nk, topk]`；无效位填 `-1`。

正确性验收目标：与 ops-transformer QLIv2 在同语义输入下比较 `indices`。QLIv2 设备侧 TopK **槽位次序非分值降序**；对拍时仅在测试工具中请求 QLIv2 返回 values 并按其重排，本算子公共接口不暴露 values/`return_value` 能力。**不声称性能优势。**

源码：[公共定义](../../flashinfer/operators/sparse_indexer/quant_lightning_indexer/operator.py)、[CPU golden / sample](../../flashinfer/operators/sparse_indexer/quant_lightning_indexer/cases.py)、[AOT/runner](../../flashinfer/ascendc/sparse_indexer/quant_lightning_indexer/)、[设备实现](../../csrc/kernels/sparse_indexer/quant_lightning_indexer/)、[契约与 CPU 语义测试](../../tests/test_quant_lightning_indexer.py)。

## 与 Sparse Attention 的契约

Indexer 与 sparse attention 是两个 domain。本算子输出 token 级 TopK；`attention.sparse_flash_attention` 需要 block 级 `sparse_indices`。二者不能直接同形对接，必须显式做 token→block 转换。同 SoC 端到端组合**未验证**。

## 验证与限制

- 已验证：首期格公共 schema / infer 拒绝用例；Ascend C NativeBinding 与 production inventory 精确登记；CPU golden 边界矩阵；host OpDef/InferShape/Tiling 可 AOT 编译；GI 风格 cube（FP8 BlockMmad）+ vector quant reduce + TopK 路径在全部当前 sample 矩阵上与 CPU golden `indices` 严相等（本机 NPU，`matrix` PASS，含 N1=64 / decode+topk=2048 / PA page 轴非连续由设备内 `keyStride0` 寻址 / 越界 page 当零页）；与 QLIv2 在 `smoke_tiny` 上 value-resort 后 `indices` 严相等（原始槽位次序不同属 QLIv2 设备排放）。
- 未验证：图 capture/replay、隔离环境产品性能基线、非首期 `quant_mode`/布局、PA 页内其它轴非连续（当前要求页内轴密排，不合法 stride 由 host tiling 拒绝）、任意轴乱 stride。

性能记录：[1000 条 QLIv2 功能对比](../reports/quant-lightning-indexer-vs-qli-v2-production-1000.md)、[最差可采集 case 的 msprof 分析](../reports/quant-lightning-indexer-worst-case-msprof.md)。当前实现的 AIV TopK 标量路径明显慢于 QLIv2，不构成产品性能声明或性能优势。
