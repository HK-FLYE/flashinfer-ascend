# Sparse Indexer 接口与实现范围

面向集成者：本页说明 `sparse_indexer` 逻辑 domain 的当前成员、公共接口，以及与 sparse attention 的组合边界。

## 当前 family

公共 domain 为 `sparse_indexer`。当前登记一个成员：

| logical op | public API | 状态 |
| --- | --- | --- |
| `sparse_indexer.quant_lightning_indexer` | `flashinfer.quant_lightning_indexer` | Ascend950 首期热点格：公共契约、AOT 声明与 CPU golden 已登记；设备 kernel 仍为桩，QLIv2 对拍未完成 |

`sparse_indexer.lightning_indexer` / `general_indexer` 未在本分支创建。独立 metadata 产品算子首期不做；分核调度内嵌 tiling/kernel。

## QuantLightningIndexer（首期热点格）

公共入口为 `flashinfer.quant_lightning_indexer`，返回 `(indices, values)`。`return_value=0` 时 `values` 为空形状 `bfloat16` Tensor。计算语义对齐 DeepSeek V4 DSA 生产热点：

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

正确性验收目标：与 ops-transformer QLIv2 在同语义输入下比较 `indices`（默认严相等）。**真机 kernel 与对拍未验证，不声称功能完成或性能优势。**

源码：[公共定义](../../flashinfer/operators/sparse_indexer/quant_lightning_indexer/operator.py)、[CPU golden / sample](../../flashinfer/operators/sparse_indexer/quant_lightning_indexer/cases.py)、[AOT/runner](../../flashinfer/ascendc/sparse_indexer/quant_lightning_indexer/)、[设备桩](../../csrc/kernels/sparse_indexer/quant_lightning_indexer/)、[契约与 CPU 语义测试](../../tests/test_quant_lightning_indexer.py)。

## 与 Sparse Attention 的契约

Indexer 与 sparse attention 是两个 domain。本算子输出 token 级 TopK；`attention.sparse_flash_attention` 需要 block 级 `sparse_indices`。二者不能直接同形对接，必须显式做 token→block 转换。同 SoC 端到端组合**未验证**。

## 验证与限制

- 已验证：首期格公共 schema / infer 拒绝用例；Ascend C NativeBinding 与 production inventory 精确登记；CPU golden（FP8 反量化 + ReLU + 加权 + mask_mode=3 + stable TopK）在首期 sample 矩阵上的形状/填充/可见窗边界。
- 未验证：设备 kernel 数值、相对 QLIv2 的 indices 对拍、图 capture/replay、性能、非首期 `quant_mode`/布局。
