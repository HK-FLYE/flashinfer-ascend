# IndexerRegionScore

目录名为 `index_region_score`，对外沿用原算子名 `IndexerRegionScore`、设备入口
`indexer_region_score` 和 [ACLNN 两阶段接口](docs/aclnnIndexerRegionScore.md)。
`op_host/` 保留本地 arch35 tiling 实现和分页 tiling 辅助头；构建脚本从固定版本的原仓归档中获取
OpDef、InferShape、通用 Host 辅助文件及公共 ACLNN 包装。
`op_kernel/arch35/` 包含 Ascend950 设备入口和连续及分页实现。

迁移保留原有输入检查、Score 与 IDs-only 两种输出方式、mask、输出排序、连续及分页
tiling POD 和有界 workspace。架构相关 tiling 适配保留在本仓。源码来自
ops-transformer-personal 的 `experimental/attention/indexer_region_score/`，
固定版本为 `a44c2d02dd8626a656162228fff63956a07e4ec7`，保留各文件的原始版权和许可声明。

## 设备端阶段

`op_kernel/arch35/` 下的头文件按职责组织。

| 文件 | 职责 |
| --- | --- |
| `indexer_region_score_pipeline.h` | Cube/Vector 打分流水、mask 策略、D128 Q 驻留和 IDs-only 分批执行。 |
| `indexer_region_score_tiling_data.h` | host 和 device 共用的 tiling key 及连续、分页的 POD 定义。 |

`op_kernel/arch35/block/` 按 sparse indexer 的阶段组织 IRS 实现。
`RegionScoreRunner` 绑定 Key 缓存、mask、TopK 和 Q 驻留策略。普通路径与 D128
共享计算流程，D128 的 Q 驻留由模板参数控制。各阶段使用原 tiling 和 workspace。

| 阶段 | IRS 封装 | 实现 |
| --- | --- | --- |
| 调度 | `block_scheduler.hpp` | `BlockScheduler<SchedulePolicy>` 由策略定义游标和工作坐标。IRS 使用核组请求分片、K 宏块和 IDs-only 批次三种游标。 |
| Q/Key 搬入 | `key_loader.hpp` | 包含连续、分页和 D128 分页 Key 适配、L1 双缓冲及事件管理。普通及 group-fused Q、连续 Key、分页 Key 的连续区段及等间距页共用 CATLASS `CopyGmToL1GMMPTD`。宽源步长使用 `DataCopy`，离散页沿用四页展开的搬运路径。 |
| MMAD | `block_mmad.hpp` | 封装 IRS 的 MMAD 和带 ReLU 的 Fixpipe，保留分段发布 Cube 结果及通知 AIV 的顺序。 |
| head 归约 | `block_reduce_head.hpp` | 封装权重的 `DataCopyPad` 搬入、类型转换和缩放。Block 还负责运行时 head 数的 H4 累加及分数搬出，固定 H4 沿用原向量特化。 |
| TopK | `block_topk.hpp` | 统一候选选择和可选 ID 压紧的阶段编排，集中绑定 Fused、Native512 和 B8 策略。 |

`op_kernel/arch35/detail/topk_selection.hpp` 集中实现 radix、长短行选择、排序和 ID 压紧，
由 TopK Block 组合使用。`detail/request.hpp` 提供打分流水和选择阶段共用的请求位置计算。

IRS 继续直接复用 `common` 的 `AlignUp`。阶段封装位于 IRS 目录内。
完整 Score 与 IDs-only 共用请求和 K 宏块游标。批次游标统一推进全局 token 范围，
`RegionScoreRunner` 管理工作区和批次同步，Cube 与 Vector 流水管理各自的预取和事件。

本目录以独立 OPP 交付，复用固定版本原仓的 OpDef、InferShape、通用 Host 辅助实现及
手写 ACLNN 包装的 nullable Score 语义。运行下面的命令时，构建脚本从原仓工程归档中
取得这些文件，并将本地 tiling 适配、迁移的设备源码和 common 头文件放入一个新目录，再用原仓的
构建工具编译。原仓的 Torch 注册和 TorchAir converter 随归档进入临时构建目录，
供独立兼容构建使用。构建脚本在临时目录生成 `op_kernel/indexer_region_score.cpp`
和 `op_host/indexer_region_score_tiling.cpp`，分别转发到 `arch35/` 下的
设备入口和 tiling 实现。需要可用的 Ascend950 CANN SDK、CATLASS 和原仓的第三方构建依赖。
原仓和 FlashInfer 的生产登记均不受构建影响。

```bash
source /path/to/cann/set_env.sh
export FLASHINFER_CATLASS_ROOT=/path/to/catlass
bash csrc/kernels/sparse_indexer/index_region_score/build_opp.sh \
    /path/to/ops-transformer-personal /path/to/new-irs-build
```

产物在新目录的 `build_out/`。Torch 扩展可在同一构建目录用
`bash build.sh --torch_extension --ops=indexer_region_score --vendor_name=irs_port`
构建（vendor 与 OPP 保持一致）；运行时按原仓要求加载对应 OPP，原
`cann_ops_transformer` 调用名称不变。比较性能时，对原版本与迁移版本使用同一
SDK、同一卡和同一批用例，分别加载各自的 OPP，在独立进程中运行原仓的
`tests/ttk/event_bench` 和 `tests/paged_bench` 测试。

迁移验收结果见 [VALIDATION.md](VALIDATION.md)。

## 可选的 FlashInfer 前端接入

本提交交付 IRS 算子源码和独立 OPP。下面的示例适用于另行集成了
`flashinfer.sparse_indexer.indexer_region_score` 和对应 ACLNN 桥的版本。
本目录的独立调用可使用前一节的原仓 Torch 扩展或 ACLNN 两阶段接口。

```bash
source /path/to/cann/set_env.sh
source /path/to/installed-irs-vendor/bin/set_env.bash
python -B -m flashinfer.aclnn.build --output-dir build/irs-aclnn
export FLASHINFER_ACLNN_ADAPTER_PATH="$PWD/build/irs-aclnn"
```

进程启动前在 `ASCEND_CUSTOM_OPP_PATH` 中只启用一个可信的 IRS OPP；
切换版本请启动新进程。首次真机调用加载 OPP。也可提前调用
`enable_indexer_region_score()` 注册 Torch schema，注册和 Meta 推导不加载 OPP。

```python
from flashinfer.sparse_indexer import indexer_region_score

ids = indexer_region_score(query, weights, key_summary, top_k=8)
score, ids = indexer_region_score(query, weights, key_summary, top_k=8, out=score_buffer)
```

不提供 `out` 时，后端直接传空 Score 指针，保留 IDs-only 的有界工作区。
提供 `out` 或 `selected_region_ids` 时复用调用方的张量。张量的 stride 和底层
存储视图传给原 ACLNN，沿用分页与输出布局处理。设备调用走当前 NPU 流；
本阶段只验收 eager 执行，不支持 ACL graph capture。
