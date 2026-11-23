> 公开接口保留唯一必选key_summary，并显式配置cache_mode（默认0）：0为连续且禁止block_table；
> 1为分页且要求block_table、cu_seq_lens。其他枚举值仅预留，当前一律拒绝；连续mask3仍可用cu分多请求。
> 不提供独立paged_key_summary；六输入七属性新ABI需同时重建OPP和Torch扩展。核心优化与POD不变。

# aclnnIndexerRegionScore

[查看本仓设备入口](../../op_kernel/arch35/indexer_region_score.cpp)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

`aclnnIndexerRegionScore` 为每个 Query token 的每个 group 计算各可见 region 的分数，并在同一次 kernel 调用中输出 TopK 逻辑 region ID；`out`（outScore）非空时同时输出 FLOAT 分数矩阵，为空时仅输出IDs。整体计算公式如下：

$$
s_{g,t,r}=\sum_{h=0}^{H-1}\left(\mathrm{headScale} \times \mathrm{weights}[t,g,h]\right)\cdot \operatorname{ReLU}\left(\sum_{d=0}^{D-1}\mathrm{query}[t,g,h,d]\cdot K_{t,r,d}\right)
$$

其中，`g`、`t`、`h`、`r` 分别表示 group、Query token、head 和逻辑 region ID；`K` 是按 cacheMode 选择布局的 keySummary，`ReLU(x)=max(0,x)`，且 $s_{g,t,r}$ 对应 `out[g,t,r]`。`selectedRegionIds[g,t,:]` 是可见 region 中按 `s` 选择的 TopK `r`，其排序由 `sortMode` 指定。

具体计算过程如下：

1. 对该 Query/group 的每个 head，分别计算 Query 向量与各 region 的 Key summary 向量的点积，再对点积结果执行 ReLU。
2. 将每个 head 的结果乘以对应的 head 权重和 `headScale`，再对 head 求和，得到各 region 的分数。
3. 从该 Query/group 可见的 region 中选出分数最高的 TopK，并输出对应的逻辑 region ID。

排序方式由 `sortMode` 指定：

- `sortMode=0`：按分数降序排列；分数相同时，较小逻辑 region ID 优先。
- `sortMode=1`：选中的逻辑 region ID 按升序排列。

cacheMode（默认0）决定同一个keySummary的布局，当前仅支持0/1，其他值预留但拒绝：

- Query 与实际使用的连续或分页 Key summary 必须具有相同 dtype：FP8 E4M3FN、BF16 或 FP16。权重可独立选择 FP16、BF16、FP32。
- BF16/FP16 输入直接执行原生 16 位 Cube 乘法和 FP32 累加，不转换为 FP8；输出 Score 维持 FP32，TopK IDs 维持 INT32。本算子没有 Value/V 输入。
- FP8 支持 D=256；BF16/FP16 支持 D=128、256。分页容量和实际可见长度保持原有语义。

- `cacheMode=0`：连续 Key，使用 `key[R,1,D]`；Query、权重和输出分别为 `[T,G,H,D]`、`[T,G,H]`、`[G,T,R_out]`。
- `cacheMode=1`：paged Key summary，使用统一 `keySummary`、`blockTable` 和 `cuSeqLens`；Query、权重和输出分别为 `[T,G,H,D]`、`[T,G,H]`、`[G,T,R_out]`。

region ID 是从 0 开始的逻辑序号 `r`，对应 `out[g,t,r]` 的最后一维列号，由输入 Key 的逻辑顺序隐式确定：

- 连续模式下，`r` 对应 `keySummary[r,0,:]`。
- paged 模式下，每个 request 的 region ID 从 0 开始编号。令 `S=keySummary.shape[1]`，`r` 对应 `blockTable[n,r//S]` 指定物理页中的 `r%S` 位置；request `n` 由 `cuSeqLens` 确定。

例如，`S=16`、`r=35` 且 `blockTable[n,2]=100` 时，实际 Key 为 `keySummary[100,3,0,:]`，选中后输出 ID 为 35。

有效 ID 候选范围为 `0 <= r < R_eff`，其中 `R_eff` 是解析后的有效 region 容量：连续模式且 `maxRegionLen=-1` 时为 `keySummary.shape[0]`，其余情况为 `maxRegionLen`；`maskMode=3` 时，对 token `t` 还须满足 `r<v(t)`。`topK` 大于候选数时，剩余 `selectedRegionIds` 槽位以 `-1` 表示无效占位。

paged 模式下，token `t` 属于 request `n`（`cuSeqLens[n] <= t < cuSeqLens[n+1]`）时，令 `S=keySummary.shape[1]`：

$$
p=blockTable[n,\lfloor r/S\rfloor],\qquad s=r\bmod S,\qquad K_{t,r,d}=keySummary[p,s,0,d]
$$

`p=-1` 或 `p` 不在物理 page 范围内时，该 Key 按 0 处理。两种模式均直接使用 `[T,G,H]` 布局的 `weights`，接口内部不执行维度置换。

`maskMode=3` 时，`actualKvLengths[n]` 是 request `n` 的有效 KV 序列长度（token 数量，包含本批 Query 对应的 token）；decode 时等于包含当前 token 的 `seq_len`（模型侧称 1 基长度，直接传 token 数，不减 1）。令 `qLen=cuSeqLens[n+1]-cuSeqLens[n]`（未提供时单 request 的 `qLen=T`）；对属于 request `n` 的第 `j` 个 packed token，保留的历史 region 数如下。

`actualKvLengths[n]` 表示该请求当前有效的 KV token 总数，包含本批 Query 对应的 token。本批 Query 必须是该请求 KV 序列末尾的连续 token，`qLen` 表示该请求本批 Query 的 token 总数。

$$
v(t)=clamp(\lfloor(actualKvLengths[n]-qLen+j)/regionSize\rfloor,0,R_{\mathrm{eff}})
$$

这里向下取整：当前位置 `p=actualKvLengths[n]-qLen+j` 所在 region 不参与历史候选；TopK 仅选择 `r<v(t)`。模型侧可随后追加当前 region，本算子不执行该追加。`r>=v(t)` 的 score 未定义。mask0 不读取 `actualKvLengths`，调用方必须传 `nullptr`。

例如 `regionSize=8` 且 decode 的 `qLen=1`：`actualKvLengths=8` 对应当前 token 的内部零基位置 7，可见历史 region 数为 0；`actualKvLengths=9` 对应位置 8，可见历史 region 数为 1。本批 Query 必须是该 request 有效 KV 序列末尾的连续 token；KV 长度按原始 token 数表达，不要求等于压缩 Key summary 的 region 数。

分页每个 K macro 在 FP8 下最多512 region、BF16/FP16下最多256 region，包括 `S=1024`；Host 根据 `K_MACRO%S` 选择页对齐或支持页内起点的搬运策略，不改变用户接口或 tiling key。

## 函数原型

本算子使用两段式接口：先调用 `aclnnIndexerRegionScoreGetWorkspaceSize` 获取 workspace 大小和执行器，再调用 `aclnnIndexerRegionScore` 执行。

```cpp
aclnnStatus aclnnIndexerRegionScoreGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *weights,
    const aclTensor *keySummary,
    const aclTensor *actualKvLengths,
    const aclTensor *blockTable,
    const aclTensor *cuSeqLens,
    int64_t maskMode,
    double headScale,
    int64_t cacheMode,
    int64_t maxRegionLen,
    int64_t regionSize,
    int64_t topK,
    int64_t sortMode,
    const aclTensor *out,
    const aclTensor *selectedRegionIds,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

```cpp
aclnnStatus aclnnIndexerRegionScore(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

仅保留原两阶段接口，TopK-only 使用同一 `GetWorkspaceSize` 签名，显式传空 Score：

```cpp
aclnnStatus ret = aclnnIndexerRegionScoreGetWorkspaceSize(
    query, weights, keySummary, actualKvLengths, blockTable, cuSeqLens,
    maskMode, headScale, cacheMode, maxRegionLen, regionSize, topK, sortMode,
    nullptr,  // outScore: IDs-only; provide an FP32 tensor for full Score
    selectedRegionIds, &workspaceSize, &executor);
```

256MiB算法预算包含算法所需CANN库预留、一个FP32 Score池、活动批次辅助区和对齐余量；
外层Contiguous、对齐连续化、ViewCopy写回及executor其他开销另计，GetWorkspaceSize总值允许超过256MiB。
最小合法批次无法放入256MiB时返回不支持，禁止申请完整Score回退。
第二段`aclnnIndexerRegionScore`与原执行入口使用相同workspace/执行器协议。

## aclnnIndexerRegionScoreGetWorkspaceSize

**参数说明**：`out`（outScore）是输出模式的唯一判断：非空保留完整Score路径，`nullptr`启用内部固定256MiB算法workspace上限的TopK-only路径。公开接口不接收workspace预算。

所有 Tensor 格式均为 ND；`T>0`，paged 模式还要求 `P,N,C>0`。两种输出模式均支持非连续输入和输出view：Q、weights、metadata等输入通过外层自动连续化供kernel消费；已支持的页内连续、page stride及offset对齐的summary继续直接读取，其他summary布局自动连续化。计算用输入和输出由接口保证对齐，输出通过ViewCopy写回原view。256MiB仅约束Host规划的IRS算法workspace，不能用executor总空间与该值比较；外层连续化可使总GetWorkspaceSize更大。`query`、`weights`、`keySummary`、`selectedRegionIds`、`workspaceSize` 和 `executor` 始终必选；其余 Tensor 的条件见下表。

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续 Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `query` | 输入 | Query 向量。 | 不可为空；`G=query.shape[1]`，`H=query.shape[2]`；`G=1` 或 `G` 为正的 4 的倍数，`4 <= H <= 128` 且 `H % 4 == 0`，FP8 的 D=256；BF16/FP16 的 D=128 或 256；两种 cache 模式使用相同布局。 | FLOAT8_E4M3FN、BF16、FLOAT16 | ND | `[T,G,H,D]` | √ |
| `weights` | 输入 | 每个 Query/group/head 的有符号权重。 | 不可为空；两种 cache 模式均支持 FP16、BF16、FLOAT，shape 必须为 `[T,G,H]`，其中 `H` 与 `query` 一致。FP16/BF16 在 kernel 的 UB 中转换为 FP32，FLOAT 直接参与 FP32 加权归约。 | FLOAT16、BF16、FLOAT | ND | `[T,G,H]` | √ |
| `keySummary` | 输入 | 统一Key summary。 | 始终必选；cacheMode=0时[R,1,D]并禁止blockTable；cacheMode=1时[P,S,1,D]且要求blockTable、cuSeqLens；分页S支持1..1024。 | FLOAT8_E4M3FN、BF16、FLOAT16 | ND | `[R,1,D]` 或 `[P,S,1,D]` | √ |
| `actualKvLengths` | 输入 | 每个 request 的有效 KV 序列长度（token 数量，包含本批 Query 对应的 token）。 | `maskMode=3` 时必须提供 INT32 `[N]`；`maskMode=0` 时必须传 `nullptr`。kernel 结合 `cuSeqLens`（连续模式未提供时按单 request 解释）和 `regionSize` 推导每个 token 的可见历史 region 数。 | INT32 | ND | `[N]` | √ |
| `blockTable` | 输入 | 逻辑 page 到物理 page 的映射。 | cacheMode=1时必选；-1 或越界 page ID 对应零 Key。 | INT32 | ND | `[N,C]` | √ |
| `cuSeqLens` | 输入 | packed Query 的 request 累计长度。 | cacheMode=1时必选；连续模式的 `maskMode=3` 可选，未传时按单 request `[0,T]` 解释。长度为 `N+1`；调用方须保证首元素为 0、单调不减且末元素为 `T`。 | INT32 | ND | `[N+1]` | √ |
| `maskMode` | 输入 | 可见性 mask 模式。 | 仅支持 0（无 mask）和 3（可见前缀）。 | - | - | - | - |
| `headScale` | 输入 | 所有 head 权重的统一缩放系数。 | 两种 cache 模式均支持；不缩放时传 `1.0`，此时 kernel 跳过额外乘法；其他值会在归约前乘到每个 head 权重上。 | - | - | - | - |
| `cacheMode` | 输入 | Key存储枚举。 | 默认0；0连续且禁止blockTable，1分页且要求blockTable/cuSeqLens；其他值当前拒绝。 | - | - | - | - |
| `maxRegionLen` | 输入 | 本次实际计算的最大逻辑 region 数。 | paged 模式须满足 `0 <= maxRegionLen <= C*S`；连续模式中 `-1` 表示使用 `keySummary.shape[0]`，否则须满足 `0 <= maxRegionLen <= keySummary.shape[0]`。取 0 时 score 内容未定义，选择输出全部为 -1。 | - | - | - | - |
| `regionSize` | 输入 | 每个 region 压缩的原始 KV token 数。 | 正整数，默认 8；仅 `maskMode=3` 的可见范围推导使用。 | - | - | - | - |
| `topK` | 输入 | 必选的历史 region 选择数。 | 必须显式指定，范围 1～4096。 | - | - | - | - |
| `sortMode` | 输入 | TopK 输出次序。 | 0：按分数降序；1：选中逻辑 region ID 升序。分数相同时较小 ID 优先；有效候选分数须大于 `-1e29`，NaN 和无效哨兵不参与选择。 | - | - | - | - |
| `out` | 输出 | 可选region 分数矩阵（outScore）。 | `nullptr`时仅输出IDs并使用内部有界单池；非空时支持非连续 Tensor。`R_eff=(maxRegionLen==-1 ? keySummary.shape[0] : maxRegionLen)`，`R_out=out.shape[2]>=R_eff`；仅保证有效可见候选的分数，遮蔽列及额外容量尾列内容未定义。 | FLOAT | ND | `[G,T,R_out]` | √ |
| `selectedRegionIds` | 输出 | TopK 的零基逻辑 region 序号，对应 `out` 最后一维列号。 | 不可为空；候选数不足 `topK` 时，剩余槽位以 `-1` 表示无效占位。 | INT32 | ND | `[G,T,topK]` | √ |
| `workspaceSize` | 输出 | executor所需总Device workspace字节数。 | 不可为空；包含算法及外层连续化/写回等空间，可超过256MiB。 | - | - | - | - |
| `executor` | 输出 | 算子执行器。 | 不可为空。 | - | - | - | - |

**返回值**：`aclnnStatus`。`ACLNN_ERR_PARAM_NULLPTR` 表示必选参数或条件必选 Tensor 缺失；`ACLNN_ERR_PARAM_INVALID` 表示模式组合、数据类型、shape、属性不满足约束。

## aclnnIndexerRegionScore

**参数说明**：

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续 Tensor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `workspace` | 输入 | Device workspace 地址。 | `workspaceSize` 为 0 时可以为空；否则必须按第一段接口返回值分配。 | - | - | - | - |
| `workspaceSize` | 输入 | Device workspace 字节数。 | 必须与第一段接口返回值一致。 | - | - | - | - |
| `executor` | 输入 | 算子执行器。 | 必须使用同一次第一段接口返回的执行器。 | - | - | - | - |
| `stream` | 输入 | ACL 执行流。 | 不可为空。 | - | - | - | - |

**返回值**：`aclnnStatus`，具体返回码参见 ACLNN 返回码说明。

## 调用示例
示例代码如下。编译前按[算子目录说明](../../README.md)构建并加载自定义 OPP。

```cpp
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "aclnn_indexer_region_score.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void Check(int status, const char *operation)
{
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
    }
}

class Runtime {
public:
    explicit Runtime(int deviceId) : deviceId_(deviceId)
    {
        try {
            Check(aclInit(nullptr), "aclInit");
            initialized_ = true;
            Check(aclrtSetDevice(deviceId_), "aclrtSetDevice");
            deviceSet_ = true;
            Check(aclrtCreateStream(&stream), "aclrtCreateStream");
        } catch (...) {
            Cleanup();
            throw;
        }
    }
    ~Runtime() { Cleanup(); }
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;
    aclrtStream stream = nullptr;

private:
    void Cleanup() noexcept
    {
        if (stream != nullptr) {
            (void)aclrtSynchronizeStream(stream);
            (void)aclrtDestroyStream(stream);
            stream = nullptr;
        }
        if (deviceSet_) {
            (void)aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (initialized_) {
            (void)aclFinalize();
            initialized_ = false;
        }
    }
    int deviceId_;
    bool initialized_ = false;
    bool deviceSet_ = false;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) : bytes_(bytes)
    {
        if (bytes != 0) {
            Check(aclrtMalloc(&data_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        }
    }
    ~DeviceBuffer()
    {
        if (data_ != nullptr) {
            (void)aclrtFree(data_);
        }
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    void *Data() const { return data_; }
    size_t Bytes() const { return bytes_; }

private:
    void *data_ = nullptr;
    size_t bytes_;
};

class Tensor {
public:
    template <typename T>
    Tensor(const std::vector<int64_t> &shape, aclDataType type, const std::vector<T> &host)
        : buffer_(host.size() * sizeof(T))
    {
        std::vector<int64_t> strides(shape.size(), 1);
        int64_t elements = 1;
        for (size_t i = shape.size(); i > 0; --i) {
            strides[i - 1] = elements;
            elements *= shape[i - 1];
        }
        if (elements != static_cast<int64_t>(host.size())) {
            throw std::runtime_error("Tensor shape and host data size differ");
        }
        Check(aclrtMemcpy(buffer_.Data(), buffer_.Bytes(), host.data(), buffer_.Bytes(),
                          ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D");
        tensor_ = aclCreateTensor(shape.data(), shape.size(), type, strides.data(), 0,
                                 ACL_FORMAT_ND, shape.data(), shape.size(), buffer_.Data());
        if (tensor_ == nullptr) {
            throw std::runtime_error("aclCreateTensor failed");
        }
    }
    ~Tensor()
    {
        if (tensor_ != nullptr) {
            (void)aclDestroyTensor(tensor_);
        }
    }
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
    aclTensor *Get() const { return tensor_; }
    template <typename T>
    void CopyTo(std::vector<T> &host) const
    {
        if (host.size() * sizeof(T) != buffer_.Bytes()) {
            throw std::runtime_error("Output host data size differs");
        }
        Check(aclrtMemcpy(host.data(), buffer_.Bytes(), buffer_.Data(), buffer_.Bytes(),
                          ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy D2H");
    }

private:
    DeviceBuffer buffer_;
    aclTensor *tensor_ = nullptr;
};

// 第二段接口管理单次执行的 executor；调用前发生错误时由此 guard 清理。
struct Executor {
    aclOpExecutor *value = nullptr;
    aclOpExecutor *Release()
    {
        auto *executor = value;
        value = nullptr;
        return executor;
    }
    ~Executor()
    {
        if (value != nullptr) {
            (void)aclDestroyAclOpExecutor(value);
        }
    }
};

int main(int argc, char **argv)
{
    try {
        if (argc > 2) {
            throw std::runtime_error("Usage: indexer_region_score_example [deviceId]");
        }
        const int deviceId = argc == 2 ? std::stoi(argv[1]) : 0;
        Runtime runtime(deviceId);

        // E4M3FN 的原始字节：1=0x38，2=0x40，3=0x44，4=0x48。
        Tensor query({1, 1, 4, 256}, ACL_FLOAT8_E4M3FN, std::vector<uint8_t>(4 * 256, 0x38));
        Tensor weights({1, 1, 4}, ACL_FLOAT, std::vector<float>(4, 1.0f));
        std::vector<uint8_t> page(16 * 256, 0);
        const uint8_t values[] = {0x38, 0x40, 0x44, 0x48};
        for (size_t r = 0; r < 4; ++r) {
            for (size_t d = 0; d < 256; ++d) {
                page[r * 256 + d] = values[r];
            }
        }
        Tensor keySummary({1, 16, 1, 256}, ACL_FLOAT8_E4M3FN, page);
        Tensor blockTable({1, 1}, ACL_INT32, std::vector<int32_t>{0});
        Tensor cuSeqLens({2}, ACL_INT32, std::vector<int32_t>{0, 1});
        std::vector<float> scores(4, 0.0f);
        std::vector<int32_t> ids(2, -1);
        Tensor out({1, 1, 4}, ACL_FLOAT, scores);
        Tensor selectedRegionIds({1, 1, 2}, ACL_INT32, ids);

        uint64_t workspaceSize = 0;
        Executor executor;
        Check(aclnnIndexerRegionScoreGetWorkspaceSize(
                  query.Get(), weights.Get(), keySummary.Get(), nullptr,
                  blockTable.Get(), cuSeqLens.Get(),
                  0, 1.0, 1, 4, 8, 2, 1, out.Get(), selectedRegionIds.Get(),
                  &workspaceSize, &executor.value),
              "aclnnIndexerRegionScoreGetWorkspaceSize");
        DeviceBuffer workspace(workspaceSize);
        const auto launchStatus = aclnnIndexerRegionScore(
            workspace.Data(), workspaceSize, executor.Release(), runtime.stream);
        // 即使启动返回错误，也先等待可能已提交的任务，再释放设备资源。
        const auto syncStatus = aclrtSynchronizeStream(runtime.stream);
        Check(launchStatus, "aclnnIndexerRegionScore");
        Check(syncStatus, "aclrtSynchronizeStream");
        out.CopyTo(scores);
        selectedRegionIds.CopyTo(ids);

        std::cout << "scores:";
        for (float score : scores) {
            std::cout << ' ' << score;
        }
        std::cout << "\nselected_region_ids: " << ids[0] << ' ' << ids[1] << '\n';
        const float expected[] = {1024.0f, 2048.0f, 3072.0f, 4096.0f};
        for (size_t i = 0; i < scores.size(); ++i) {
            if (!std::isfinite(scores[i]) || std::fabs(scores[i] - expected[i]) > 1.0e-3f) {
                throw std::runtime_error("Score validation failed");
            }
        }
        if (ids[0] != 2 || ids[1] != 3) {
            throw std::runtime_error("Selected region ID validation failed");
        }
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
```
