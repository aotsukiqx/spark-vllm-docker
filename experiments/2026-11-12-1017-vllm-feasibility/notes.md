# 1017 on vLLM — 可行性判定与完整因果链

日期：2026-11-12。结论：**可行**（safetensors 路线），800K ctx 配置 boot + smoke 通过。

## 成功组合（唯一）

| 项 | 值 |
|---|---|
| 镜像 | `eugr/spark-vllm-b12x:nightly-20261015`（vLLM `g8244d1c44` 官方 CI 同 commit 构建，instanttensor 0.2.2 fork 变体内置） |
| 加载 | `--load-format safetensors`（vLLM 原生 mmap） |
| engram | `--engram-config '{"cpu_offload":false,"table_memory":"disk"}'`（16M 词表 embedding NVMe 驻留） |
| ctx / mem | `--max-model-len 819200` / `--gpu-memory-utilization 0.78`（操作员经验区） |
| mods | fix-dspark-dsv4-d2t + add-dsv4-topk256 |
| 网络 | 交换机双链路（NCCL_IB_MERGE_NICS=1），RoCE |

## 失败因果链（15+ 轮，每层修复暴露下一层）

1. **网络**（10 轮）：GitHub git-transport 断连 → codeload tarball 三件套（vllm-src + triton_kernels + 8 ext-src + 子模块填实）
2. **DNS 劫持**：容器网桥解析到 4943999.com → `--network host`
3. **显存贴边**：0.84/0.85 三节点 NV OOM 重启 → 0.78（经验区 0.75-0.80）
4. **engram 默认 device 表**：16M 词表 embedding 全上 GPU → `table_memory='disk'`（对照 pico/sglang 的 NVMe 驻留）
5. **instanttensor↔fork 耦合（两头堵）**：fork 8244d1c4（10-15，含 PR#1029 CSF loader）vs PyPI 0.1.9（缺 has_csf_blocks）/0.2.x（缺 weight_utils 8 项内部 API）；官方 nightly 镜像的 0.2.2 是 fork 自建变体（同版本号不同内容）
6. **1017 CSF 权重 + instanttensor 全生态断**：官方 CI 组合也缺 `_determine_buffer_size`（CI 验证的是 0731 非 1017）→ **safetensors 绕开整个 instanttensor 路径**

## 性能待补（bench 运行中）

- 单流 decode（对照 pico 90 chat/144 code；sglang 87.7/124.8）
- 16 流 aggregate（对照 pico 441；sglang 427）
- prefill（对照 pico ~10k；sglang 5.3k）
- boot 时间（safetensors 476G mmap + AOT 1721 kernels：~35 分钟）

## 升级项（带完整证据链，符合"vLLM 永远无法满足才上升"门槛的候选）

- **instanttensor 的 1017/CSF index-aware loading 接口**（`_determine_buffer_size` 等 8 属性）：影响的是加载性能而非可行性；应先量化 safetensors vs instanttensor 的 boot/首请求差距再决定是否推上游
