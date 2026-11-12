# spark-ds41f 优化移植 · 开发计划

## ⚠️ 参考项目更正与方向声明（2026-11-12，权威）

**参考 repo 是 [Picom-code/pico-dgx-ds4](https://github.com/Picom-code/pico-dgx-ds4)**
（fork of blaesus/dsgx10）——**不是** MiaAI-Lab 的 DeepSeek-v4.1-Flash-DGX-Sparks。
后者是 sglang TP4 产线方案，在 pico 的 DESIGN.md 里明确列为 *baseline*；两者勿再混淆。

| | 参考：pico-dgx-ds4 | baseline：MiaAI sglang kit | 本仓库（我们） |
|---|---|---|---|
| 引擎 | 自研 C++（dsgb10：engine/attn/comm/dspark/engram/kernels） | SGLang + adapter | vLLM fork（local-inference-lab）+ spark-vllm-docker |
| 指标 | 单流 90/144（chat/code），16 用户 441 tok/s，prefill 10k，**1M ctx** | 87.7/124.8，4.9-5.9k prefill，1M ctx | 待测 |
| engram | NVMe 按秩分片 + C 库 O_DIRECT + CUDA graph 内 IO | OFFLOAD_MODE=nvme + 预打包 | `--engram-config table_memory=disk` |
| 配置方式 | 单文件 fleet.conf（KEY=value，C++/bash 双端解析） | .env profile → boot.py 组装 | recipe YAML → launch-cluster.sh |

**拓扑方向（用户裁定 2026-11-12）：面向 200G 交换机 fabric 优化，不移植 pico 的
switchless 有向环。** 本集群生产 = 交换机双链路（rocep1s0f1 + roceP2p1s0f1，
`NCCL_IB_MERGE_NICS=1`）；计划中原 Phase 2 的 switchless ring 移植项作废，
仅保留与交换机拓扑无关的 NCCL 修复（#2393 acquire fence 等）。

## 窗口状态（2026-11-12 更新）

**Phase 0 已 PASS**（Build#1k 镜像 + Build#2 3 分钟全缓存复现 + vllm/nccl commit 一致）。
受限网络构建全链打通（codeload 三件套 + 子模块填实 + FlashKDA 预补丁，沉淀 skill）。

**1017 部署因果链**（每层修复后暴露下一层）：
1. 网络（10 轮）→ codeload 本地源三件套
2. engram 默认 `table_memory='device'`（16M 词表 embedding 全上 GPU）→ `--engram-config '{"cpu_offload":false,"table_memory":"disk"}'` + `0.78` + `262144`（参考值）
3. **instanttensor↔fork 版本耦合（两头堵）**：fork 8244d1c4（10-15，含 PR#1029 CSF block loader）需要配套 instanttensor；PyPI 0.1.9 缺 `has_csf_blocks`/`scale_expansion`，0.2.x 缺 weight_utils 的 8 项内部 API——配套版只存在于发布镜像内
4. **解法（当前）**：eugr registry 有全 nightly 序列；`nightly-20261015` = fork HEAD 同日 CI 配套镜像（vllm+instanttensor+flashinfer 组合一致），正在拉取

**当前目标**：nightly-20261015 镜像 + 双 dspark mod + engram disk + **800K ctx**（用户指定）boot 1017 → smoke → 16 流 A/B（Phase 1 host env）→ 收窗报告。

窗口回滚点：集群仓库 tag `pre-ds41f-port`（6430ce1）；生产 = sglang canary-mig
（回滚：`systemctl --user enable --now dsv41-watchdog.service` 自动拉起）。
收尾必做：watchdog enable、`.env` 的 CONTAINER_NAME 残留清理评估、窗口产物归档 `experiments/`。

集群侧工件：`~/nvmodels/window-1011/`（launch/bench/链式 pipeline + logs/ + experiments/）。

---

## 目标框架（用户裁定 2026-11-12，最高优先级）

**以参考 repo（pico-dgx-ds4）的算法与思路分析 vLLM 框架，性能迁移为第一目标。**
只有经分析确认"修改 vLLM 永远无法满足"的能力缺口，才作为问题上升级别（换引擎/混合架构），
且必须给出量化证据。迁移开发与测试遵循：**可回滚（每步独立 patch + tag）+
可量化验证（每项迁移先建基线，A/B 双臂，数据入 `experiments/<日期>-<主题>/`）**。

### pico 优化清单 → vLLm 迁移映射（分析驱动，逐项量化）

| pico 机制 | pico 实现 | vLLM 侧对应/差距 | 迁移状态 |
|---|---|---|---|
| Engram NVMe 分片 + 侧流预取 | 按秩分片 + O_DIRECT + hash-id 即预取（覆盖 1ms GPU 窗口） | `--engram-config table_memory=disk`（内建）；预取/分片粒度未知 | **boot 中**（nightly 镜像 800K）——先量化 gather 延迟与 IO 等待再谈优化 |
| DSpark draft-verify | C++ 自实现，k=5，59ms/步 | vLLM fork 已有 dspark（topk256/d2t mod 已就位） | 待量化：accept rate、tokens/step vs pico 的数字 |
| fast-load / 权重预切分 | .dsw 按秩 mmap | instanttensor（加载路径已通） | 待量化：boot 到 serve 时间 vs pico |
| prefill 10k tok/s | 引擎级 | vLLM fork B12X 栈 | **待建基线**（pico 10k / sglang 5.3k / 我们 ?） |
| 会话 spill（KV→NVMe） | SPILL 目录 | vLLM 无直接对应（KV 池内 pinned） | 观察项：800K/1M 时 KV 池压力 |
| 交换机 fabric 优化 | （pico 是 ring，不适用） | 生产双链路 MERGE_NICS 已配 | NCCL #2393 fence 保留候选 |

每项迁移的交付物：分析笔记（差距归因）→ patch/mod → A/B 数据 → go/no-go → 回滚点 tag。

## 前提结论（证据，2026-11-10 两侧仓库实测）

| 事实 | 证据 |
|---|---|
| 本仓库 vLLM pin 浮动 `VLLM_REF=main`，无 lock | `Dockerfile:303` |
| 本仓库 NCCL clone 未 pin（浮动 HEAD） | `Dockerfile:109` |
| patch series 机制已有，但 case 只认 vllm/flashinfer | `docker/apply_patch_series.sh:7-8` |
| launcher 无 cpuset/malloc/usage-stats 注入 | `launch-cluster.sh:1557,1345`（grep 全库为空） |
| recipe env → run-recipe.py:1085 `container_env` → `-e` 透传链完整 | `run-recipe.py:1085,1440` |
| DSpark adaptive verification 已在主线+qwen3.8 patch | `docker/patches/qwen3.8/vllm/03-pr59740-60226.patch:556`、`recipes/4x-spark-cluster/deepseek-v4.1-flash.yaml:53` |
| `0019` fill_defaults 补丁零 fork 耦合（只改 `vllm/env_override.py` + 测试，80 行） | patch 头部 diffstat |
| `0184` 是大重构（7 文件，-738/+109，重写 adaptive_verification.py） | patch diffstat |
| `0018` SP 阈值依赖 `b12x_roce_all_reduce.py` 暴露 `all_reduce_max_bytes` | patch:142-145；本仓库同款 RoCE allreduce 栈 |

## 已完成（先于本计划）

| 项 | 提交 |
|---|---|
| `fix-dspark-dsv4-d2t` mod（vLLM PR #47429） | 0f89100 |
| `add-dsv4-topk256` mod（FlashInfer PR #3817 Python 侧） | add0442 |
| mods 永久测试套件 + CI 接线 + README | b03e2d6 |
| `deepseek-v4-flash-dspark` recipe（两 mod 消费者） | 7d5bed2 |

另有既有基础设施（非本计划产出，但 Phase 0 依赖）：
wheel 构建已写 `.vllm-commit`/`.flashinfer-commit` provenance 标记并进入
`build-metadata.yaml`（`build-and-copy.sh:1467-1475`），`image_identity.py` 已挂。

## Phase 0 — 固定基线（前置）

无 pin 则所有 A/B 数字漂移；spark-ds41f 的结论建立在 base commit + patch head + tree hash 三锁之上。

1. `VLLM_REF` 显式 pin：**已确认无需开发**——`--vllm-ref <full-sha>` 即完整 pin 语义
   （本地路径 `build-and-copy.sh:215-228` resolve 到 commit；远程路径 Dockerfile `git checkout --detach $VLLM_REF`；
   `.vllm-commit` provenance 已写入 `build-metadata.yaml`）。默认浮动行为不变。
2. NCCL pin：**已落地**——`Dockerfile` `ARG NCCL_REF=ad2d192c…`（2026-11-11 上游 HEAD），
   clone 后 `git checkout --detach $NCCL_REF`；`/workspace/nccl-commit.txt` 记录实际 commit；
   `build-and-copy.sh` 新增 `--nccl-ref` 选项透传 build-arg，`build-metadata.yaml` 新增 `nccl_ref` 字段。
3. 验收：两次构建同参数 → vllm/nccl commit 一致。**待真实集群构建验证**（本地无 docker）；
   验证点：`build-metadata.yaml` 的 `vllm_commit`/`nccl_ref` + 容器内 `/workspace/nccl-commit.txt`。

状态：**代码完成，集群构建验收待做**。

## Phase 1 — 零风险 host 层（可并行）

全部落在 launcher 已有注入点，改动量 <30 行：

| 项 | 改动点 | 方式 |
| `--cpuset-cpus=5-9,15-19` | `launch-cluster.sh:1557` `docker_args_common` | 加 `VLLM_SPARK_CPUSET` 环境变量，默认空=不 pin（保守），recipe/`.env` 可开 |
| `MALLOC_ARENA_MAX=2` `MALLOC_MMAP_THRESHOLD_=1048576` | `launch-cluster.sh:1345` `get_env_flags()` 或 recipe `env:` 段 | 直接加进 dsv4.1 recipe env，recipe → `container_env` 链路现成 |
| `VLLM_NO_USAGE_STATS=1` `DO_NOT_TRACK=1` | 同上 + `Dockerfile` ENV | 同上 |
| host `vm.watermark_boost_factor=0` | `setup-cluster.py`（有可逆 host 修改机制） | 可逆 sysctl 项，doctor 检查 |

验收：`test_launch_cluster_earlyoom.sh` 模式加 launcher 测试断言 flag 生成；A/B 一次 16 流 bench（预期 decode ±1% 内、boot 间方差收窄——spark-ds41f 数据：step -0.86%，SD 0.61%→0.56%）。

进展（2026-11-11）：
- malloc/usage-stats 三项已落入两个 dsv4.1 recipe `env:` 段（recipe → `container_env` → `-e` 链路现成）。
- cpuset 无需开发：`launch-cluster.sh:15` 已有 `VLLM_SPARK_EXTRA_DOCKER_ARGS` 注入
  （`VLLM_SPARK_EXTRA_DOCKER_ARGS="--cpuset-cpus=5-9,15-19" ./launch-cluster.sh …`），此前无文档。
- 待做：host `vm.watermark_boost_factor=0`（defer：需集群窗口 A/B 验证后再决定是否扩
  doctor 检查面，避免为未验证 sysctl 固化代码）、16 流 A/B bench（需集群）。
- 已补：launcher 回归测试 `VLLM_SPARK_EXTRA_DOCKER_ARGS` 注入断言
  （`tests/test_launch_cluster_image_sync.sh`，提交 1206895，顺带修复 bash3.2 空数组
  pre-existing 阻塞——套件从死在第 2 测试提升到过 9/19）。

状态：**代码项全部落地（env 四项 + cpuset 通道 + 回归测试），集群侧 A/B 与 sysctl 决策待做**。

### Gate 评审（2026-11-11，PlanGate 子代理裁决摘录）

两个改变排期的技术事实：
1. dsv4.1 产线（`--exp-b12x`）vLLM 无 commit pin：`EXP_B12X_VLLM_REF` 浮动分支硬编码，
   且与 `--vllm-ref` 互斥（`build-and-copy.sh:37,885`）。Phase 0 的"`--vllm-ref` 已够"仅对默认 main 道成立。
2. Phase 3 补丁承载改判：走 runner `--installed` 安装树补丁层（`Dockerfile:876-890` 模式，
   只重建 runner、复用缓存 wheel），不走 regular series（series 要重编 b12x wheel 且对非 main ref 跳过）。

裁决要点：Phase 0 基线构建先行（不并入 Phase 2，需未打补丁控制镜像）；Phase 1 立即收口 +
16 流 A/B（免构建）；Phase 2 soak 提前编排但与 bench 互斥；0002-0004 switchless ring 移出
（改浮点求和顺序污染判据）；Phase 4 砍尾版 defer 至 Phase 3 收口（可复现性收益非吞吐）；
Phase 5 gate 改写为：①dsv41 全 sha pin 有构建证据 ②Phase 2 已裁决 ③接口探针
（`all_reduce_max_bytes` + 205 阈值 TP4 重算）通过；Phase 4 从 gate 剔除。

已按裁决落地（本日）：
- `--b12x-vllm-ref <sha>` 覆盖（`build-and-copy.sh`）+ 3x/4x dsv4.1 recipe build_args 全 sha pin
  （`8244d1c4`，dev/karmic-kraken 2026-11-11 HEAD）
- `build-metadata.yaml` 的 `nccl_ref` 字段默认值现从 Dockerfile ARG 解析（不再是字面 "default"）
- `--nccl-ref`/`--b12x-vllm-ref`/默认 pin 三个回归测试（`tests/test_build_and_copy.sh`），
  顺带修复 mock docker 两处 bash3.2 空数组 unbound-variable（套件从死在第 5 测试解锁到 27+ 连过）


## Phase 2 — NCCL patch（3-5 天）

1. `apply_patch_series.sh:7-8` case 加 `nccl:https://github.com/NVIDIA/nccl`，`Dockerfile:109` clone 后插 series 应用步骤（模式照抄 flashinfer lane）。
2. 第一片：**nccl#2393** acquire fence（AArch64 IB CTS 重排 → RoCE hang）。单点 `dmb ishld`，上游 open PR，修 #1983。与历史 NCCL RoCE hang 症状同源。
3. 第二片（可选）：0002-0004 switchless ring 双向/均衡——**注意改变浮点求和顺序**，bit-exact 基线会变，需单独 A/B 决策。
4. 验收：容器内 `libnccl.so.2` 版本串含补丁标记；4 节点 72h soak 无 hang（对照历史 hang 频率）。

状态：**未开工**。

## Phase 3 — vLLM host 开销补丁移植（按序）

走 `docker/patches/qwen3.8/` 同款 regular series 机制（或新开 `docker/patches/dsv41/vllm/series`）：

1. **`0019` fill_defaults**（首选）：2 文件 80 行，torch 层修复，vllm main 直接 `git apply` 大概率干净；B12X 64 参数 MoE launch 873µs→48µs host 开销。验收：其自带 `tests/test_custom_op_fill_defaults.py` 移植通过 + prefill TTFT A/B。
   - 注意区分：`Dockerfile:881` 已有的 `patch_torch_schema_enumeration.py` 是另一个 fill_defaults 相关修复（schema 参数枚举），不等于本项的 `0019` custom op 移植。
2. **`0020` dead-row warmup**：1 文件，首请求 TTFT 568→210ms。依赖 DEAD_ROWS Triton 变体存在于主线（构建时验证，缺失则降级跳过）。
3. **`0039` dummy-context blocks**：2 文件正确性修复，startup profiling 跨 KV group 串记录。低成本高确定性。
4. 先查主线：fresh-draws（vllm#61161）等是否已被 main 收录（`VLLM_REF=main` 浮动，构建时 log 实际 commit 对照）。

状态：**未开工**。

## Phase 4 — DSpark 调度（1-2 周，独立可裁）

1. **`0184` 固定 row-cost 预算**（`adaptive_verification_row_cost` 常量替代 boot 计时）：核心价值是可复现性（draft 数不再随 boot 时序漂移）。7 文件重构，需把其 `adaptive_verification.py` 改写对齐主线版本——主线已有该模块（qwen3.8 patch 证明），但结构可能已变，工作量在 rebase。先小：只取 `speculative.py` 的 row_cost 常量 + argmax 公式，砍掉曲线文件删除部分。
2. dead verification rows（survival 阈值 → MoE 跳过 expert 读取）：在 1 站稳后评估。
3. 验收：同一镜像 3 次冷 boot，draft 数分布一致（0184 前 vs 后）；吞吐 A/B。

状态：**未开工**。

## Phase 5 — Prefill 序列并行（大项，gate 后启动）

`0014-0018`：205 token 阈值（TP3/2MiB/5120 bf16 推导）以上 all-reduce→reduce-scatter。spark-ds41f 实测 prefill +8~12%。前置条件：确认主线 `b12x_roce_all_reduce.py` 接口可暴露 `all_reduce_max_bytes`（0018 patch:142）。**Phase 1-4 收益落地后才开**，避免多变量混淆归因。

状态：**未开工，gate 待评审**。

## 工程纪律（贯穿，借 spark-ds41f）

- 每优化一目录：`experiments/<日期>-<主题>/` 含方法、原始数据、裁决（go/no-go + 回滚点）
- 所有 A/B 同一镜像基线、双 repeat、报告保留 adverse rows
- Bench 工具：本仓库现用 vllm 自带 bench；若要对齐 stall/Poisson 维度再评估引入 servebench 子集
- 每个 phase 独立 PR，逐个 gate，随时可停在任一 phase（Phase 1/2/3 互不依赖）
- 回传 upstream 只在集群验证通过后（fork → eugr PR 链路已就绪）

## 排期与分工建议

| 周 | 内容 | 角色 |
|---|---|---|
| W1 | Phase 0 + Phase 1（并行） | infra 1 人 + bench 1 人 |
| W1-2 | Phase 2 NCCL | infra |
| W2-3 | Phase 3 补丁移植 | vllm 工程 1 人 |
| W3-4 | Phase 4 DSpark | vllm 工程 + bench |
| W5+ | Phase 5 gate 评审后启动 | 2 人 |

总计 4-5 周到 Phase 4 收口。Phase 1+3.1（cpuset + fill_defaults）是预期收益/成本比最高的两个先手。
