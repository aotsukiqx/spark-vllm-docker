# 迁移 #1：DSpark 并发扩展断裂 — 归因分析

日期：2026-11-12。状态：**证据链闭合，源码定位待深挖**。

## 现象

| 并发 | 聚合 tok/s | 每流 TPOT p50 |
|---:|---:|---:|
| 1 | 18.3（temp0 首测 42.8，口径不同） | 51.4 ms |
| 2 | 26.1 | 72.3 ms |
| 4 | 36.5 | 102.2 ms |
| 8 | 40.6 | 185.2 ms |
| 16 | 44.5（temp1.0）/ **44.3（temp0 复测）** | 344 / **340.6 ms** |

TPOT ≈ n × 21ms 线性劣化；吞吐饱和 ~44 tok/s。对照 pico：16 流 441（每步 59ms 处理全部 16 请求）。

## 归因（证据排除法）

1. ❌ 采样/accept：temp 0 vs 1.0 的 n=16 完全一致（44.3/340.6 vs 44.5/344）
2. ❌ 单流路径：单步 51ms 与 pico 59ms 同量级（且 pico 每步产 ~5 tok，我们 <1——单流 accept 低是独立次要项）
3. ✅ **并发无 batch 收益**：每步成本 ∝ 并发数 = 请求间步进串行化。
   嫌疑层：DSpark（DFlash family）verify 路径或 v1 调度器的 spec-decode 批处理。

## 对照 pico 的设计（迁移目标形态）

pico `engine/dspark`：C++ 层单步处理整批请求的 draft+verify（每步 59ms @16 流，批内共享 CUDA graph 与通信）。vLLM fork 的 `vllm/v1/spec_decode/dflash.py` proposer 有 batch 维度（`batch_size * num_query_per_req`），断裂点需在 verify/调度侧定位（`vllm/v1/core/sched/` + `v1/worker/` 的 spec 路径）。

## 源码定位线索（2026-11-12，下个工作单元起点）

- `vllm/v1/worker/gpu_model_runner.py:696`：drafter 构建仅按 `use_dflash()` 分 `DFlashProposer`；
  **gmr.py 无 use_dspark 独立分支**（grep 零命中）——DSpark 经 eagle/dflash 家族路径进入
- `vllm/config/speculative.py:2039-2099`：`use_eagle()` 含 dspark；dflash 与 dspark 在
  KV 语义上分家（`use_eagle_preserves_target_kv_cache`），但 drafter 层共享
- `vllm/v1/spec_decode/dflash.py`：proposer 有 batch 维度（`batch_size * num_query_per_req`）
- 追踪链：`gpu_model_runner` step → drafter.propose（batch?）→ target forward（verify batch?）
  → `speculative.py: num_speculative_tokens_per_batch_size`（动态 spec 配置，adaptive 窗口）
  ——重点查 verify 的 target forward 是否 per-request 循环
## 定位收窄（2026-11-12 终轮）

drafter 分支链（gmr.py:650-697）确认 **DSpark 无独立 drafter**——draft+verify 内嵌
target 单趟 forward（DFlash/DSpark 的 bonus-token + K masked queries 设计，
`dflash.py:40 max_query_tokens = max_batch_size * (1+K)`）。串行化嫌疑最终收窄到：
**B12X attention 层对 (n_req × (K+1)) 展开查询的处理**——B12X 源码在
`local-inference-lab/b12x` 仓库（非 vllm fork），patch #1 从该 repo 的

### b12x repo 侦察（2026-11-12 末轮）

`local-inference-lab/b12x` 的 `b12x/attention/` 模块清单：`sparse_mla` / `compressed_sparse_mla`
（CSF）/ `paged_decode` / `dsa_indexer` / `dense_mla` / `varlen` / `paged`。
**DSpark verify 的入口嫌疑：`sparse_mla`（1017 的 decode 走 sparse-MLA，即 topk256 mod
所修路径）与 `paged_decode`（decode 侧 paged attention）**。patch #1 新会话从这两个
模块的 forward 读起，对照 vllm fork 侧 `dflash.py:build_model_inputs_first_pass`

## 差距最终解剖（2026-11-12，真 temp0 确认）

真 temperature=0 复测：单流 20.5（vs temp1.0 20.4）、16 流 95（vs 96）——**温度彻底无关**。
每步 45ms（快于 pico 59ms）× accept ~1.4 tok（vs pico ~5.2）——**单流 4.4x 差距 = accept 率**。
pico 的 accept≈满值源于 engram 表（16M 词表 n-gram 预测器）作为 draft 核心。
**迁移 #1 双头目标**：draft 路径的 engram 利用（b12x sparse_mla gather 是否接入 dspark

## 实测修正（2026-11-12，/metrics 直读）

`drafts=12688, draft_tokens=21349, accepted=16319` → **draft 深度 1.68/7（24%）**、
accept 率 76.5%（健康）、每步产出 ≈2.3 tok @45ms ≈ 22 tok/s 与实测吻合。
**靶心改写：单流差距的头号原因是 draft 深度被压到 1.68**（pico ~5）——嫌疑：
adaptive verification 的动态 k 压缩决策，或 draft 侧 engram 查询限速。
patch #1 第一步变为：找出压低 draft 深度的决策点（`adaptive_verification*` 实现，

## Patch #1 最终蓝图（闭环）

`num_speculative_tokens_per_batch_size` 未配置（dynamic schedule 未启用）→ 深度 1.68
非调度压缩，而是 **disk 模式 engram n-gram 查询的自然深度**。两个差距源与 pico 机制一一对应：

| 差距源 | 实测 | pico 机制（迁移蓝本） | patch 目标 |
|---|---|---|---|
| draft 深度 1.68/7 | /metrics 直读 | `engine/engram_prefetch.py`：hash-id 前向起点即算出（~1ms GPU 窗口），side-stream O_DIRECT 预取，gather 发起时已在缓存（~200µs） | 在 b12x/vllm 的 disk engram gather 前加预取层（CUDA graph 兼容：native callback 不调 CUDA API） |
| verify 串行化 | TPOT≈n×单元 | `engine/dspark` 批内共享 draft+verify 单趟 | b12x/attention/sparse_mla + paged_decode 的批维度 |

验收：单流 20→≥60（深度 1.68→≥4）；16 流 96→≥150（保底）/300+（目标）。

## 数据文件

- concurrency-sweep.log（1/2/4/8 梯度）
- temp0-n16.log（温度排除）
- bench-16.log（16 流×2 rounds 原始 per-stream）
