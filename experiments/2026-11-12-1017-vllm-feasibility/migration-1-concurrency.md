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

## 下一步（源码定位 → patch 或上游证据）

1. 定位 verify 侧批处理：`vllm/v1/` 下 DSpark/DFlash verify 的 batch 维度执行 vs 逐请求循环
2. 若 fork 逐请求循环 → mod/patch 批化（验收线：16 流聚合 44 → ≥150 tok/s，TPOT ≤100ms）
3. 若架构性限制（spec verify 与 continuous batching 冲突）→ 按"可量化上游证据"升级

## 数据文件

- concurrency-sweep.log（1/2/4/8 梯度）
- temp0-n16.log（温度排除）
- bench-16.log（16 流×2 rounds 原始 per-stream）
