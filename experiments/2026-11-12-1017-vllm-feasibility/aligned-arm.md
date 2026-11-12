# 对齐臂结果：官方 dspark 参数 = 免费 +116%（2026-11-12 17:05）

boot 14min（AOT 缓存命中）。对照前基线（同镜像同 checkpoint 同 800K）：

| 并发 | 基线 agg | 对齐 agg | 基线 TPOT | 对齐 TPOT |
|---:|---:|---:|---:|---:|
| 1 | 18.3 | 20.4 | 51.4ms | 46.0ms |
| 4 | 36.5 | 43.6 | 102.2ms | 83.2ms |
| 16 | 44.5 | **91.4 / 96.1** | 344ms | **160 / 157ms** |

## 对齐参数（scripts/serve-ds41-flash-dspark-tp4-rdma.sh 源）

- `enable_adaptive_verification: true` + `adaptive_verification_cost_scale: 1.0`
- `draft_sample_method: greedy`
- `num_speculative_tokens: 7`
- `disk_resident_scales: true`（engram scales 驻盘）

## 判读

- 差距 pico 441 → 我们 96：仍 4.6x，但**主要剩余归因收窄**：
  - TPOT 形态仍近线性（46→83→160ms），串行成分仍在（adaptive verification 缓解未消除）
  - 官方 TP4 脚本默认 `MAX_NUM_SEQS=4`——16 并发超出官方调优范围，剩余差距大概率在
    verify 批化/调度（迁移 #1 patch 目标不变，验收线仍 16 流 ≥150）
- 方法论注记：**任何自定义 launch 前先 diff 官方 scripts/serve-*（本仓库 recipes 亦缺这组参数）**
  ——spark-vllm-docker 的 dsv4.1 recipes 应同步这组参数（后续 commit）

数据：aligned-16.log；launch：~/nvmodels/window-1011/launch-1017-aligned.sh
