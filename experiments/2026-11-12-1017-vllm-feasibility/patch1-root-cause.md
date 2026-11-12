# Patch #1 收敛：单一根因 — verify 成本 ∝ rows（B12X 未批化）

日期：2026-11-12 续。基于源码 + 实测 + cost 模型三点互证。

## 机制链（全链打通）

```
adaptive_verification._assign_draft_token_budget
  budget = argmax(expected_accepted_tokens / cost)
  cost(rows) = draft_cost + base_verify + scale * (verify_cost(rows) - base_verify)
                     ↑ verify_cost(rows) 来自 boot 实测曲线（build_cost_tables_from_curves）
```

- 硬上限排除：max_total_logits = 1GB/(vocab*4) ≈ 2076 rows >> 16 reqs
- 实测深度 1.68/7 = utility 模型在**陡峭 verify 成本曲线**下的正确选择
- 陡峭的来源：verify 成本 ∝ rows → B12X attention 对 (rows) 展开查询未批化（TPOT≈n×单元 实测印证）
- 因此：**draft 深度小 与 并发串行化是同一根因**；批化修复后 adaptive 自动放深度，不改 adaptive 代码

## 调优旋钮（不修代码的中间手段）

`adaptive_verification_cost_scale < 1` 让模型低估 verify 边际成本 → 更大 budget。
风险：若 verify 真慢（串行化未修），总吞吐未必提升——A/B 臂正在测（adaptive OFF 等价最激进）。

## 验收线（不变）

批化 patch 落地后：16 流 96 → ≥150（保底）/300+；深度应自然升至 ~4-5（4-gram 表能力上限）。

## 落地入口（新会话/后续）

- b12x 仓库：`b12x/attention/sparse_mla` + `paged_decode`（verify 的 rows 维批化）
- vllm fork：`v1/worker/gpu/spec_decode/dspark/speculator.py` + `dflash2/`（query 展开与 batch 契约）
- 对照：pico `engine/dspark`（单趟批内 draft+verify）

## pico dspark 实测参照（engine/dspark/README.md）

- **固定 k=5 每步都 draft，无 adaptive**（89 节点单 CUDA graph 服务每步，PDL launch）
- 参考 run（prose temp1）：深度恒 5，命中 1/2/3/4/5/6 tok 的步数分布 47/31/12/3/2/1
  → **期望接受 ≈1.8 tok/步**（≈36%）——与我们 1.42 同量级！
- 其单流 90-144 tok/s = 步长 59ms × 接受 1.8；我们 45ms/步 × 1.42
  → **单流差距主因是每步有效接受量 × 步频的乘积差**，其中我们 adaptive 只 draft 1.86
  而 pico 恒 draft 5——verify 5 行的成本被其 CUDA graph 摊平（批内一次 forward）
- 待 no-adaptive 臂实测验证：关 adaptive 后深度/吞吐是否向 pico 形态收敛

## 决定性 A/B（2026-11-12 18:30，同 code 语料，单变量）

| 并发 | adaptive ON | adaptive OFF |
|---:|---:|---:|
| 1 | ~50-55 | 15.3 |
| 16 | **71.6 / 75.9**（TPOT 192ms） | 42.7 / 42.9 / 42.4（TPOT 340ms） |

**adaptive ON 反而 +76%**；OFF 臂 accept 率 42.9% 且 verify 成本吃满。
**结论：adaptive 非瓶颈而是功臣（cost 模型正确）。剩余差距 100% 在 verify 批化**——
batch 化降低 rows 成本后 adaptive 自动加深度，两头同受益。
**no-adaptive 路径作废**，生产配置回到 aligned（adaptive ON）。

## 第三候选（测试中）：CUDA graph capture 尺寸

官方公式 `max_cudagraph_capture_size = MAX_NUM_SEQS × (k+1)`：官方 4×8=32。
我们沿用旧值 **48**，而 16 请求 × 8 = **128** → verify 掉出 full graph 走 piecewise
（no-adaptive 臂 TPOT 340ms 的部分来源）。已建 graph128 臂：capture 128 +
官方桶列表（1..8 逐个 + 8..128 步长 4）→ A/B 中。

**结果：无收益**（graph128 16 流 74.1/76.2 vs aligned 71.6/75.9，TPOT 持平）——
vLLM 的 FULL_AND_PIECEWISE 已自适应选择小桶，capture 尺寸非瓶颈。**实验关闭**，
但 recipes 保留官方公式值（与官方脚本一致、消除隐患）。

## 配置层实验总账（三个全做完）

| 实验 | 结果 | 判定 |
|---|---|---|
| 官方参数组 | +116%（44.5→96） | ✅ 固化进 recipes（commit 已有） |
| adaptive OFF | −43% | ❌ adaptive 是功臣 |
| capture 128 官方桶 | ±0% | ❌ 非瓶颈 |

**结论：配置层收益已穷尽。剩余 4.6x 只在 verify 批化代码层。**
