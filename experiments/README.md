# Experiments — 可回滚、可量化的迁移验证

每项性能迁移一个目录：`experiments/<YYYY-MM-DD>-<主题>/`，内容：

```
notes.md      分析笔记：pico 机制 → vLLM 差距归因（先分析后动手）
patch/        本项的 patch/mod（对应 repo commit，可独立回滚）
baseline.json 对照基线（pico / sglang / 本仓库三方，注明镜像与参数指纹）
arm-{a,b}/    A/B 双臂数据（bench 原始输出 + 摘要）
verdict.md    go/no-go + 回滚点 tag + adverse rows 保留
```

## 基线对照表（持续更新，来源见各目录）

| 指标（4x DGX Spark, TP4） | pico（C++ 引擎） | sglang baseline | 本仓库 vLLM |
|---|---:|---:|---:|
| decode chat 1 流 tok/s | 90.3 (mean 320 prompts) | 87.7 | **42.8**（median×3，256tok，含思维链，2026-11-12 首测） |
| decode code 1 流 tok/s | 144.0 | 124.8 | 待测 |
| decode @16 用户 tok/s | 441 | 427 | **44.5/44.8**（2 rounds；TPOT p50 344ms、TTFT ~1s；**零并发扩展**：聚合≈单流） |
| prefill tok/s | ~10k | 5.3k | 待测 |
| max context | 1M | 1M | 800K（已 boot+smoke ✓，safetensors+engram disk） |
| boot 到 serve | — | — | ~35min（safetensors 476G + AOT 1721 kernels） |

首测口径注意：vLLM 首测含 reasoning tokens（thinking 默认开）；pico/sglang 数字为
chat prose 口径——下轮 A/B 加 `thinking off` 对齐后重测单流。
bench_16.py（~/nvmodels/window-1011/）为采集器。

## 纪律

- 每项迁移独立 commit + 回滚 tag；禁止多项混在一个 A/B 里
- 报告必须保留 adverse rows（不利的重复也要入档）
- "vLLM 永远无法满足"的判定必须引用本目录的量化证据链
