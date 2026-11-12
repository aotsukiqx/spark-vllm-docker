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
| decode chat 1 流 tok/s | 90.3 (mean 320 prompts) | 87.7 | 待测 |
| decode code 1 流 tok/s | 144.0 | 124.8 | 待测 |
| decode @16 用户（copies）tok/s | 441 | 427 | 待测 |
| prefill tok/s | ~10k | 5.3k | 待测 |
| max context | 1M | 1M | 800K（验证中） |
| 首次目标（本窗口） | — | — | boot+smoke+16 流基线 |

bench 口径对齐 pico：temp 0、256-512 completion tokens、多 prompt 取中位数；
bench_16.py（~/nvmodels/window-1011/）为采集器。

## 纪律

- 每项迁移独立 commit + 回滚 tag；禁止多项混在一个 A/B 里
- 报告必须保留 adverse rows（不利的重复也要入档）
- "vLLM 永远无法满足"的判定必须引用本目录的量化证据链
