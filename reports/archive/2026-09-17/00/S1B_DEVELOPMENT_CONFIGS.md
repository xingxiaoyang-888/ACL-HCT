> 历史快照：以下状态、建议和授权只代表原文写作时点，不是当前执行指令。当前结论见 [报告总览](../../../README.md)。原始字节另存于归档 ZIP。

# S1B 首批双 seed 有界开发配置

父任务于2026-09-16冻结两个独立开发 run：`configs/wordnet_b_development_seed11.json` 和 `configs/wordnet_b_development_seed23.json`。它们是重复训练种子，不是 HPO 或参数网格；各自从初始化开始，不 resume 10步 pilot。生产训练源码和原 `wordnet_b_pilot.json` 未修改。

两个配置仅 seed 不同。保留 hidden/head=128、c=1、scaled_radius=1.2、fanout=[16,16]、method=none、batch_positives=128、learning_rate=.003、threads=2、candidate_chunk=4096、max_padded_messages=32768。冻结 max_steps=1024、max_seconds=3300、evaluation=full_validation、evaluation_max_seconds=180、evaluate_every=256、save_every=50。沿用 runner 的最终评估行为；完整 valid 才能按既定 query-micro filtered all-parent MRR 选择 best，同分保留更早。时间耗尽的部分评估不能选 best，未完成过完整 valid 必须明确没有 selected checkpoint。遗留 probe_queries=16 字段在 full_validation 模式下不控制查询范围。

启动前条件是固定 release 的归档 CPU 门槛通过，且原 pilot 的只读真实完整 valid 成本测量通过。父/实验任务负责核验和 Slurm 提交；代码不自行调度。首轮两张独立 GPU 并行跑两个 seed，各 job 至多1小时、合计初轮至多2 GPUh，保护用户保留的 CVPR 两卡。不做 DDP，也不因允许更多卡而自动扩网格。每个 seed 使用独立输出目录并记录源版本、实际设备、调度占用、训练与评估时间。

按一次固定顺序同卡比较的约2.33秒/step估算，1024步约2386秒，尚需加入完整 valid、checkpoint、初始化和主机开销；这不是运行保证或统计稳健的吞吐估计。3300秒 deadline 优先；runner 在 step/child 边界检查，单次操作可能超出软时限，外层1小时为硬上限。

若完成1024步，共抽取131072条正例曝光，约为67539训练正例的1.940686倍。每个 batch 内无放回抽128个正例，但跨 batch 重新随机抽样，所以这只是曝光当量，既非1.94个无重复完整 epoch，也不保证覆盖全部训练正例或收敛。固定 seed 的 best 是否具备结构能力须另验收，不能因明晚汇报而放松统计或数值门槛。

配置检查通过：两个 JSON 均由 `TrainConfig(**config).validate()` 接受，所有未登记修改字段逐一与 pilot 相等，pilot 原始字节不变。此次仅新增配置/说明，没有启动 CPU 训练、GPU训练或更改学习率。

```bash
# 实验任务核验前置门槛后，在各自单卡 allocation 和固定 release 中运行：
PYTHONPATH=src python -m acl_hct.train --prepared /path/to/wordnet-b-v1 \
  --output /path/to/new-seed11-run --config configs/wordnet_b_development_seed11.json \
  --device cuda --source-commit FULL_RELEASE_SHA
# seed23 使用另一个全新输出目录及 wordnet_b_development_seed23.json。
```
