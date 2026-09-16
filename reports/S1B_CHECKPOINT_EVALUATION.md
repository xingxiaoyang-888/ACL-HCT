# S1B 只读完整 valid 评估入口

`python -m acl_hct.evaluate_checkpoint` 对已存在的协议 B checkpoint 做完整 valid / 全实体父候选评估。它复用 `load_prepared`、两层 full-neighborhood 编码和 `filtered_parent_ranks`，不训练、不 resume、不保存权重、不创建或覆盖 `best.pt`，也不改变既有选择结果。训练器的严格 resume 条件未改动。

当前用途是对 pilot `last.pt` 测量完整 valid 成本并报告开发指标。即使评估完整结束，也不能事后把只有 probe 的 pilot 宣称为按完整 valid 选出的最佳模型。输出始终包含 `selection_performed=false`；checkpoint 原有 selection 字段只作历史记录，不能据此断言 last 权重等于此前 best。

## 来源与输入校验

- 调用者必须提供外部已记录的 checkpoint SHA256、完整训练 commit 和对应训练 release 目录。先对读取的 checkpoint 文件做 SHA 校验，再用 `weights_only=True` 在 CPU 加载。
- checkpoint 内训练 commit 必须相符；所有已记录源码的规范 LF 哈希必须与 `training-release/src/acl_hct/` 文件相符，且必须包含训练入口及七个基础依赖。路径必须留在该目录内。历史源码只读取，不导入。
- 外部 checkpoint SHA 是信任锚。release 字节匹配证明与 checkpoint 内记录一致，不冒充独立 Git 来源认证。训练 source/hash 与当前评估 source/hash 分开输出；无 Git 的评估归档必须提供 `--source-commit`。
- prepared manifest 和完整 valid 顺序/hash 必须与 checkpoint 相符；模型按保存配置重建，FP32 有限权重严格匹配 state dict。失败 checkpoint 被拒绝。
- 只读 loader 从不打开 test/truth 标签文件。编码使用完整 G_obs，不屏蔽验证边之外的训练边；两层均 full。ranking 使用全部 valid 查询及全部实体 ID，按 child 排除自身和其他 valid 真父节点，沿用精确同分平均秩。
- 完成后再次核对 checkpoint SHA；输出必须是全新的 `.json` 路径，已有报告不会覆盖。

## 有界执行和报告

`--max-seconds` 为 (0,3600]，默认 600；线程 1–8，候选 chunk 1–8192。deadline 在来源/输入验证、模型构造、完整编码和每个 ranking child 之间检查；单个阶段或 child 可以超过剩余时间，外层调度 wall limit 才是硬上限。时间耗尽会输出 `incomplete_time_limit`，完整 valid MRR 字段保持 null，已完成部分如有则只保留在标为 incomplete 的 ranking 对象中。

报告记录校验、模型构造、full encoding、ranking 和总耗时，完整 valid 查询数/child 数、实体数、逐查询秩、micro/child-macro MRR、Hits、运行版本及 PyTorch CUDA allocated/reserved 峰值。CUDA 必须在单卡 Slurm allocation 内执行，保留绑定，关闭 TF32，无混合精度；入口不提交作业。CPU 完成不等于 GPU 成本已验证。

已发布历史 pilot 的已审核来源为训练 commit `fce017259a6bdc513f21b31304f096239e357a75`，last SHA256 为 `038fd4bc72c71dd544da48453e24349dd9018035b662cf2e5c2799cb9615638a`。真实 prepared 的完整 valid 应为 8456 条、8210 个 child，实体 82115；以输入哈希核验结果为准，不在代码中把这些数字硬编码为普遍数据合同。

以下命令中的路径和 `FULL_EVALUATION_RELEASE_SHA` 由实验任务替换；必须在已批准的固定评估 release 和单卡 allocation 内执行。训练 release 仍指向原始 fce0172，不能用优化后评估 release 冒充训练源码。

```bash
PYTHONPATH=src python -m acl_hct.evaluate_checkpoint \
  --prepared /path/to/wordnet-b-v1 \
  --checkpoint /path/to/original-pilot/last.pt \
  --training-release /path/to/fce0172-release \
  --expected-checkpoint-sha256 038fd4bc72c71dd544da48453e24349dd9018035b662cf2e5c2799cb9615638a \
  --expected-training-commit fce017259a6bdc513f21b31304f096239e357a75 \
  --source-commit FULL_EVALUATION_RELEASE_SHA \
  --device cuda --threads 2 --candidate-chunk 4096 --max-seconds 600 \
  --output /path/to/new-full-valid-report.json
```

## 已完成验证

CPU 40 实体 fixture：`python -m pytest tests/test_checkpoint_evaluation.py -q`，10 passed / 30.79 s。验证完整 valid 多于一条 probe、全部候选、test/truth 不读取、禁止 optimizer/save 调用、输入和 checkpoint 目录字节不变、无 best 文件、artifact/commit/source/valid/manifest 不匹配拒绝、编码前时间耗尽、ranking 时间耗尽、未分配 CUDA 拒绝，以及实际无 Git 归档 CLI 区分训练与评估来源并拒绝覆盖输出。仅出现旧 PyTorch TypedStorage 弃用警告。

同轮原训练器五项回归通过（包括完整 valid 选择和精确 resume）；它们未因新增只读入口而改动。本提交未运行真实完整 valid 或 GPU 作业，不提供该成本或模型能力结论。效率补丁与历史 pilot 证据分别见 [S1B_EFFICIENCY.md](S1B_EFFICIENCY.md) 和 [S1B_PILOT.md](S1B_PILOT.md)。
