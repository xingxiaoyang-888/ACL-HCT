> 历史快照：以下状态、建议和授权只代表原文写作时点，不是当前执行指令。当前结论见 [报告总览](../../../README.md)。原始字节另存于归档 ZIP。

# E2 独立确认实现与交接

研究问题：两个固定模型在新的整图随机重复和封存结构面板上，是否再现开发阶段的径向偏移、结构变化与任务变化？**当前没有真实确认结果，科学目的尚证据不足；本次交付是确认实现与可复算分析的工程准备。** 开发校准已经独立验收，不能据此宣布层级坍缩，原 E2 整体仍部分达到。

## 固定登记与实现

`configs/e2_independent_confirmation.json` 的 canonical SHA256 为 `04f625d9bb5d61402c8f8ad1440ab4de21fad7f73ce64abb112a1138080f4d12`。入口将该登记固定在源码中，拒绝修改 R、预算、面板、检查点、统计家族或其他字段后直接执行。新协议 `E2-independent-confirmation-v1`、基础随机种子 2026091702；两模型、五个 fanout 全部保留，总计 2816 次配对整图重复、448 次完整任务评价。

登记绑定完整开发索引与十份原始 summary SHA；静态审计重新按开发 MC 方差计算计划样本量，不使用效应均值或符号。两个 best768、原 baseline/training 锚点、数值门槛、全 valid 复现均保持。新入口只读取登记的确认面板；在首次结构输出计算前检查双面板清单及 child 不交叠。原开发入口保留既有默认随机流、面板和重复次数。

真实执行必须有同一固定源码的新 CUDA fixture、审核记录、登记 SHA 和开发输入。源码清单固定，不随 Python 导入顺序变化。每片创建全新目录；失败保留已经写出的记录并非零退出，不覆盖或拼接 partial。seed23-f4 内部 6600 秒，其余 3300 秒；实际时限包含输入验证耗时。外部调度仍由实验任务按协议配置。

## 独立分析

`confirmation_analysis.py` 只导入标准库、NumPy、SciPy 与纯标准库登记模块，不导入 PyTorch、模型或运行入口。逐一校验原始 summary/manifest/NPZ/数组 SHA、面板内容哈希、新随机流及配对计划哈希、数值门槛、重复次数和任务位置。从逐 rep 几何/结构标量计算 MC 统计；从保存的完整排名重新计算 micro、child-macro MRR 并与固定 full 参照相减。

40 项几何、20 项结构、10 项任务是同一个 70 项主检验家族。输出带符号均值、样本 SE、Student-t 边际 95% 区间、双侧 p、Holm 校正 p 和按 70 项 Bonferroni 的同时区间。缺失、失败和零观测方差保留 NA/不可推断标记并以 p=1 留在全家族。实际边际半宽是否达到规划目标单独报告；不根据结果延长重复。经验数值底仅作独立幅度提示，不调整 p 值，不声称严格误差界。固定两个模型、同一 valid 的限制仍适用。

分析结果索引为 JSON 对象 `{"shards": [...]}`，每行含 `seed`、`fanout`、`summary_file`（索引同目录下的 summary 副本文件名）、`summary_sha256`、`artifact_directory`（对应 entry/fanout 原始归档目录）。不要求将 NPZ 放进 Git。缺片仍生成完整 70 行，整体状态为 `incomplete_evidence`、CLI 退出 2；来源或哈希冲突直接拒绝分析。

## 复现命令与执行顺序

从固定发布包运行；`SOURCE_COMMIT` 使用实际审核过的完整版本，路径参数由实验任务填写。

```text
python -m acl_hct.e2_confirmation --config configs/e2_independent_confirmation.json --development-index reports/e2-multibudget-all-results-index.json
python -m acl_hct.e2_confirmation --config configs/e2_independent_confirmation.json --cuda-fixture --source-commit SOURCE_COMMIT --output-dir NEW_FIXTURE_DIRECTORY
python -m acl_hct.e2_confirmation --config configs/e2_independent_confirmation.json --execute --seed 11 --fanout 4 --source-commit SOURCE_COMMIT --prepared PREPARED --checkpoint BEST768 --training-release TRAINING_RELEASE --baseline-report ORIGINAL_BASELINE --approval-record APPROVAL --cuda-fixture-record CUDA_QUALITY_JSON --development-index reports/e2-multibudget-all-results-index.json --output-dir NEW_SHARD_DIRECTORY
python -m acl_hct.confirmation_analysis --config configs/e2_independent_confirmation.json --results-index CONFIRMATION_INDEX --cuda-fixture-record CUDA_QUALITY_JSON --source-commit SOURCE_COMMIT --output NEW_ANALYSIS_JSON
```

审核记录沿用 `user_authorized`、`user_message_reference`、`quality_review_passed`、`entry_criteria_frozen`、`source_commit`、`config_sha256`；`scope` 改为本次确认协议，另需 `registration_sha256`、`cuda_fixture_passed=true` 和 `cuda_fixture_artifact_sha256`。该记录应由主管审核后交接，不由诊断入口自行批准。新增依赖为 `analysis` extra 中的 SciPy >=1.10；当前实验环境已有 SciPy，无需触碰其他项目环境。

顺序：本地 CPU 与无 Git 发布包检查 → 主管固定版本审核 → 40 节点单卡 CUDA fixture → 主管 CUDA 复核及执行交接 → 十个登记科学分片 → 独立分析与科学验收。40 节点 fixture 使用自己的合成面板和 6/3 重复，不读取真实确认结构输出。

## 测试与失败记录

机器证据见 `e2-confirmation-quality-tests.json`。覆盖登记变更拒绝、仅方差计划审计、确认面板选择/不交叠、新随机流、不同任务截止位置、逐 rep 与排名独立重算、Cauchy 解析 t 分位数及手算 Holm、零方差/缺片、开发冒充确认拒绝、归档篡改、数值失败不可推断、CUDA 来源绑定、入口失败退出及不可覆盖、无模型分析导入、旧开发和 checkpoint/标签加载回归。

初次检查发现长列名列表在 NPZ 中转为数组，分析器原先使用列表 `.index`；初轮 1 failed / 19 passed。已显式规范为列表，修复后新测试 11 passed；后续加强面板内容及数值门槛完整性后，相关集成测试 2 passed。

最终无 Git 发布包检查覆盖 24 项：首轮 23 passed / 1 setup error（38.96 秒），原因是手工测试包遗漏已有 `scripts/prepare_benchmark.py`，不是运行源码错误。补入已跟踪 scripts 后，仅重跑受影响的 checkpoint/标签加载检查，1 passed / 7.88 秒；源码/测试 SHA 未变，24 项最终均通过，无跳过。保留原失败及补测记录，未重复已经通过的项目。仅有 PyTorch TypedStorage 弃用提示。测试均为 CPU 工程小图；没有新增真实确认科学输出或 CUDA/GPU 费用。
