# E2 原模型能力对照：实现与 CPU 验证

研究问题：相同原始文本、监督数据与有界训练预算下，现有双曲 GNN 是否优于无图文本 MLP 的完整 valid 父检索？**当前没有新 MLP 科学训练结果，模型竞争力仍证据不足；本次工程准备目的达到。** 这补齐原 E2 的必要能力参照，不更改已经登记的确认，不代表 E5 或 SOTA 比较。

## 实现语义核对

原 `ranking.py` 在评分缓存中强制执行 Lorentz 原点 log，不能直接接收欧氏 MLP 输出。因此新增 `text_capacity.py`，直接缓存欧氏嵌入的关系头分量，保留相同的全实体候选、排除 self/其他真父、目标重新纳入、精确计算同分的平均排名、micro 与 child-macro 定义。缓存与逐对评分分别核对，且参数更新使缓存失效；没有修改原排名 API。

批次使用原 `seed+1` 独立 CPU `randperm` 流，每步最多 128 个既定正例组；每组保留原固定四负例，BCE 不重新采负例。原 query-mask 和邻居采样用于 GNN；MLP 无图输入，不执行它们。测试直接捕获原训练器批次及本训练器查询，确认组序列、完整正负查询及最终批次 RNG 状态一致。

原训练器可能在 partial valid 后继续；本对照按协议在该情况立即停止，保留 partial 排名而不选模。每 256 步只做一次原完整 valid，严格更高的 micro MRR 才替换 best；相同值保留首次赢家。没有额外探针选模、最终补跑验证、resume 或超参搜索。

## 固定登记、模型与来源

新增 `configs/e2_model_capacity_control.json`，canonical SHA256 为 `c40ad1d32bb338047f307749bf0ef023c204ce1cf39550e003d77ce665df7634`；科学 CLI 在源码中固定此哈希，拒绝配置修改。保留 seed11/23、原 prepared manifest 与 valid hash、两个原 GNN baseline raw SHA / best768 / 训练来源锚点。执行前核对原 baseline 的四次完整排名、metric 复算、查询覆盖、首次 best 和相同数据。

MLP 为原 128 维特征 → 128 Linear+ReLU → 128 Linear+ReLU；关系头 `[parent, child, parent-child]` → 128 Linear+ReLU → 1 logit。Linear 均带 bias；没有图聚合、dropout 或归一化层。原 train-only 文本特征不重新拟合。该定义含 82,433 个参数，原 GNN 在这组维度下也是 82,433 个参数；相同参数数不等于相同表达能力、FLOPs 或收敛程度，不称严格单因素几何消融。

训练时只编码本批查询出现的独立实体。由于没有跨实体算子或随机层，这与全实体编码的同一查询损失一致；FP64 工程测试核对了损失及全部参数梯度。验证仍编码所有实体，不缩减候选。

全部源码变化仅为两个新增模块；既有 `49609014610fe86dbf6b19fe8a1fe94df7ecd019` 确认源码及配置与 Git blob 逐项核对为未改动。能力对照使用自己的发布目录与来源记录，不将其模块上传覆盖确认固定目录。数据加载复用原严格 train+valid loader；读取 observed_graph 仅用于输入合约检查，不传给 MLP。测试拦截标签读取并核对 prepared 文件在运行前后字节不变，没有 test/truth 读取。

## 有界执行与可复查输出

科学运行固定 Adam lr0.003、1024 步、每 256 步全 valid、每 50 步 last 保存；内部总预算 3300 秒，含执行输入验证，单次 valid 最多 180 秒。每 seed 单 L40/2CPU/16GiB、外部 60 分钟，与确认共享最多四个 ACL 并发上限，保留 CVPR 两卡；实际调度由实验任务处理。

`run.json` 保存每步损失、批次身份 SHA、四次完整排名、选中步、参数数、源文件规范 LF SHA、输入/批准/CUDA/原 baseline 锚点和成本。`best.pt` 保存首次最佳时的真实模型、优化器及 RNG；`last.pt` 保存最终状态，二者各有原始 SHA 和权重 SHA。后续需另行核对 selected weights 复现与汇总证据。

数值非有限、梯度缺失、验证不完整或超时均停止，保留失败记录/最后权重；科学 CLI 对未 complete 退出 2。新目录必须不存在，不覆盖、继续或拼接旧训练。完整结果逐 seed 报告与 GNN 四个相同步及各自 best 的差值，符号为 text minus GNN；不足预算的片段明确不能作为能力结论。n=2 不作训练随机性显著性证明。

## CPU 证据与交接命令

初始新测试 15 passed / 22.34 秒；新增生产宽度缓存门槛后，最终无 Git 发布包 16 passed / 22.07 秒，无跳过。最后补强工程缓存非有限失败的 JSON 保存，仅重跑受影响检查，1 passed / 6.07 秒，保留两版来源 SHA 与检查记录。仅有旧 PyTorch TypedStorage 弃用提示，机器证据见 `e2-capacity-quality-tests.json`。

测试覆盖生产维度/有限梯度/参数数、查询实体编码与全编码损失/梯度一致、缓存与直接评分及失效、穷举排名/多父过滤/同分、固定登记与批准/CUDA来源拒绝、原批次和固定负例语义、图输入无依赖/标签边界/输入不变、四次完整 valid、首次最佳权重、partial/非有限梯度/超时停止、来源/优化器/RNG/权重保存、无 Git 静态入口及失败非零退出。所有训练为 40 节点 CPU 工程小图，没有真实能力对照训练或新增 GPU 使用。

从主管审核的独立发布包运行，`SOURCE_COMMIT` 为这批实际固定源；路径由实验任务填写：

```text
python -m acl_hct.capacity_control --config configs/e2_model_capacity_control.json
python -m acl_hct.capacity_control --config configs/e2_model_capacity_control.json --cuda-fixture --source-commit SOURCE_COMMIT --output-dir NEW_CUDA_FIXTURE
python -m acl_hct.capacity_control --config configs/e2_model_capacity_control.json --execute --seed 11 --source-commit SOURCE_COMMIT --prepared ORIGINAL_PREPARED --baseline-report ORIGINAL_BASELINE_RAW --approval-record CAPACITY_APPROVAL --cuda-fixture-record CAPACITY_CUDA_QUALITY --output-dir NEW_SEED11_DIRECTORY
```

seed23 同样执行自己的固定登记；原 baseline 必须用 SHA 一致的原始 JSON 字节，不能使用经换行转换的副本。科学批准字段为 `user_authorized=true`、`user_message_reference`、`scope=E2-model-capacity-control-v1`、`config_sha256`、`source_commit`、`quality_review_passed=true`、`entry_criteria_frozen=true`、`cuda_fixture_passed=true` 和 `cuda_fixture_artifact_sha256`。原确认 CUDA 证据不能代替能力对照门槛。

新增 CUDA fixture 不读取真实数据或封存确认面板：40 节点、128 维、4 步、4 次完整 synthetic valid；内部 120 秒、验证最多 20 秒，外部有界时间由主管交接。它额外核对全部 1600 对缓存与直接评分误差不超过 1e-5。主管先审核本源及 CPU 证据，再安排该单卡工程门槛及两个正式有界训练；当前不发布竞争力结论，也不追求超过原 GNN 后反复调参。
