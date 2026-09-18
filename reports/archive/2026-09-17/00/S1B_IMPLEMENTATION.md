> 历史快照：以下状态、建议和授权只代表原文写作时点，不是当前执行指令。当前结论见 [报告总览](../../../README.md)。原始字节另存于归档 ZIP。

# S1b：冻结协议、预处理与原骨干实现

2026-09-16。本阶段实现已按父任务 `docs/operations/S1B_PROTOCOL_DECISIONS.md` 冻结V1；详细输入输出见 `S1B_PROTOCOL_DRAFT.md`（保留原文件名，内容已更新为冻结版）。本报告不表示真实训练或正式基线能力已完成。

## 已实现

- `taxonomy.py`：WordNet词名/gloss与@/@i；MeSH名称/preferred scope note、descriptor所有原始位置、逐路径祖先及歧义前缀。rich parser保持原边语义，MeSH歧义不强选owner。完整元数据只进入evaluator文件。
- `protocols.py`：全部实体按split_seed=20260914做80/10/10，父关系按child同组；G_obs=train边+reverse，无额外闭包/可达筛除。训练文本fit集合就是train实体组，包括其根/孤立实体；feature_seed/negative_seed与split seed分开，默认11。
- TF-IDF/IDF/SVD train-only拟合，保存词表、IDF/components、fit/text/feature hash。requested128，不足秩显式记录effective维数。scikit-learn作为可选benchmark依赖，本机1.3.2；只取name/definition，路径字段不是特征。
- 训练每正例固定4个均匀负父，无self/已知train真父、单组无重复；验证使用known=set，避免O(EV)成员扫描。query mask在两层采样前统一移除当前batch全部正边与reverse，N据屏蔽后候选计算。
- `backbone.py`：真正两层自建Lorentz-Mean，宽度无关的缩放半径界1.2、统一有序关系MLP；显式采样计划在方法间共享，按padded消息预算分块。full_reference/frozen_layer与逐层encode分别表达固定层输入诊断和真实两层传播。
- `ranking.py`：一次eval先Log及父/子投影缓存，再分块相加ReLU与最后线性层；与直接MLP数值对照。全部实体减self，评估器过滤其他真父，同分平均秩；query-micro MRR主指标与child-macro补充，候选规模/Hits/每query秩记录。每次只存一个child的N个分数，多父复用。模型切train或权重版本改变即拒绝旧缓存。限时未完成明确标incomplete，不能作完整valid选择指标。
- `prepare_benchmark.py`：离线CLI校验raw SHA256与rich结构=既有source manifest，训练输入/评估真值分文件，拒绝覆盖非空输出；统计实体/监督child/正关系/root/isolate数量与G_obs采样覆盖率，固定threads并记录预处理时间/依赖源hash。没有git依赖，没有下载或自动训练。

## 验证及边界

源码fixture覆盖：多父/instance-hypernym、MeSH多位置和歧义；固定划分/正反边隔离；held-out文本或路径改变不能改变训练fit-state；FP32/64两层全邻域与回退一致、两层非零有限梯度、冻结层接口；两步query-masked优化；宽度2/128/512保持半径界；排名代数缓存、旧缓存拒绝、过滤/平均同分和独立直接MLP排名；临时非git目录的完整预处理CLI压缩fixture与manifest核验。

此前S1b+原数据/契约26项通过，S1b+排名12项通过；加入CLI预处理与E1归档绘图回归后执行15项针对性检查：15 passed / 30.44s。本机Python3.10.11/torch2.0.1+cpu/sklearn1.3.2。未以扩大容差或标签路径生成特征通过测试。

尚未执行真实WordNet/MeSH新预处理，没有真实checkpoint或GPU作业。本次提交可先交实验任务在固定release的CPU资源执行WordNet预处理；训练runner、保存随机状态/优化器/验证选择的接入继续完成后，才交单L40的10–20步有界pilot。不能把fixture训练说成真实基线或竞争力证明。

## 可复现与交接

```powershell
$env:PYTHONPATH="src"
python -m pip install -e '.[test,benchmark,plots]'
python -m pytest tests/test_benchmark_preparation.py tests/test_ranking.py tests/test_e1_archive.py -q
# 正式预处理由实验任务在固定源码CPU分配执行；输出目录须不存在或为空
python scripts/prepare_benchmark.py --dataset wordnet --raw data/WordNet-3.0.tar.bz2 --source-manifest reports/wordnet-manifest.json --output-root data/processed/wordnet-b-v1 --split-seed 20260914 --feature-seed 11 --negative-seed 11 --dimension 128 --threads 2
```

模型训练可读：features.npz、observed_graph.json、train_queries.json、input_manifest.json。完整路径/真值和valid/test标签分到evaluator_*，训练runner不得读取test/truth文件。输入清单完整记录train实体、实体顺序hash及G_obs hash；实际预处理后仍须验收不泄漏和成本。

## 顺带修复E1图标签

plot_e1现在按JSON的exact/MC模式区分标题和误差说明；全精确组不再称MC估计。此修改只重建图，不重新运行正式实验。第一组E1仍缺simple scaling和匹配零均值噪声，不宣称所有科学对照已完成；bias/MSE的相反结果必须保留。
