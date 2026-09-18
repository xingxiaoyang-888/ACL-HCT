> 历史快照：以下状态、建议和授权只代表原文写作时点，不是当前执行指令。当前结论见 [报告总览](../../../README.md)。原始字节另存于归档 ZIP。

# S1b WordNet CPU预处理验收

**已完成并通过父任务审查，由代码任务统一发布。** 真实WordNet预处理及只读验收通过；没有训练、验证排名或checkpoint，未调用正在开发的训练runner。

## 固定来源与环境

源码固定为 `2ca02d6033fd5503b82bbcd5b72c5f172b07eb6e`，git archive独立release，部署前逐文件哈希核对通过；input_manifest记录的4个预处理依赖源码哈希亦与该归档一致。没有包含活动工作区的S1b修改。源码归档SHA256及依赖源码指纹见 `s1b-wordnet-run.json`。

沿用ACL独立Python3.11.7/NumPy1.26.4环境，仅安装benchmark所需的离线包：scikit-learn1.5.2、SciPy1.14.1、joblib1.5.2、threadpoolctl3.6.0，pip check通过。与实现作者本机sklearn1.3.2不同，本轮实际版本和完整包清单已记录；没有修改CVPR或共享环境，没有下载新数据。

仅处理官方WordNet3.0名词数据，原始SHA256：`6c492d0c7b4a40e7674d088191d3aa11f373bb1da60762e098b8ee2dda96ef22`。CLI先核对原包哈希和rich parser结构与既有manifest一致，再进入特征处理。输出为新的 `wordnet-b-v1-2ca02d6` 目录，未覆盖已有数据；私有完整路径保存在 `.local/s1b/`。

## 冻结参数与分组

协议 `B-child-grouped-80-10-10-v1`；split_seed=20260914、feature_seed=11、negative_seed=11、dimension=128、threads=2。82,115个稳定实体全体参与分组，每条父关系按child归组；实际边比例并不强制80/10/10。

| 集合 | 实体数 | 正关系数 | 有监督child | 根 | 原图孤立实体 |
|---|---:|---:|---:|---:|---:|
| train | 65692 | 67539 | 65692 | 0 | 0 |
| valid | 8211 | 8456 | 8210 | 1 | 0 |
| test | 8212 | 8432 | 8212 | 0 | 0 |

划分hash：`f1f36717a156541c6b2fc69bcbea34c65cdb5f7a8af1822ba822f33fd809f7ec`。独立重建排序+固定随机置换，确认分组无交叉且覆盖全部实体。根被分到valid是固定种子产生的结果，未人为重抽样。

67,539条训练正关系配270,156个负关系，共337,695个有标签训练查询，正负比1:4。逐块检查4个负父互异、非self、不属于该训练child的全部已知训练真父；没有借用验证/测试标签作负例过滤。

## 文本特征与输入隔离

- 实际特征形状82,115×128，float32，全部有限；requested/effective维度均为128，没有降秩。
- 词表75,261项，空文本0；特征仅由name/definition构造，不含tree编码、祖先或深度字段。
- 拟合实体清单严格等于65,692个train实体，IDF由训练文本独立重算并在atol=1e-12、rtol=0内一致，词表每项都出现在训练文本中。
- 保存并核对特征字节hash、词表/IDF/SVD components组成的拟合状态hash、完整/训练文本hash。SVD train-only依据固定源码及held-out扰动fixture、拟合实体记录验证；没有重复拟合第二份SVD，不能把这部分说成独立数值复现。
- 模型输入文件为features.npz、observed_graph.json、train_queries.json、input_manifest.json；完整真值/路径及valid/test标签分开存于evaluator_truth.json、evaluator_valid.json、evaluator_test.json。

特征SHA256：`9eaec4b64aaa30f70622113233c54a4e90e79812a8d5476d50aa5a363ee05cde`；fit-state hash：`0bc19e77689a85d3c9497a7e0067b11023121eeba87f9fffcd63b682a8aee8d8`。这次验收的是预处理文件隔离；尚未执行训练runner，不能据此宣称未来加载器已经强制隔离评估文件。

## G_obs与采样覆盖率

G_obs精确等于训练父边及反向边，共135,078条有向消息边；没有补闭包、自环或held-out正反边。保留自然多跳路径。与模型输入文件实际边集合独立比较完全一致；graph hash为 `601da72c716825195861844f9ca4d15f967f449ff51560a5c3bbc052c8ff54ec`。

以下是正式训练输入G_obs的静态审计，**在每batch query masking之前**。空邻域13,313个；k/N的非空均值不为这些空行虚构分母。受影响比例的分母是全部82,115实体，消息删减比例的分母是135,078条有向消息。

| fanout | 受影响节点 | 占全部实体 | 消息删减比例 | 非空节点平均k/N | 受影响节点平均k/N |
|---:|---:|---:|---:|---:|---:|
| 4 | 4268 | 5.1976% | 26.1982% | 0.970241 | 0.520274 |
| 8 | 1690 | 2.0581% | 17.1975% | 0.989339 | 0.565975 |
| 16 | 622 | 0.7575% | 10.7316% | 0.996409 | 0.602820 |
| 32 | 202 | 0.2460% | 6.3963% | 0.998857 | 0.610589 |
| 64 | 70 | 0.0852% | 3.6808% | 0.999600 | 0.607236 |

fanout8仅影响约2.0581%的节点，但移除约17.1975%的消息，反映高度节点集中贡献消息量。这个观察不等于层级坍缩；也不能直接用原图覆盖率替代划分后G_obs指标。运行时query masking还会减少候选N，后续pilot应另记录实际采样覆盖率。

## 实测CPU与内存

Slurm申请2核/8GB/30分钟、0GPU，实测CPU为Intel Xeon CPU Max9468。作业COMPLETED/0:0，占用43秒；对应版本的9项预处理fixture通过（9.73秒）。

| 阶段 | 实际进程墙钟 | 用户CPU时间 | 系统CPU时间 | 峰值RSS |
|---|---:|---:|---:|---:|
| 真实预处理 | 22.74秒 | 24.15秒 | 2.36秒 | 601,148 KiB（约587.1 MiB） |
| 独立只读审计 | 4.85秒 | 4.36秒 | 0.65秒 | 571,444 KiB（约558.1 MiB） |

预处理入口内部计时22.4948秒，审计内部计时3.9811秒。GNU time放在srun内部、直接包住实际Python进程，因此这里RSS是计算进程的驻留高水位；Slurm未提供有效TotalCPU值，未将其0计数当作实际CPU用量。原始日志和JobID在私有记录，资源已释放。GPU小时=0，准备产物总计162,257,450 bytes（约154.7 MiB），不进入Git。

## 复现与下一阶段门槛

在固定release根目录、已准备好依赖的CPU分配执行：

```bash
PYTHONPATH=src python scripts/prepare_benchmark.py --dataset wordnet --raw /path/to/WordNet-3.0.tar.bz2 --source-manifest reports/wordnet-manifest.json --output-root /path/to/new-empty-output --split-seed 20260914 --feature-seed 11 --negative-seed 11 --dimension 128 --threads 2
```

该命令拒绝非空输出目录。不要再次执行到本轮已完成的目录。后续训练任务应使用本轮产物hash和冻结协议，待明确的已提交runner与CPU验收就绪，再由父任务交接单L40短步pilot。本轮没有自行改变实体划分、文本拟合集合、负采样规则、全候选排名协议或启动真实训练。

新增公开交接文件：`reports/S1B_PREPROCESSING.md`、`reports/s1b-wordnet-audit.json`、`reports/s1b-wordnet-run.json`。处理后的数据、transform及完整实体/真值清单保留在ACL数据目录；公开审计JSON只含计数、hash、文件清单和通过项。

E1图另已用绘图源码2ca02d6从未修改JSON重建；E1计算源码仍为6cd0caa，来源分别记录。临时render_exact适配器已移至私有归档，不再提交。E1公开清单为原始JSON、结果报告、运行元数据及三张PNG共6个文件。

实验任务没有执行git add/commit/push，没有修改代码任务源码或配置。以上两组共9个公开文件已通过父任务审查，由代码任务统一提交发布。
