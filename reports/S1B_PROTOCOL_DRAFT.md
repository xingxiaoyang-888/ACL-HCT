# S1b 任务与输入输出约定（冻结 V1）

父任务已按 `docs/operations/S1B_PROTOCOL_DECISIONS.md` 冻结本版。协议fixture、固定commit实际预处理与训练runner验收后，可按授权执行有界pilot；当前文档不代表真实训练已完成。只定义原始自建两层Lorentz-Mean起步基线，不宣称复现某篇模型或已具竞争力。

## A/B 与实体分组

- 协议A是已知完整taxonomy上的表示保真，可用T传播，只评价已知结构保存；其manifest与checkpoint必须标为A。当前准备入口只生成B，不把B的完整G_obs参照叫作A。
- 协议B是未知直接父关系恢复。以全部稳定实体ID（含根/孤立实体）排序后用固定split_seed=20260914置换，按floor(0.8n)、floor(0.1n)、剩余分成train/valid/test。根/孤立实体也分组，为文本拟合集合提供唯一规则；父任务已接受全部实体分组口径。
- 每条正关系按子实体分组，所有父边同组；MeSH实体=descriptor，不展开位置副本。边数比例因此不保证恰好80/10/10。重复边先去重，不做可达路径筛除。训练图自然多跳路径允许存在，不作旧helper式全可达隔离。
- 消息图只有train父→子及明确反向。模型可知全部节点ID和文本，属于transductive设置；held-out子实体仍可能作为train子节点的父实体出现在可见边中。这不补回held-out父边。空邻域仅self回退，不额外添加自环。

## 文本、路径与真值隔离

WordNet固定noun @/@i，文本=全部synset词名+gloss，保留多父。MeSH文本=DescriptorName/String+PreferredConceptYN=Y的ScopeNote；保留每个descriptor全部原始tree位置、逐位置已知祖先ID集合及歧义前缀列表。tree编码不进入特征。官方三位置歧义继续隔离相接直接边，祖先元数据遇歧义不强选owner，显式标记不完整。

TF-IDF词表/IDF/SVD只在实体train组的name/definition上拟合，包括该组根/孤立实体；其余实体仅transform。启动requested_dimension=128，fixture/不足秩语料下effective_dimension=min(128,n_train-1,vocab-1)，显式记录，不能悄悄宣称仍128维。输出训练实体列表、原文本hash、fit-state hash、实际维数、版本、vocabulary/IDF/components及特征hash。自然文本语义词保留，不宣称文本完全无层级信号。

结构/路径/真值文件与模型输入文件分开：模型只读features、node order、G_obs、train queries；评估器才读完整真值/路径和valid/test labels。不能把tree path、全图深度、测试边/闭包作为输入特征。

## 训练查询与屏蔽

训练正例仅train直接关系；每个正例固定4个均匀负父候选，无self、无该train子实体的其他真父，单正例内无重复。所有同子实体真父在train，因此负例过滤只需train labels，不调用valid/test truth。跨正例允许重复负候选；seed及候选hash保存。

每个优化batch的全部正查询边及反向边，在生成该batch的两层采样计划**之前**从G_obs屏蔽；N在屏蔽后重新计算。其他可见多父边保留。所有方法共享同一被屏蔽图和分层无放回抽样计划；plan不可引入不可见ID或重复ID。当前准备模型不自动训练，runner必须将屏蔽接口接通后再验收。

## 任务头、选择与评估候选

两层自建L-Mean，每层linear→按向量范数做宽度无关的缩放半径界1.2→Exp_o→等权Lorentz归一化邻居均值；两层之间Log_o。没有把TinyGNN冒称两层正式骨干。统一头为MLP([Log_o(z_parent),Log_o(z_child),差值])，hidden128/ReLU/单logit，BCEWithLogits。原始完整/采样路径同参数、同头；完整指G_obs全邻域。

开发主选择指标为valid filtered all-parent MRR，并列选更早checkpoint。首轮WordNet、训练seed11，单L40先10–20步成本定标；首作业至多1小时，完整验证频率据成本登记。**本轮不生成真实checkpoint，不打开测试集选配置。**

排名候选=全部已知实体ID减self，对每个正父-子query保留目标父，过滤该子的其他真父。过滤信息仅评估器持有；valid/test按子分组，因此同子真父没有跨split副本。按候选块打分，报告候选规模/过滤后规模；同分用平均秩（1+严格高分数+0.5×其他同分数）。主MRR为query-micro，同时输出child-macro MRR及Hits@1/3/10。固定validation_probe仅用于测速/监测，不能替代完整valid选checkpoint。若开发期启用1000抽样候选，必须固定并另命名sampled-candidate，不能与all-parent MRR混报。当前代码准备阶段不把抽样排名当完整排名。

## 冻结诊断接口与成本界

`full_reference`缓存两层的固定输入、变换后消息、完整输出；`frozen_layer`仅对指定层采样且固定该层输入，`encode`则真实逐层采样传播，二者分开。完整oracle缓存只出现在评估接口，不送进部署correction。诊断前使用eval/no_grad并明确checkpoint来源。

聚合按padded message预算分块；最大单行超预算时显式失败，不改fanout。CPU采样/索引组装仍有Python循环，本轮没有大图吞吐证据。划分后G_obs必须重算受影响节点、k/N和删减消息比例；若任一fanout处处k=N，作为等价完整配置记录，不重复训练。

## 待验收清单

1. 父任务已冻结实体分组、文本fit集合和全候选排名规则，见决定文档。
2. 离线fixture验证多父/多位置、split一致性、文本fit隔离、batch查询屏蔽、两层梯度、完整/采样及冻结诊断接口。
3. 固定代码提交后，实验任务才能做真实预处理/审计与有上限的开发测量；训练runner和验证checkpoint保存仍需独立接入验收。

评分缓存使用第一层MLP的精确代数分解，在一次固定checkpoint/eval内复用父/子投影；优化器或load_state_dict更新后旧缓存失效。每次只存一个子实体的N个分数，候选激活分块，多父query共享评分。已与直接MLP值及完整候选独立排名对照。
