> 历史快照：以下状态、建议和授权只代表原文写作时点，不是当前执行指令。当前结论见 [报告总览](../../../README.md)。原始字节另存于归档 ZIP。

# E2 骨干诊断第1步：原实现语义与证据边界

研究问题：原双曲 GNN 落后于无图 MLP，是否存在消息边界、梯度、self 通路或训练/推理语义方面的实现原因？**源码证实的是平滑径向压缩、非空邻域缺少显式 self/残差，以及训练图掩码与采样相对于 valid 全邻域的变化；没有证据证明消息被硬截断到 1.2，或约 65% 的消息梯度已经死亡。检索劣势的主因仍未识别。** 本步只读实现审查目的达到，不能据此修改已冻结模型或宣称某项修改会恢复竞争力。

审查对象为原训练源 `b6269f24e33602c10c39ee6b9bac57fe0901a405`，不是根据后续代码猜测旧行为。backbone/geometry/aggregation/train/protocols/ranking 六个当前文件与该版本 Git blob 的规范 LF 内容逐项一致，以下本地文件行号因此可以定位原训练实现。MLP 来源为 `e1dd5ee3e9b2ac2a89cdf6b49dd71247987e4246`，近边界诊断来源为确认固定源 `49609014610fe86dbf6b19fe8a1fe94df7ecd019`。

## 1. 半径与 near-bound：平滑渐近界，不是硬边界命中

原层先计算 `u=Linear(inputs)`，再令 `v=(R/√c)u/√(1+||u||²)`；`from_spatial` 的实际操作是把 v 作为原点切向量进行 exp，并不是将 v 直接作为双曲点的空间坐标。故消息的原点缩放距离为

`r=√c·d(origin,message)=R·s/√(1+s²)`，其中 `s=||u||`、`R=1.2`。

任何有限 s 的 r 都小于 1.2；没有按 r≥1.2 将输出切平的分支。c=1 固定，1.2 是人工选择的消息尺度界，不是 Lorentz 空间的几何边界。`check_point` 的工程支持域是缩放原点距离≤3，越界抛错，不是将训练点 clamp 到 1.2。

| 已证实的定义 | 精确来源 |
|---|---|
| R=1.2 固定标量，Linear 后按向量范数平滑压缩 | [backbone.py:76](F:/ACL/_HGT/src/acl_hct/backbone.py:76)，76–86；原两份训练配置 `scaled_radius=1.2,c=1` |
| v 进入原点 exp，原点距离由切向量范数决定 | [geometry.py:112](F:/ACL/_HGT/src/acl_hct/geometry.py:112)，112–116；exp 75–87 |
| 训练 near_bound_fraction 为每层全体节点的聚合前消息，`acosh(√c·time)≥0.95R` | [train.py:234](F:/ACL/_HGT/src/acl_hct/train.py:234)，234–241；trace 来自本步掩码/采样 forward、在 optimizer.step 前计算 |
| 确认 near_bound_message_fraction 为选定权重完整图参考的聚合前消息，阈值1.14；另统计聚合输出 | [e2_development.py:236](F:/ACL/_HGT/src/acl_hct/e2_development.py:236)，236–247；阈值是固定诊断定义 |

1.14=0.95×1.2，等价于 `s≥0.95/√(1−0.95²)=3.0424349223`，不是“碰到边界”的事件。训练日志统计的是当步更新前消息，即使日志在 optimizer.step 后整理，也不能称更新后或完整图参考。

从已发布原确认摘要读取，未重新做模型 forward：

| seed，best768，完整参考第二层 | 消息 r≥1.14 | 输出 r≥1.14 | 消息平均 r / 最大 r |
|---|---:|---:|---:|
| 11 | 65.7578% | 42.6804%（35047/82115） | 1.147337 / 1.192619 |
| 23 | 64.1040% | 39.5884%（32508/82115） | 1.144655 / 1.191939 |

第一层两 seed 的消息比例均为0。不能把第二层的消息65%左右写成所有层、聚合输出或训练参数的边界比例，也不能直接当作语义根半径；此处参考的是物理原点。

## 2. 梯度：径向衰减已证实，实际训练死区未证实

压缩映射对 u 的 Jacobian，沿 u 的径向特征值为 `(R/√c)/(1+s²)^(3/2)`，垂直方向为 `(R/√c)/√(1+s²)`。有限 s 时两者均正；径向衰减更快。c=1 时：

| s | 消息缩放距离 r | 径向导数 | 横向导数 |
|---:|---:|---:|---:|
| 0 | 0 | 1.2 | 1.2 |
| 3.042435 | 1.14 | 0.0365332 | 0.374700 |
| 8 | 1.190733 | 0.00228987 | 0.148842 |
| 16 | 1.197663 | 0.000291260 | 0.0748539 |

因此 near-bound 提示幅度方向的优化可能困难，与横向方向不同；不能将这一局部 Jacobian 等同于端到端参数梯度。平滑压缩对有限 u 仍保留方向与可逆的径向信息，单靠尺度界不能证明信息完全丢失。

源码确有 clamp，需要按作用区分：

- [geometry.py:80](F:/ACL/_HGT/src/acl_hct/geometry.py:80)，80–84：exp 的非负范数保护及小量解析分支；[geometry.py:96](F:/ACL/_HGT/src/acl_hct/geometry.py:96)，96–100：log 的位移量非负保护、小量展开和安全的未选分支。它们不是按消息半径1.2截平。
- [geometry.py:103](F:/ACL/_HGT/src/acl_hct/geometry.py:103)，103–109：norm2 有非负 clamp，distance 在重合点非光滑；原 BCE 任务路径使用原点 log 后的关系头，不以 distance 为训练损失。不能用 distance 的重合点行为替代任务梯度诊断。
- [aggregation.py:120](F:/ACL/_HGT/src/acl_hct/aggregation.py:120)，120–145：确有候选修正 max_step 截断，但原配置 method=none，active 恒 false，输出取未修正 p；它不是原消息半径压缩。原训练日志 clipping_rate=0 也只说明候选修正未触发，不能说明径向压缩不存在。
- [backbone.py:94](F:/ACL/_HGT/src/acl_hct/backbone.py:94)：GNN 关系头有 ReLU，可能出现零导数单元；GNN 编码层没有 MLP 那样的逐坐标 ReLU。原梯度检查 [train.py:230](F:/ACL/_HGT/src/acl_hct/train.py:230)，230–231，只查 missing/nonfinite，零张量会通过，没有记录零/近零梯度比例或各层 Jacobian。

微型 CPU 检查在三维人工 u、s=0/0.001/0.1/1/3.0424/8/16 上核对解析 Jacobian、FP64 差分及原 geometry 的 `log_origin(exp_origin(squash(u)))` 链。14 个 FP64/FP32 点均有限且径向/横向导数为正；链的最大解析误差分别5.77e−15/2.03e−6。该证据排除了这些人工点上的普遍零梯度解释，不保证真实全部参数、其他输入域或训练步骤没有梯度问题。

## 3. self、mask 与采样：正常聚合没有显式自身通路

原 prepared 的消息边由训练 parent-child 关系及 reverse 构成，不额外加入 self loop：[protocols.py:29](F:/ACL/_HGT/src/acl_hct/protocols.py:29)，29–35；[protocols.py:40](F:/ACL/_HGT/src/acl_hct/protocols.py:40)，40–50。聚合入口不主动追加自己：[backbone.py:63](F:/ACL/_HGT/src/acl_hct/backbone.py:63)，63–68。

`self_points=points[start:end]` 容易误读为每个节点都有 self message；实际只在 k=0 的空邻域使用。非空行是所选邻居消息的等权 Lorentz 归一化均值，没有拼接自身、残差或可学习 self 权重：[aggregation.py:116](F:/ACL/_HGT/src/acl_hct/aggregation.py:116)，116–120。空行保留的是本层已变换的自身消息，不是跳过 Linear 的输入残差。

三节点 CPU 例子证实：邻域 `[1],[0],[]` 的输出分别为消息1、消息0、消息2；非空节点0对自身消息的直接导数为0，而对邻居非零。屏蔽正例0–1及 reverse 后两行变空，输出改为各自消息。该例不是对真实网络最终 self 梯度的测量；两层无向传播可以经回返的二跳路径间接带回自身信息，不能称所有自身文本完全丢失。

| 阶段 | 图和采样语义 | 精确来源 |
|---|---|---|
| 训练 | 每步从原固定组抽批次；把本批全部正例及 reverse 同时移除，然后按剩余 N 分别采两层计划，fanout16/16 | [train.py:219](F:/ACL/_HGT/src/acl_hct/train.py:219)，219–227；[protocols.py:109](F:/ACL/_HGT/src/acl_hct/protocols.py:109)，109–118；[backbone.py:20](F:/ACL/_HGT/src/acl_hct/backbone.py:20)，20–34 |
| valid | 原 G_obs 不作本批 mask，两层完整邻域；valid 目标及 reverse 已在 prepared 构图排除，故不因未再次 mask 而成为目标边泄漏 | [train.py:198](F:/ACL/_HGT/src/acl_hct/train.py:198)，198–202；load_prepared 103–107 检查 withheld child/edge |
| 采样 | N≤16 完整保留且不耗随机数；N>16 均匀无放回。sampling_rng 与 batch_rng 分开；两层共享同一 sampling_rng 顺序 draw，不是确认入口那种独立派生层种子 | [aggregation.py:19](F:/ACL/_HGT/src/acl_hct/aggregation.py:19)，19–25；train 132、222 |

训练存在两个变化轴：batch mask 改图/度数/空邻域，以及采样相对于全邻域。前者还使编码依赖同批的其他正例。child-grouped split 使 valid child 的全部父关系不在 G_obs，但它仍可能作为 train child 的父而保留其他边，这是明确的 transductive 协议：[protocols.py:13](F:/ACL/_HGT/src/acl_hct/protocols.py:13)，13–35。训练 child 和 valid child 的结构角色并不相同；当前没有按角色/度数分解劣势的因果证据。

已验收确认中，同一 best 的未做训练 batch mask 的 S/S-f16 任务均值差约−0.000065/−0.000560，均未经统一 Holm 建立非零。它不能排除采样训练对学习轨迹的影响，也不能证明 batch mask 无影响；但不足以支持“全部检索劣势都由 full 推理造成”的单一解释。

## 4. 与 MLP 的公平性：共享预算成立，严格单因素比较不成立

| 比较项 | 已证实 | 边界或缺口 |
|---|---|---|
| 数据与监督 | 同一 train-only TF-IDF/SVD 特征、原固定正例与每正4负；词表/IDF/SVD只拟合 train，所有实体文本可 transform | [protocols.py:84](F:/ACL/_HGT/src/acl_hct/protocols.py:84)，84–105；负例53–81；不是独立 test 泛化 |
| 优化调用与预算 | 两 seed、128正例/步、1024步、Adam lr0.003、BCE；均未给 scheduler/梯度裁剪；4次完整 valid、严格更高才替换 best | 原 train 129–132、205–210、219–232；[capacity_control.py:201](F:/ACL/_HGT/src/acl_hct/capacity_control.py:201)，201–204、264–288；相同步数/学习率不保证各架构同样收敛或同 FLOPs |
| 批次与负例身份 | 来源不同但 seed+1 独立 CPU randperm 语义相同，既有验收已逐步核对2048批 SHA 与固定负例 | 原 train 132、219–224；capacity_control 204、277–282；不是给 MLP 另采简单负例 |
| 结构与参数 | 两层128、相同384→128→1关系头，共82433参数；GNN 是径向压缩/exp/图归一均值/log，MLP 为两个 Linear+ReLU 且不聚合 | backbone 82–105、111–113；[text_capacity.py:14](F:/ACL/_HGT/src/acl_hct/text_capacity.py:14)，14–17、24–37；同时改变图依赖和非线性，不是只改曲率 |
| 训练计算 | GNN 对全图两层传播，MLP 只编码查询实体；无跨实体层时 MLP 的该计算方式与全实体编码同一查询损失/梯度一致 | 原 train 225–227；text_capacity 31–37；一致性已有 FP64 工程证据，不能把耗时差当受控加速实验 |
| 选模 | 都有同样4次 full-valid 机会；GNN选768，MLP选1024，最佳步不同 | train 205–210；capacity_control 264–268；既有保存权重已复现，最后一步选中不证明充分收敛 |

**初始化核对**：两模型均先设置相同 seed，再以相同形状和顺序构造两层 Linear 与关系头；ReLU、几何常量不消耗初始化随机数。微型 CPU 在 seed11/23 上，8个对应参数张量逐字节相同，说明代码初始化没有单独给某模型有利权重的机制。

初次检查的本机 PyTorch 是2.0.1+cpu，实际两种训练均为2.5.1+cu124；当时重建的 MLP 初始 SHA **未匹配**实际保存的初始 SHA。这一跨版本不匹配历史及私有机器证据保留，不能把该次检查改称历史复现通过，也不能凭它认定历史比较违规。

**原2.5.1环境的初始化核查现已补齐**：[e2-backbone-initialization-quality.json](F:/ACL/_HGT/reports/e2-backbone-initialization-quality.json) 记录 CPU 上 PyTorch2.5.1+cu124、Python3.11.7、NumPy1.26.4，仅 manual_seed 后构造原冻结 GNN/MLP，无数据/检查点加载、网络前向、优化器或GPU。seed11/23 各8个对应参数逐字节相同，全部 MLP 初始张量 SHA 与各自历史 run 保存的 SHA 匹配，构造后的 CPU RNG 状态也相同。因此历史版本的构造与 MLP 记录字节核对已通过。原 GNN 初始 SHA 未直接记录，仍是由冻结构造及初始化顺序重建，不能表述为两份历史初始 checkpoint 的直接比较；相同初值也不证明架构、FLOPs或收敛充分性相同。

## 5. 假设、缺口与最小下一验证

| 待解释的机制 | 当前等级 | 最小验证建议（本步未执行） |
|---|---|---|
| 径向压缩导致优化各向异性、消息幅度集中 | 映射导数与 near-bound 定义已证实；导致 MRR 劣势仍是假设 | 主管另行放行真实只读诊断后，用固定训练小批与固定已存权重，记录两层 preactivation 范数/径向与横向 Jacobian、BCE 参数梯度范数及零/近零比例；不做 optimizer.step |
| 缺少显式 self 使文本检索信息难保留 | 非空无直接 self 与空行分支已证实；最终信息损失未证实 | 先保留本报告三节点解析例；真实只读阶段按空/非空和结构角色检查自身文本敏感度。任何 self/残差模型都是新版本，不能在旧 checkpoint 上补层或把微型改善当能力通过 |
| batch mask 与 sampled/full 改变训练/验证分布 | 图及采样语义已证实；其效应大小未识别 | 固定同一批训练查询与同一权重，只改变 masked/unmasked、sampled16/full 两轴，记录空邻域切换及评分差；需要真实 forward 新放行，当前禁止，不新增训练 |
| 初始化差异或共同预算未充分优化 | 共同调用/预算成立；原2.5.1构造及MLP历史初值字节核对已通过，GNN历史初值为重建；架构最优性仍有缺口 | 保留旧2.0.1不匹配历史，不再把2.5.1初始化审核列作待补工作。当前loss/四次valid不足以判定收敛，不自行加步数/调lr |

初始化微查已补齐；下一步按[固定权重诊断方案](F:/ACL/_HGT/docs/operations/E2_BACKBONE_FROZEN_DIAGNOSIS_PROPOSAL.md)准备压缩梯度与mask/采样的独立入口、固定配置和CPU小图测试。该方案不直接测量最终自身文本信息保留，不给self归因下结论。真实数据前向/反向仍须工程复核及用户同意；结构修改及重新训练另立版本与批准门槛。现有E2条件性损伤结果不撤销，也不因本报告自动进入E3。

## 本步证据与操作范围

私有脚本 `.local/e2_backbone_implementation_checks.py`，机器证据 `.local/e2-backbone-implementation-checks.json`；命令为 `python .local/e2_backbone_implementation_checks.py`。14个三维代数点、三节点聚合/掩码、六节点采样接线及两组模型构造检查通过；内部 CPU 用时约1.70秒。本脚本禁用网络级 encode/score/forward，未载入检查点、真实 feature 数据或优化器；未训练、使用GPU或打开test/truth文件。

原六文件 SHA 与公开摘要原始 SHA 已记录。初步只读打印脚本有两次包装错误（行号打印越过文件末尾1行；一次遗漏PYTHONPATH），均非研究代码/数值失败，机器证据保留；不以初始化跨环境 SHA 未匹配冒充通过。本轮仅新增本报告及私有脚本/证据，不改 STATUS、冻结源码或现有科学记录，不提交。
