# E2入口与局部试跑：实现交付

研究问题：真实训练模型固定邻域中的采样偏移是否可以可靠分辨，现有模型是否足以支持后续层级损伤研究？当前结论：已补齐可审查的真实checkpoint入口和局部统计流程；尚无本轮真实模型观测，研究目的仍属证据不足。本次工程交付目的是让入口质量先于局部诊断执行，不把数值通过写成层级能力或研究收益。

## 固定范围与来源

协议见 [E2_ENTRY_AND_PILOT.md](../docs/operations/E2_ENTRY_AND_PILOT.md)。用户已授权逐步E2；本次仅seed11/23已选best768，原训练源b6269f2。固定配置 `configs/e2_entry_local_pilot.json` 的规范SHA256为 `052e30ec2f243ce9ef0d609696c7483ff390785417703e587dfe8458298014e9`。各checkpoint SHA、原始baseline报告SHA、训练配置及step都已固定，不按新诊断重新选择。

运行入口 `python -m acl_hct.e2_pilot --config configs/e2_entry_local_pilot.json` 默认仅静态输出。执行须加 `--execute --seed 11`（另一独立作业为23）以及 `--prepared`、`--checkpoint`、`--training-release`、`--baseline-report`、`--approval-record`、`--source-commit`、`--output` 和分配内 `--device cuda`。所有路径由实验任务在隔离目录提供；baseline-report用本仓库原始对应seed报告。输出必须为全新.json，源码身份在无Git归档中必须显式提供。

批准记录须包含真实用户消息引用及 `user_authorized=true`、`scope=E2-entry-local-pilot-v1`、`config_sha256`、最终 `source_commit`、`quality_review_passed=true`、`entry_criteria_frozen=true`。这是来源审计记录，不能由合成fixture或主管就绪替代用户授权。当前用户授权已明确；固定版本质量审查仍由主管完成。

先验证checkpoint逐字节hash及全部训练源码hash，再核对训练config、prepared manifest、valid查询hash和原始完整valid选择记录。遵循原runner首次最大micro MRR规则，不加载优化器续训、不保存checkpoint。结束重新核对checkpoint及内存权重哈希。列表/元组参数通过规范JSON哈希比较，避免序列序列化类型差异误报。

严格输入白名单为manifest、G_obs、train_queries、evaluator_valid、entity_split及features；复用的完整valid排名也仅过滤当前valid child的其他真父。没有打开evaluator_test/evaluator_truth。entity_split含测试实体ID仅用于原始transductive分组校验，未取得测试关系。

## 分步门槛

1. 原生FP32完整两层前向，同权重/特征重跑FP64，统一将两者归一化到FP64未来叶。保存投影改变量及数值代理，原生模型/任务输出不被诊断投影替换。完整输入再聚合检查一致性。
2. 使用best权重实测两层消息及输出半径、near-bound比例；门槛为0.95×1.2=1.14，训练last比例不能替代。按既有H_dev规则固定根及面板，报告根覆盖、直接/正远祖顺序和unknown。只打开development面板的结果；确认池清单冻结但不评估。没有臆定0.5机会参照，也没有无图文本比较结论。
3. 仅入口做一次全valid、全实体候选过滤排名，180秒内未完整完成则停止。当前完整valid与原始best768记录逐一核对query对/数量和child数量，micro及child-macro MRR绝对差均须<=1e-8，记录差值；任一不符则不采样。无需每次采样重新做任务评估，本批没有R_task矩阵。
4. 上述质量通过后，先local L1固定16次，再完整L1输入下的local L2（F/S）固定16次。L1失败/超时停止L2。第一层和第二层独立SHA256种子流，seed2026091605，namespace含checkpoint seed。没有S/F、S/S、修正或N1。

FP64一致性门槛预先固定如下，均为绝对值；这不是效应量或结构能力阈值：

| 检查 | 最大容差 |
|---|---:|
| 原生点提升后、投影前流形残差 | 2e-4 |
| 统一投影ambient位移 | 1e-4 |
| FP64流形、完整及局部Log/Exp往返、自Log | 1e-10 |
| 原生对同权重FP64完整前向距离 | 1e-4 |
| 切向约束 | 1e-9 |
| 完整重聚合及未采样行原生输出差 | 1e-5 |
| MSE分解残差 | 1e-10 |
| 当前完整valid micro/child-macro MRR相对原best768差 | 1e-8 |
| variance/MSE最小允许值 | -1e-12 |

所有观测必须有限。误差超过容差、完整valid不完整、缺来源或几何域失败均停止依赖步骤，不重抽、不放宽门槛、不换checkpoint。阈值取自既有FP32/FP64受限几何域的工程容差及更宽的跨精度上限，不保证小于上限的科学信号均可解析。每节点经验数值代理是同权重完整FP32/FP64差、自Log和往返差最大值，不是严格误差上界；低于代理的均值范数和方向投影明确不可解析，MC误差另列。

## 统计与覆盖

共同完整基点下的Log偏移按完整图采样重复流式累计，保存每节点方差、MSE、偏差范数、可为负的噪声扣除平方偏差、交替奇偶独立半样本内积。预定方向为完整H_dev参考根的向外径向，不从本批采样偏移拟合；锚点重合或数值不可解析时保留NA。组级MSE和方向投影MC SE使用16次整图重复，不能用节点×重复扩大样本量；偏差范数等非线性统计没有冒称这些SE是其置信区间。

分组保留V、A、P、P\A、V\P、固定度数及根可达/unknown；记录所有ID和分母。仅局部固定输入诊断允许A外数学采样偏移为零：实际原生重聚合差仍检查并记录，超过容差即失败，不将批处理舍入解释为采样偏移。高维均值向量仅保存A，所有节点的标量和全体组分母保留。此规则不能复用于真实两层传播。

## 预算、失败与测试

每seed独立单L40/2CPU/16GiB、外层1800秒，内部1500秒预留保存余量；最多两作业并行。入口含两次完整前向及全valid，局部32次只复用完整消息场；不占用GPU等待代码。已有完整valid编码/排名约25秒仅作为预算依据，不能当成新FP64/局部诊断实测时长。每个82,115×129 FP64场约80.8MiB，数值对照与两组流式统计预计为数GiB级张量，16GiB主存/单L40可容纳；实际峰值和耗时仍必须记录，超限即停而非扩大资源。

时限在阶段/采样重复/排名child边界检查，单个操作可越过内部剩余预算，Slurm提供硬上限。入口、L1完成及最终状态原子保存；外部终止时最近快照只证明当时已完成部分，不能当整体完成。文件无自动resume入口。任何失败的partial重复仅记录计数/原因，不输出选择性成功重复的科学均值。

必要离线检查包含40点局部独立参考、128维近边界fixture、完整/失败分步门槛、严格标签读取、checkpoint选择与只读权重、固定配置/批准拒绝、阈值边界、真正无Git归档CLI。无真实checkpoint诊断、远程作业或科学试跑由代码任务执行。最终在真实无Git临时归档中执行 `python -m pytest tests/test_e2_pilot.py -q`：**22 passed /21.16秒**，无跳过；仅PyTorch旧TypedStorage兼容警告。测试/源码/配置规范LF哈希见 [e2-entry-quality-tests.json](e2-entry-quality-tests.json)。首次检查14通过、1项发现tuple/list来源比较误报，改为规范JSON哈希后该项单独通过；最终归档包含复现门槛及128维边界fixture，全部通过。未重复无变更的旧suite。
