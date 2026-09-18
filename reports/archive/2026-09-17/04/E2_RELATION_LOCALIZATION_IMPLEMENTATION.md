> 历史快照：以下状态、建议和授权只代表原文写作时点，不是当前执行指令。当前结论见 [报告总览](../../../README.md)。原始字节另存于归档 ZIP。

# E2 关系间隔定位 A：最小工程交付与来源修复

2026-09-17。研究问题：旧 O/C/Q 相对 S 的父子及远祖顺序变化，能否由父子相对径向移动定位，并与检索变化按节点对齐？**离线工程准备目的达到；A 的真实关系定位结果尚未产生，科学目的仍待 CPU 分析。** 原 E2 目的仍部分达到；本报告不把代码正确称为层级恢复。

## 范围与实现

仅新增 `src/acl_hct/recovery_relations.py`、`src/acl_hct/relation_entry.py` 和 `tests/test_recovery_relations.py`。复用已接受的四个成功分片、36份原档案、两个冻结模型各16次原评价；seed11 的0–7严格取独立卡 attempt2。旧32个运行源码、旧测试、配置、结果及原四个未跟踪文件保持原字节。无需原图、特征、检查点重新加载，不运行模型、采样、优化器或训练。

固定 `p[root]` 为参考根，保持原面板、覆盖和子节点权重。逐关系保存父/子距离、相对 S 的变化和间隔，核对 `delta_gap=delta_child-delta_parent`；保存 S→X、F→S、F→X 的全部九种正确/平局/错误转移，不筛掉不利关系。保存逐子节点指标，包括无可评价关系与未知深度；未知不转为零。

分组仅使用冻结参考信息：原七种 degree strata、原 H_dev 最短深度、F 正负平局、参考间隔绝对值与旧校准偏移范数。间隔未分辨/零偏移单列，其余使用 NumPy linear 四分位、合并重复边界，相同值归同组。分组在候选分析前固定，跨片核完整设计签名，不创建交叉分组或子组显著性筛选。

总体顺序指标复用原 child-macro 加权口径。关系子组保留原子节点权重除以原可评价关系数，再按该组的关系权重质量归一；该描述性子组不能替代总体。degree/depth/bias 子节点组保存完整未知覆盖；无法分配至几何关系组的未知保留在总体覆盖并明确标记。原总体正确率及未分辨比例须在旧标量核对容差 `1e-12` 内复现，原覆盖必须完全一致。

使用全部保存排名核完整 validation 覆盖、过滤候选数及排名指标，再按节点ID对齐面板的子节点 reciprocal rank 变化。层级/检索同升、同降与相反变化只作描述；交集覆盖明确保存，完整 MRR 单独保留，交集不替代总体且不作因果比例解释。

## 必要验收与真实元数据修复

命令（工作根 `F:\ACL\_HGT`）：

```powershell
python -m pytest tests/test_recovery_relations.py -q --junitxml=.local/e2-radial-component-quality/a-tests.xml
```

28项必要检查通过，6.37秒，0失败/错误/跳过。覆盖已知关系翻转和父子变化分解、多父节点原权重、固定参考根、未知覆盖、四分位重复边界、原结构指标复现、检索排序与完整覆盖、跨片固定设计/重复/身份、原档案绑定与时限、纯标准库监督入口、精确源码/质量/输入/当前协议门。

另只读核验真实四份 run/progress/supervisor，共12份小JSON，无完整 NPZ 或模型读取。首次真实 schema 回归复现 `KeyError: source_lf_sha256`：原 run.provenance 没有该字段，32源哈希实际在 progress.json。修复只从原绑定 progress 读取并核源码哈希，交叉核 run/progress 的 source_commit、config、protocol、science 阶段及 release；不改旧原件、不允许缺字段、不放宽来源门。新增四类 progress 实值漂移拒绝检查，真实四片 header 复检通过。首轮子进程测试缺少 pytest 的源码路径属于夹具错误，显式任务 PYTHONPATH 修复，失败证据保留。

机器质量与证据哈希见 [e2-relation-localization-quality.json](../../../e2-relation-localization-quality.json)。本地 CPU 库版本只支持上述工程检查，不能替代原环境运行。

## CPU 入口、输出与下一步

新入口在启动数值子进程前计时，worker 全过程1680秒，allocation1800秒、最多4CPU与16GiB、0GPU；包含来源门、Torch/NumPy 导入、全部旧档案字节/数组校验、分析及写档。只终止自己的直接子进程，记录超时及实测回收延迟；原 PyTorch `2.5.1+cu124`、NumPy `1.26.4` 强制核对。逐原重复读档，不同时驻留全部4.53GB旧数据，也不复制这些原点档案。

```powershell
python -m acl_hct.relation_entry --execute --config <原登记配置> --bindings <原minimum-A-input-bindings.json> --protocol <当前协议> --approval <主管A释放记录> --quality <本工程质量JSON> --source-commit <完整发布commit> --shards <seed11-r0-7-attempt2> <seed11-r8-15> <seed23-r0-7> <seed23-r8-15> --output <全新A目录>
```

应在发布的 src 路径或已安装包下执行。无 execute 时只显示静态0GPU预算。实际执行要求主管记录精确绑定两新源码、质量、输入清单和当前协议；原输入清单旧协议版本与新增停止说明版本分别留证，无需改写旧输入清单。输出为两份固定设计档、32份逐关系/子节点档案、每重复单因素汇总、原完整validation指标、进度及监督记录。部分输出不作完整科学结果。

下一步只由实验任务在源与质量放行后运行 A 的有界原环境 CPU 作业。B 的径向/正交两方向干预另行工程与数值验收，采用协议已固定的 R−S/T−S/R−T 全18项检验，不沿用最初遗漏直接方向对照的“12项”短讯。本次 A 交付没有新 GPU 申请或真实定位结果。完成当前 E2 A/B 与验收交付后先停，讨论已有结果及下一方案；不自动扩展实验、实现/训练新修正或进入后续阶段。

固定科学定义及用户授权见 [E2_RADIAL_COMPONENT_PROTOCOL.md](../../../../docs/operations/E2_RADIAL_COMPONENT_PROTOCOL.md)。当前原校准、评价样本和参考 Q 都已参与上一轮观察，本次定位属于探索性分析，不冒充独立确认。
