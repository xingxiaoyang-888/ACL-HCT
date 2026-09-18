> 历史快照：以下状态、建议和授权只代表原文写作时点，不是当前执行指令。当前结论见 [报告总览](../../../README.md)。原始字节另存于归档 ZIP。

# E2 联合恢复pilot薄入口工程交付

研究问题：在原冻结模型的f4采样场景中，独立平均误差移除O与实用third修正C能否恢复层级顺序和父节点检索，O是否优于等幅无关方向Q？**本工程阶段目的达到；真实恢复效果与平均方向作用仍证据不足，尚无本轮科学运行。** 原E2科学目的仍部分达到，匹配编码器重训继续暂停。本联合pilot不以先超过MLP为前提，也不代表进入E3–E5。

2026-09-17，独立O/Q适配器的34项已验收检查保持原件；新增入口的41项本机CPU合成集成检查通过（22.17秒）。精确命令、版本、时间、源文件LF表与只读原件检查见 [机器质量记录](../../../e2-frozen-recovery-quality.json)。本机PyTorch2.0.1+cpu，不能替代登记的2.5.1+cu124与L40数值门。新增工程未读取真实prepared、checkpoint或校准NPZ，未使用GPU，未产生科学效果数值。此前只读来源准备使用了校准readiness及旧JSON清单描述符。

## 原环境CPU输入门的配置表示修复

原环境首次CPU预检在`train_config.__dict__ != spec.training_config`处误报配置漂移，worker19.1748秒、时限240秒、无超时、0GPU，未做真实前向/反向/优化器。失败证据原件保留；不能将其视为CPU门通过或科学结果。

原训练源码以asdict保存配置：Torch检查点保留fanouts元组，JSON登记和baseline报告将它编码为列表。TrainConfig.validate不转换它的类型，直接字典相等因此误拒绝相同配置。原verify_selection早已用canonical digest处理相同表示。joint入口只将这一处比较改为严格全字段canonical JSON比较，字段缺失、实值或标量类型变化仍拒绝；不改变登记配置、输入SHA、原checkpoint选择、科学源、抽样IDs或数值/资源门。

本机用合成检查点的真实Torch序列化元组复现原报错，再验证双seed加载、17件单次读取/哈希与改fanout、seed、head_hidden、learning_rate及缺字段负例。两份真实旧baseline JSON另经登记rawSHA及全字段参数核对，并用于config-only合成Torch序列化回归；没有打开真实checkpoint或校准NPZ。新负例首版误用setattr修改frozen dataclass导致测试本身失败，现改用dataclasses.replace；所有失败JUnit保留，未改生产dataclass。修复后的46项目标检查通过（39.71秒，0失败/错误/跳过，1项本机TypedStorage弃用警告）。修复质量与新源身份须重绑定，随后仅重跑同一CPU输入步骤；CUDA和science尚未放行。

## 固定范围与实现

[配置](../../../../configs/e2_frozen_recovery_pilot.json)固定原seed11/23 best768、原G_obs、原FP32权重与关系头、原完整valid及diagnostic_confirmation结构面板。独立旧校准分别为256/512次，p是原FP32完整L2的统一FP64提升/归一化，b是该p上的逐节点平均切空间误差。原件及所选数组的形状、dtype、SHA均已登记；半均值向量未保存，不伪造它们。原方差使用总体分母R，均值估计协方差迹为variance/(R−1)；half_cross和noise_corrected_bias_squared保留有符号值。

每模型16次新整图评价，分为[0,8]与[8,16]，每片只计算一次F，再对8次重复配对S/O/C/Q。PlanStreams独立种子2026091705，命名空间`E2-frozen-recovery-v1/seed{seed}/repeat{repeat}`，由全局重复ID独立派生两层实际计划。Q种子2026091706、模型固定命名空间，所有重复复用同一个q。归档保存实际计划的hash和完整派生元数据，可由原G_obs及冻结PlanStreams重建。C仅调用原模型两层third路径，共享S的实际计划，不接收p、b、q、F输出或结构标签；原保护规则与max_step=0.1不变。

F/S/C使用原生FP32点及原FP32头；结构诊断统一提升/归一化为FP64，并固定旧p的参考根。O/Q先在p上完成FP64几何，再一次转FP32。零幅度行直接保留原S原生点，不经过往返转换替换它。F原生点的提升须与旧p匹配既定容差，F完整valid的逐查询排名必须重现原best768记录。F的本次结构读数也由本次原生FP32 F统一提升得到。

新增模块复用原输入验证字节码、结构面板、完整过滤排名和无损NPZ/JSON归档：

- `recovery_inputs.py`：显式原件allowlist、已核bytes缓存与receipt；使用私有FunctionType globals注入内存I/O，旧模块globals不变。旧函数中的重复原bytes SHA调用取已计算hash对象的copy；canonical payload与解码数组仍独立核验。
- `recovery_registration.py`：精确配置、32个源文件LF、用户/质量/数值/阶段release绑定。存在Git时必须与声明HEAD及Git树源字节一致；无Git的部署归档如实记录caller-declared commit，靠质量LF表和外部包来源绑定，不冒充Git独立证明。
- `recovery_entry.py`：纯stdlib静态入口与进程监督。240/1080秒计时在张量worker导入之前开始；导入、release、原件核验/解码、前向、完整排名与保存全部在worker时限内。只结束自身直接张量worker；只读Git来源探针不持有GPU，不另启GPU/训练子进程。
- `frozen_recovery.py`：原环境CPU输入预检、40节点生产维度合成fixture、固定八重复科学片。每条件完整排名上限180秒，剩余总时限更小时服从剩余时限。科学执行只接受已review的CPU和CUDA门。
- `recovery_analysis.py`：检查全部片归档，重算原生点的结构读数并从完整保存排名核验MRR/Hits；不在汇总中再次执行全实体关系头。须完整四片、每seed全部16全局ID、稳定p/b/q/权重/F/面板身份一致，才汇总18项比较。每片approval、Slurm和时间元数据单独保留，不加入稳定身份相等门。

## 原件、数值与归档门

原环境CPU预检选定17原件：共享prepared五件，加每seed checkpoint、baseline、entry JSON/NPZ及fanout_4 JSON/NPZ六件。科学单片只选相应11件。总catalogue的历史release包等是外部部署/来源材料，不由worker当作额外数据打开。每选定原件在监督worker内只读取、完整SHA一次，从同一不可变缓存bytes解码；`worker_input_receipts`保存原件SHA、bytes、`original_path_reads=1`、`full_file_hash_checks=1`和`decoded_from_verified_cached_bytes=true`，并记录`all_bulk_inputs_verified_inside_worker=true`。新生成结果归档的独立读取/复核不属于旧原件加载范围。

转换数值规则在看结果之前由主管审阅，状态保持proposal/runtime pending：所有O/Q节点绝对geodesic转换误差≤1e−4；实际移动可分辨节点的相对误差≤1%。逐节点定义如下：

```
coordinate_scale = 1 + ||analytic FP64 O/Q point||_Euclidean
actual_step = d(promoted native S, analytic FP64 O/Q)
cast_error = d(analytic FP64 O/Q, uniformly promoted once-cast native O/Q)
resolution_proxy = max(old empirical floor, 8 * eps_float32 * coordinate_scale)
resolved = actual_step >= 100 * resolution_proxy
```

代理不是认证误差保证。保存所有节点的error、step、resolution、relative_error及resolved掩码/比例；不可分辨节点仍进入全部任务/结构读数。unknown及原覆盖分母保留，主结构没有覆盖时失败，不填零。原生转换、FP64参照、零幅度点/logit/排名、缓存score、完整排名、C共享计划、Q/RNG隔离和完整归档必须在原环境合成CUDA门通过。Fixture严格禁止所有真实输入参数；归档包含两seed的features、完整权重、邻接、p/b/q、实际两层计划、全原生点、FP64 F/S/C参照、所有40×40 direct/cached scores、完整rank rows和转换资料，供主管独立重算。

科学片保存entry中的p/b/q、权重、原生F、完整F排名/结构、校准不确定性、固定面板/参考视图与valid；每重复保存全四条件原生点、完整排名/结构、转换逐节点数组、聚合诊断、几何MSE/范数、随机流及耗时。所有NPZ数组无pickle，JSON清单逐数组SHA/shape/dtype可复核。仅完整归档后增加完成次数；中途失败保留已完成片段和当前阶段/重复ID，部分排名另存，不充当完整指标。不覆盖、续跑、重抽或随结果改转换/裁剪门。

## 统计与资源

主指标为父子顺序、远祖顺序、完整valid micro MRR；主比较C−S、O−S、O−Q，两模型共18项。保存所有逐重复差值、均值、MCSE与边际双侧Student t95%区间(df15)，全部18双侧检验做Holm。观测差值方差为零时保留均值，标`zero_observed_variance=true`、`inferable=false`、CI=null，Holm占位p=1；不以非零常数差造确定显著性。S−F作描述性采样损伤，Hits10、macro、几何和代价作辅助；这些MC区间条件于固定旧校准，不包含b估计噪声，也不是训练seed或独立test推断。

CPU预检独立命名：2核/16GiB、allocation300秒/worker240秒、0GPU。合成CUDA工程allocation300秒/worker240秒；四科学片每片allocation1200秒/worker1080秒，每worker单GPU，按需最多四片并行。GPU计划5100秒，含失败整轮上限5400秒。合法单GPU Slurm step可复用用户授权的CVPR四L40 allocation，实际绑定须仅有一个可见L40、保护原任务；不按父allocation四卡拒绝合法单卡step。证据记录job_id、step_id和可见绑定，ACL代价按实际step核算，不将整个CVPR allocation卡时计作ACL独占。资源放置与实际计费由实验任务负责；当前数值均为预算，不是消耗。

主管在结果前要求并冻结保守运行可行性门：验证并归档F完整排名后，以`F实测ranking秒数 × 32 + 180秒固定非排名余量`估算剩余工作。若超过当时剩余worker时间，以`resource_time_budget_inadequate`退出，保留F、实测与估算时间，零次S/O/C/Q新评价。独立卡与共享卡用同一规则；它仅判断运行预算，不判断数值质量或效果，不能据F效果选择数据。资源调整后仍用原全局重复ID、新输出目录，失败实际时长计入整轮5400秒；不扩大预留。输入加载分阶段耗时、科学phase_costs以及CUDA allocated/reserved峰值VRAM均记录，CPU峰值VRAM为null。

## 可复现入口与下一步

安装现有analysis依赖后，静态检查不导入torch/numpy、不打开真实输入、不申请资源：

```powershell
python -m acl_hct.recovery_entry --config configs/e2_frozen_recovery_pilot.json
python -m pytest tests/test_frozen_recovery.py -q --junitxml=.local/e2-frozen-recovery/feasibility-tests.xml
```

三执行阶段共用`--execute --config --phase --source-commit --approval-record --quality-record --output`。CPU与science必须显式提供`--prepared-root --checkpoint-root --training-release --reference-root --calibration-root`；fixture禁止这些参数。science还需`--seed 11或23 --repeat-start 0或8 --cpu-preflight-record --cuda-fixture-record`。输出分别为`cpu-preflight.json`、`cuda-fixture.json`、科学`run.json`与entry/repeat归档；每阶段另有`supervisor.json`、worker.log和失败证据。科学approval须绑定两门rawSHA及两项review_passed，所有phase须数值规则review。

完整四片后，仅CPU归档合并：

```powershell
python -m acl_hct.recovery_analysis --config configs/e2_frozen_recovery_pilot.json --shards SEED11_FIRST SEED11_SECOND SEED23_FIRST SEED23_SECOND --output NEW_MERGE_JSON
```

本次首轮合成检查1项失败、32项通过：FP64完整参照错误地只传一层邻接表，已改为两个完整层计划，失败JUnit保留。修复、补双seed全数组/门及注入等价检查后，41项通过，只有本机旧PyTorch的TypedStorage弃用警告。无失败被重标为通过、无数值阈值放宽。84个只读历史源/测试/配置/已验收适配器原件字节检查无变化，原4个未跟踪旧文件保留。

下一步为主管审阅精确源码/配置/质量并给出发布白名单；随后实验任务完成原环境CPU与纯合成CUDA检查，经独立数值review release后才执行四片。用户对同一有界联合pilot的授权已保留，不重复索要实验许可。O是末层诊断干预、C是两层局部修正，O不是C的严格上界；保持固定切空间中心残差不保证Exp后流形方差或任务表现保持。正、负、不可分辨或失败结果均须如实解释。
