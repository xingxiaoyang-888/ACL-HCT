> 历史快照：以下状态、建议和授权只代表原文写作时点，不是当前执行指令。当前结论见 [报告总览](../../../README.md)。原始字节另存于归档 ZIP。

# E2原路线11节：多预算开发校准实现

研究问题：减少采样预算时，固定输入的局部几何偏移是否伴随祖先顺序/父检索变化，真实两层传播是否与局部观察一致？当前结论：新代码已实现预定开发设计，尚未取得本批真实科学结果；研究目的仍证据不足。工程交付目的为让数值及来源门槛、配对统计和成本记录先于运行。不能将本批准备或pilot的局部采样误差当作层级坍缩证据。

原路线11节覆盖项：多fanout、local L1/F-S/S-F/S-S、采样后直接/正远祖顺序、完整valid任务变化及MC精度。剩余缺口：正式确认、完整分支结构、模型竞争力、跨数据/骨干证据。原E1 D0与E3噪声/深度工作仍未补齐。本批无修正或N1。

## 固定设计与实现

协议：[E2_MULTIBUDGET_DEVELOPMENT.md](../../../../docs/operations/E2_MULTIBUDGET_DEVELOPMENT.md)。配置 `configs/e2_multibudget_development.json` 规范SHA为 `9476fd2cd6dd1527f22ba2538f4f2c29d8adc847ef6df74d47553fa1256e6ff7`。原两seed的best768、训练源及raw baseline哈希保持。fanout4/8/16/32/64，每档128次四条件配对；只有S/S预定前8次进行全valid、全候选任务评估。层间独立子流，namespace含checkpoint/fanout/层，基础seed2026091701。没有看结果追加重复。

`e2_pilot.run`仅增加内部可注入的固定配置检查器/诊断runner，复用来源和权重不变检查；原local CLI默认行为保留。批准记录scope与经过固定验证器的protocol一致。新流程重做入口数值/full valid复现检查，以及四路径full计划一致性，再开始预定预算。仍保持原数值阈值，未因扩范围放宽。

`RadialPanel`缓存关系/child/权重/固定组设计，每条件一次批量距离及CPU传输，用bincount计算child等权、组设计加权的指标。strict1/tie0.5/error0、covered分母、unknown、正确变错误/错误变正确均不变；新增明确数值不可解析比例。所有条件使用完整参考根。只有development池被评价，确认池仅冻结清单。

各层近边界组由full输出scaled radius≥1.14固定，不使用本批偏移挑分界。同时记录完整消息近边界比例；两者不是同一统计量。跨fanout的A/P变化须看各组清单/hash，固定V和度数组才能直接作同总体预算比较。

局部A外偏移按既定差值门槛核对后置数学零；S/F、S/S始终保存全节点实际Log偏移，不抹去A外传播。完整两层路径由既有paired接口共享相应计划，未重复生成条件专属采样。

## 归档与可独立重算的证据

新输出目录必须为空/不存在。每预算一份NPZ+JSON索引、入口一份NPZ+索引，`summary.json`只存来源、组级摘要、状态及文件hash。数组不经JSON往返：dtype、shape、数据SHA、NPZ整体SHA和manifest SHA均保存；`diagnostic_archive.read_archive`校验后无pickle重建。高维重复场不保存，最终均值/矩等充分统计保留；NPZ和详细manifest作为原始运行证据私有归档，不默认纳入Git。

每次repID、双层plan hash以及每条件每组MSE/固定径向投影/结构标量都保存；S/S前8次完整task的所有排名用整型ID矩阵及FP64 ranks保存。可独立重算组级MC SE和条件间配对差，不把节点×重复数作为独立样本。partial的已完成条件标量仍有记录，计数按方法列出；partial统计明确不可用于完成性或科学推断。

每fanout成功后独立原子落盘并释放流式缓存；失败停止后续预算并保存已完成和当前部分记录。外层硬终止时以最后完整落盘的预算为准，不以当前进度猜测完成。没有自动resume或条件替换路径；修复工程问题后的继续由主管依用户E2续行授权协调。

## 使用与CUDA门槛

静态清单：`python -m acl_hct.e2_development --config configs/e2_multibudget_development.json`。

新源码先在独立单GPU Slurm分配运行工程小样：`python -m acl_hct.e2_development --cuda-fixture --source-commit <固定commit> --output-dir <全新目录>`。仅40节点，fanout4/16、各4次、R_task1，内部120秒；覆盖新四路径、结构统计、归档和数值门槛。输出 `cuda-quality.json` 供主管核验。本地CPU测试不能替代该CUDA通过记录。

正式开发使用 `--execute --config ... --seed 11或23 --source-commit ... --prepared ... --checkpoint ... --training-release ... --baseline-report <原训练run.json> --approval-record ... --cuda-fixture-record <已验收cuda-quality.json> --output-dir <全新目录>`。必须指定单个 `--fanout 4/8/16/32/64` 分片，仍每档128/8，summary明确shard_only并保留完整计划清单。

批准记录沿用用户/固定source/config/quality/entry字段，scope为E2-multibudget-development-v1，另须cuda_fixture_passed=true及cuda_fixture_artifact_sha256。入口实际读取CUDA报告并校验raw hash、通过状态、相同source commit及逐文件源码hash，不只相信布尔声明。必须先完成上一批独立验收和本批主管审核；不由代码任务申请GPU或运行科学配置。

## 成本、限制与工程验证

固定十个(seed,fanout)分片，每片L40一张、2CPU/16GiB，最多两片并行；上限外层3600秒、内部3300秒，单完整valid180秒。仅5×8次task按已有约25秒估计约1000秒，另640次四条件/结构统计，不能由旧16次局部试跑推断全部能在3300秒完成。主管已固定十片安排。40节点CUDA小样仅验新增路径，不能外推82115节点耗时；实际分片测量成本，不延长硬跑或减少科学预算。

82115×129 FP64场约80.8MiB。四个TangentStream的base/mean/双half约1.26GiB，加共同参考/方向、原生trace、临时输出及统计约数GiB级GPU张量；单预算最终高维均值原始量约323MiB。压缩率/CPU峰值/实际耗时未预写，运行记录峰值及forward/geometry-structure/task分项时间。逐预算释放，不把全部预算的高维tensor累积在报告中。

必要CPU工程检查包括：向量结构与原逐关系独立实现一致（多父、ties、unknown、非等权、空组）；配对四路径/传播A外保留；不同fanout分母；低维重复重算MC SE与条件差；全部任务候选；中途数值/任务超时停止；NPZ精确重建/篡改拒绝；固定配置/CUDA来源门槛；受影响旧入口来源与批准检查。此前版本无Git归档10项通过/21.44秒；新增分片约束及CLI状态映射后做最终针对性归档检查，**13 passed /22.80秒**，无跳过。仅PyTorch旧TypedStorage兼容警告；完整命令及最终六文件规范LF哈希见 [e2-multibudget-quality-tests.json](../../../e2-multibudget-quality-tests.json)。未执行真实数据诊断或CUDA小样。

CLI完成请求分片退出0；failed/incomplete先保存全部已有证据再退出2，入口异常非零，避免Slurm COMPLETED掩盖失败。完整十片和科学目的另行验收。
