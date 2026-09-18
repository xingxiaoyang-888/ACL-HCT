> 历史快照：以下状态、建议和授权只代表原文写作时点，不是当前执行指令。当前结论见 [报告总览](../../../README.md)。原始字节另存于归档 ZIP。

# S1b 真实10步成本pilot：恢复运行完成，历史失败保留

**原配置10步恢复运行及16-query全候选probe已完成。** 先前两次训练前CUDA初始化失败与一次成功的节点隔离诊断均单独保留。恢复运行占107GPU秒；四次分配累计128GPU秒（0.035556 GPU小时）。尚未执行完整验证或选择best checkpoint。

## 固定来源与CPU门槛

计算源码：`fce017259a6bdc513f21b31304f096239e357a75`，git archive独立release、所有归档文件哈希核对通过。数据复用 `2ca02d6033fd5503b82bbcd5b72c5f172b07eb6e` 的WordNet预处理，没有重新划分或拟合特征。

配置严格为该commit中的 `configs/wordnet_b_pilot.json`：seed11、hidden/head128、c1、scaled radius1.2、fanout16/16、method=none、128正query/步、10步、内部1800秒、16-query全候选validation_probe、每5步保存；配置及hash在GPU状态JSON中完整保留。没有参数修改。

Slurm CPU兼容作业2核/8GB，15秒，COMPLETED/0:0。训练runner与排名9项fixture通过（9.61秒），5个torch.load未来默认行为警告来自本次生成的测试checkpoint，不是失败。真实prepared输入通过加载、图/特征/query hash及验证边隔离检查：82,115节点、128维、67,539训练正query组、8,456 valid查询。导入源码哈希与固定归档一致，来源为caller_declared，git.available=false/dirty=null。

prepared input manifest hash：`b5fc4847260854e5401ffe0d5111f32b7d1138f1de5d11c881e2d4c312c25473`。CPU测试用小fixture训练，不构成真实WordNet训练。

## 历史：两次GPU分配与最小失败

用户在首次失败后明确授权第二次申请。两次均分配1张L40、2CPU、16GB主机内存，外层1小时上限；固定源码、数据及原配置不变，第二次使用独立输出目录。两次均在调用训练入口之前终止。

| 项目 | 首次 | 用户授权的第二次 |
|---|---|---|
| CUDA预检查上下文 | batch | srun步骤内 |
| nvidia-smi | 成功 | srun内成功 |
| torch.device_count()==1 | 通过 | 通过 |
| get_device_properties(0) | CUDA初始化失败 | CUDA初始化失败 |
| 最小CUDA张量计算 | 未安排 | 已安排，但在前置初始化处失败，未执行 |
| Slurm最终状态 | FAILED / 1:0 | FAILED / 1:0 |
| 分配占用 | 7秒 | 3秒 |
| 分配卡时 | 0.001944小时 | 0.000833小时 |
| 真实训练步数 | 0 | 0 |

两次nvidia-smi均列出NVIDIA L40、驱动580.65.06、总显存46,068 MiB，且分配落在同一节点。第二次Slurm传入CUDA_VISIBLE_DEVICES=0，未覆盖该值。两次错误均为 `RuntimeError: No CUDA GPUs are available`，发生于PyTorch的CUDA初始化。

将预检查移入srun没有解决问题；**根因尚未定位**。nvidia-smi能列出设备不等于CUDA运行时可用，device_count断言通过也不代表CUDA内核可执行。当前证据不能判定为模型问题、OOM、驱动版本不兼容，或断言整个集群不可用。同一节点的重复失败支持下一步针对分配设备访问和运行时初始化做有界诊断，但尚不构成节点故障的证明。

两次累计分配占用10 GPU秒，即0.002778 GPU小时；这是资源占用，不是训练计算耗时。最终队列查询确认第二次作业已退出；两次独立训练输出目录均未创建。截至两次失败收尾时没有第三次提交；后续诊断及恢复运行另行批准，见下文。没有改变全局CUDA设置、操作其他作业或降低训练配置。

## 历史失败时未获得的指标与边界

| 项目 | 两次结果 |
|---|---|
| 10步planning/total | 无；训练入口未调用 |
| full编码与16-query排名耗时 | 无；未执行评估 |
| loss/梯度有限性 | 未测量 |
| 裁剪与半径饱和比例 | 未测量 |
| PyTorch训练显存峰值 | 未测量，JSON保留null |
| checkpoint保存耗时 | 未测量 |
| last checkpoint / best checkpoint | 均不存在 |

因此不能用10秒累计分配占用推算单步训练或完整valid成本，也无法据此选择模型效率修复或完整valid预算。应先解决已分配GPU上的CUDA初始化。CPU兼容门槛及数据预处理仍然有效；当时真实训练成本、任务质量和正式checkpoint尚未取得；下文补充随后完成的恢复测速。16-query probe原定只用于开发测速，不能选best或冒充完整验证。

## 交接文件

- `reports/S1B_PILOT.md`：两次经过与总成本。
- `reports/s1b-pilot-cpu.json`：CPU fixture、真实输入加载及源码身份。
- `reports/s1b-pilot-gpu.json`：首次分配的原始历史状态，保持原文；其中retry_submitted=false描述首次收尾时点。
- `reports/s1b-pilot-gpu-attempt2.json`：用户授权第二次分配、失败与两次累计资源用量。

首次原始证据保存在 `.local/pilot/evidence.tar`，第二次在 `.local/pilot/attempt2-evidence/`；提交脚本与分配记录独立保留。公开报告省略JobID、节点名和私有连接信息。本任务未git add/commit/push。

## 节点隔离诊断与恢复运行

单次诊断在另一L40节点通过（驱动550.54.14），占11GPU秒，详见S1B_CUDA_DIAGNOSIS.md与s1b-cuda-diagnostic.json。随后监督任务批准单独恢复训练：继续排除连续失败节点，使用新的独立输出目录，固定计算源码fce0172、prepared2ca02d6及原10步配置全部不变，不从失败或诊断过程恢复任何模型权重。

恢复运行分配到又一L40节点，驱动实测580.65.06、nvidia-smi显存46,068MiB，与诊断节点驱动不同。相同srun步骤内先CUDA张量求和与同步通过，再执行训练入口；未覆盖绑定。源码文件、导入哈希、配置及prepared manifest/valid query哈希均与固定归档/既有审核相符。caller_declared与git.available=false/dirty=null按实际记录，不伪称远端Git checkout。

Slurm单L40/2CPU/16GiB、外层1小时，实际107秒，COMPLETED/0:0，资源已释放。训练进程GNU time墙钟97.69秒，runner内部95.604374秒；两者计时边界不同。host MaxRSS1,337,484KiB。此次没有失败、OOM或重试。

## 逐步成本与数值记录

| 步 | 采样与查询屏蔽秒 | 步总秒 | BCE loss |
|---|---:|---:|---:|
| 1 | 1.0033 | 8.9356 | 0.709603 |
| 2 | 0.9469 | 8.2119 | 0.668182 |
| 3 | 0.9030 | 8.2046 | 0.641073 |
| 4 | 0.8972 | 8.2238 | 0.611372 |
| 5 | 0.9964 | 8.2969 | 0.579900 |
| 6 | 0.9194 | 8.2295 | 0.550004 |
| 7 | 0.9063 | 8.2073 | 0.524562 |
| 8 | 0.8988 | 8.2052 | 0.507576 |
| 9 | 0.9020 | 8.2176 | 0.500489 |
| 10 | 0.9109 | 8.2391 | 0.502788 |

10步总计82.971482秒，平均8.297148秒；规划总计9.284146秒，平均0.928415秒，占步骤总时间11.19%。扣除规划后平均7.368734秒，包含传输、编码、打分、反向、优化与诊断，**不是纯GPU kernel时间**。首步较慢，后9步平均8.226214秒。

所有10步loss有限；固定runner每步backward后检查所有参数梯度存在且有限，通过后才更新，10步完成意味着这些检查均通过。未记录梯度范数，不能补造。loss从0.709603降到0.502788，第9步0.500489后略回升；各步训练batch不同，这不证明收敛或任务质量。1:4类别比例也使较低BCE本身不足以证明关系恢复能力。

两层全部步骤clipping_rate=0、near_bound_fraction=0；近边界阈值为0.95×1.2=1.14。第10步消息平均/最大scaled radius：第一层0.669928/0.828969，第二层1.081155/1.096258。第二层半径向约束边界靠近，但本次未触发近边界阈值，需在后续开发中继续记录，不能将其解释为采样层级坍缩。method=none时fallback表示空邻域自身回退，各步比例约16.3295%–16.3466%，不是候选修正失败。

PyTorch peak allocated=1,718,813,184bytes（1.601GiB），reserved=1,845,493,760bytes（1.719GiB）；覆盖runner计算与评估的分配器峰值，不是整GPU进程占用。没有修改TF32/AMP协议：FP32、TF32关闭、无混合精度；deterministic_algorithms=false按实记录。

## Probe、保存与后续预算边界

完整G_obs两层编码耗时6.873559秒；固定16查询/16子实体的全候选排名耗时0.075340秒。每条过滤后候选82,114，probe完成16/16，query-micro与child-macro MRR均为0.000126894153，Hits@1/3/10均0。这是validation_probe，不是8,456条完整验证查询结果；未用于挑选best。

4次checkpoint保存（初始、step5、step10、finally）合计0.035668秒，均保存last。远端last.pt存在、1,017,194bytes，SHA256为038fd4bc72c71dd544da48453e24349dd9018035b662cf2e5c2799cb9615638a；按冻结runner保存模型、优化器、RNG与来源，未额外加载验证内部内容。best.pt不存在，best_full_valid_mrr=null；不能把last称为完整验证选择的基线。

当前成本主要位于每步全图传播/反向等非规划部分；仅优化规划最多直接影响所测步骤约11.2%的墙钟，还需要剖析其余部分才归因具体瓶颈。16-query评分过短，仅粗略按8,210个valid子实体线性扩展得到约38.7秒排名，加本次编码约45.5秒；这只是未校准的预算参考，包含固定缓存成本且硬件/负载会变，不能代替完整valid测量。下一步应由监督与代码任务据此选择有界完整验证或效率剖析，不自动扩大矩阵。

## 分开记录卡时与最终交接

| 分配 | GPU秒 | GPU小时 |
|---|---:|---:|
| 首次初始化失败 | 7 | 0.001944 |
| 用户授权第二次失败 | 3 | 0.000833 |
| 节点隔离诊断 | 11 | 0.003056 |
| 本次恢复pilot | 107 | 0.029722 |
| 本轮累计 | 128 | 0.035556 |

最终交接共8文件：本报告、s1b-pilot-cpu.json、s1b-pilot-gpu.json、s1b-pilot-gpu-attempt2.json、S1B_CUDA_DIAGNOSIS.md、s1b-cuda-diagnostic.json、s1b-pilot-run.json（原始run逐字节复制）、s1b-pilot-run-metadata.json。均位于reports；历史失败JSON未改。原始日志、作业标识与路径、固定launcher、checkpoint保留在ACL私有目录，公开文件通过标识隐私检查；本任务未stage/commit/push。
