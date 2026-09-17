# E2 固定权重诊断：离线工程验收

研究问题：在两个原 best768 骨干上，batch mask 与 f16 分别如何改变评分和梯度，第二层径向集中是否伴随局部压缩和实际梯度变化？**本轮只完成独立入口及 CPU 人工小图验证，真实机制问题仍无新证据。** 工程准备目的达到；真实诊断未执行，其科学目的尚待批准后的原数据读数。

对应协议为 [E2_BACKBONE_FROZEN_DIAGNOSIS_PROPOSAL.md](../docs/operations/E2_BACKBONE_FROZEN_DIAGNOSIS_PROPOSAL.md)。原训练源码、checkpoint 和科学结果不改写。本轮仅按主管核定清单归档离线工程与已完成第一阶段报告；归档不批准真实诊断。

## 交付与固定行为

- 独立入口 `python -m acl_hct.backbone_frozen_entry`：标准库监护进程启动唯一 worker；从启动起，imports、核验、科学矩阵、等价检查与归档共用720秒。截止时只终止自身 worker。记录实测截止/回收延迟；操作系统调度并非实时系统保证。
- 固定配置 `configs/e2_backbone_frozen_diagnosis.json`，规范 JSON SHA256 `38ae7ad55156fd2b9922eff424460c5961f9d7e43dab8135bd5a23d117e7f784`。固定两 seed、best768、769/770批、128正例×5、8重复、full/f16、mask两条件；每 seed 36次科学前向/反向，另4次 plain 等价，共72+8。第769批四个等价条件先通过，才运行其余重复。
- `backbone_frozen_inputs.py` 仅打开 manifest、observed graph、train queries、features；不读取 valid/test/truth。valid身份由冻结登记及checkpoint元数据绑定。原CPU seed+1抽批流逐步复建并对历史MLP批次hash；不匹配立即停止。
- `backbone_frozen_diagnosis.py` 调用原 `encode/score`，仅在两层 Linear 添加不替换输出的 hooks。每次清空旧梯度，平均BCE backward，记录 `dL/du`；无optimizer、训练更新、ranking或新权重保存。每条件及全作业核对权重/输入身份，最终重新核验允许的输入文件和原checkpoint。
- `backbone_frozen_readings.py` 保留逐节点低维读数与逐query差；公开固定度数组、有界参数/层摘要、全部重复统计及文件hash。parent/child出现次数与unique分母分开；五条记录按正例所属组分配。分位数固定为0/.25/.5/.75/.9/.95/.99/1，标准差使用ddof=1，方向按精确正/负/零。读数与汇总在CPU提升为FP64，原模型计算保持FP32。
- `s=0` 方向分量在私有数组中使用NaN及显式defined掩码，公开使用undefined计数/null，不把虚构零值纳入摘要；非方向读数与实际梯度均须有限。缺组保留0计数/null。
- 每条件增量保存私有NPZ及hash manifest，再写原子summary。非零退出/超时保存失败清单、已归档条件、缺失矩阵及active phase；详细异常留在私有worker.log，部分结果不冒充完成。

## 运行门槛与使用

生产worker在任何prepared/checkpoint打开前，必须核对用户授权引用、协议scope、固定配置hash、诊断source commit/LF源码hash、quality文件hash及主管质量复核字段。生产只允许原PyTorch2.5.1+cu124、独立单L40 Slurm分配、FP32且关闭TF32。CLI不提供更换batch/seed/重复数/半径/容差/预算或跳过审批的选项。

批准并发布后，实验任务在隔离的分配内使用以下入口；所有占位路径与记录由实验任务填充，当前没有生成实际授权记录：

```bash
python -m acl_hct.backbone_frozen_entry \
  --config configs/e2_backbone_frozen_diagnosis.json \
  --seed 11 --prepared-root PREPARED --checkpoint ORIGINAL_BEST \
  --training-release ORIGINAL_RELEASE --batch-history HISTORICAL_MLP_RUN \
  --approval-record REVIEWED_USER_APPROVAL --quality-record REVIEWED_QUALITY \
  --source-commit PUBLISHED_DIAGNOSTIC_COMMIT --output NEW_EMPTY_OUTPUT
```

`quality-record`使用机器验收记录，`approval-record`须包含 `user_authorized=true`、`user_message_reference`、`scope=E2-BACKBONE-FROZEN-v1`、`config_sha256`、`source_commit`、`source_lf_sha256`、`quality_record_sha256`、`quality_review_accepted=true`、`entry_criteria_frozen=true`。这些字段是运行时核验要求，不是本报告授予用户同意。

## CPU验证与审查修复

可复现命令：

```bash
python -m pytest tests/test_backbone_frozen_diagnosis.py tests/test_backbone_frozen_inputs.py tests/test_backbone_frozen_readings.py -q
```

覆盖人工22节点图上FP64/FP32四路径 hooks 及全部参数梯度等价、梯度清空、空邻域自身回退、双向mask、SRSWOR及同一generator连续两层、解析Jacobian、s=0方向、原始配对差与交互、分组分母、输入契约和历史hash失败、完整两seed小图矩阵、partial留证、监护硬超时及源码/配置/审批/strict load拒绝。完整小图矩阵沿相同工程代码运行，但节点、特征、标签、构造权重均为人工；不是原科学矩阵72次已执行。

最终 **58 passed，0失败/错误/跳过，约30.02秒**；四个新模块语法编译通过。机器记录见 [e2-backbone-frozen-engineering-quality.json](e2-backbone-frozen-engineering-quality.json)，私有JUnit证据位于 `.local/e2-backbone-frozen-engineering-final-tests.xml`。完整摘要保留条件记录，频繁阶段更新使用独立小型 `progress.json`，避免反复序列化全部已完成摘要。核验过程中另一个扩大到旧checkpoint测试的命令已主动停止：旧fixture会调用optimizer/训练，不适用于本轮限制；仅结束自有pytest进程，当时仍在首个人工矩阵内，未执行旧训练fixture。

初审曾发现两处代码风险，均在真实执行前修复并加回归：原checkpoint配置的tuple与JSON的list直接比较会误拒合法输入，现比较规范化digest；矩阵循环覆盖训练seed会使采样种子错误，现区分training_seed和plan_seed，并逐格核对两seed/两批/八重复/两图的登记公式。超时留证、四个前置等价检查顺序和执行计数语义也已按独立审查补齐。

本地环境为PyTorch2.0.1+cpu。真实2.5.1/L40数值等价、数据/权重实际核验、耗时与显存均未测量；这不影响离线工程交付，但不能称真实门槛已通过。初始化报告另已补齐原2.5.1 CPU构造审核，旧2.0.1初始化hash不匹配历史保留。下一步由主管复核具体工程交付，再等待用户同意这一关键诊断。
