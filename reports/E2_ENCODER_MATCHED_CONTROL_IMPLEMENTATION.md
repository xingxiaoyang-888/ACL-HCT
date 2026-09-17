# E2 匹配编码器对照：独立工程交付

研究问题：保留原径向编码器和关系头时，将两层邻居聚合替换为自身消息，能否改变原有界任务的父节点检索能力？**独立实现与本机CPU工程验收目的达到：新全实体路径、唯一查询实体路径与原网络全空邻域路径的输出、平均BCE及全部参数梯度通过比较；尚无本次真实训练结果，任务能力及图路径作用仍证据不足。** 该辅助对照服务于可信骨干基础，不能代替“采样偏移—层级损伤—轻量修正”的机制证据。原E2科学目的仍部分达到。

本轮用户授权与主管协议见 [匹配编码器协议](../docs/operations/E2_MATCHED_ENCODER_CONTROL.md) 和 [主线决策](../docs/research/03_主线证据与下一步决策.md)。本交付未提交Git、未申请GPU、未打开真实prepared或检查点、未进行真实模型训练；实际2.5.1预检和CUDA门等待主管接收固定版本后由实验任务执行。

## 实现与边界

- `encoder_matched.py` 的 `SelfMessageLorentzNetwork` 继承原构造器和关系评分。保留两层Linear、原平滑径向映射、exp/log、参数名和82433个参数；仅将聚合输出替换为本节点消息。原空邻域最终返回self message，未添加编码器ReLU、残差、归一化或dropout。
- `encode(features)` 返回按实体原顺序排列的Lorentz点。`forward(features, queries, unique_entities=True)` 只编码唯一查询实体，再用inverse索引回映所有重复查询；`unique_entities=False` 是全实体检查路径。两种路径的GEMM行数可能改变FP32舍入，不能假设逐比特相同，必须通过原登记容差。
- `encoder_matched_control.py` 负责允许输入、历史schema、逐步batch hash、Adam训练、完整过滤排名、首次最高选模、初始快照及best重载。图仅用于输入契约与固定degree分组，不进入新编码器。旧GNN及MLP是已完成参照，不重训。
- `encoder_matched_registration.py` 是仅依赖标准库的登记和来源门。配置canonical SHA、原科学依赖、全部13个相关源码LF SHA、主管接受的quality原字节SHA和明确用户授权必须对应同一版本；未通过时不能读取prepared标签。
- `encoder_matched_entry.py` 在独立worker导入torch之前建立进程截止；CPU预检/CUDA小图门为240秒，科学worker为840秒。仅终止它启动的直接worker。保留日志、完整/部分结果及失败清单，无resume、覆盖失败或重试逻辑。OS调度和回收延迟单列测量，不承诺实时系统语义。

旧科学源码及原4个未跟踪文件的原字节未改。源码继续复用已修复的只读训练输入模块，不修改上一轮故障/质量/科学证据。

## 固定科学登记

配置 [e2_encoder_matched_control.json](../configs/e2_encoder_matched_control.json) 的canonical SHA为 `e07cb6244dcf6a25b25f9a463e90e1f953c3245a1578674799ae4010c16f691d`。

最初工程草稿SHA `2f3d44b3ccf76f4e46bfdcf7e862c65d5ef319653cc3db12a80f7ba97873065e` 到最终SHA的实质变化，仅为加入执行方已核对的原 `features.npz` 和 `evaluator_valid.json` 原字节SHA。它加强来源绑定；模型、两seed、训练步数、学习率、批次、选模机会、资源和容差均未改变。原三个train JSON、特征内容、valid canonical hash和参照历史raw hash同时核验。

原prepared、128维特征、seed11/23、128正例组×每正4负、平均BCE、Adam .003、1024步均固定。每一步独立CPU `randperm` 使用seed+1，与原MLP历史的全部1024个索引hash对齐，先核对后更新。第256/512/768/1024步各一次完整8456查询valid、全部实体候选、原过滤和精确同分平均排名；只有严格更高的micro MRR替换best。父degree0、1、2–16、>16按原未遮蔽图固定，保留空分组、查询数及整体分母贡献。

科学完成还要求从保存的best重新构造/加载模型，核对权重、源码、配置、数据与选模元数据，并重新计算一次完整valid。逐查询rank及micro/macro MRR、Hits和degree读数必须与所选步骤一致。该复现不增加选模机会；不是只对旧排名重新求均值。

真实训练先私有保存 `initial.pt`（完整实际初始state和逐张量SHA），再保存last/best、逐步抽批hash、完整排名与重载排名。原GNN历史没有初始快照；原环境构造与历史MLP对应初始化hash是核对依据，不能称为与旧GNN初始快照直接比较。

## CPU验收与尚待执行的门

必要检查覆盖生产维度/两个seed构造随机流和全部参数名、FP32/FP64三路输出/损失/梯度、流形及零输入梯度、重复/乱序索引回映、完整过滤排名与同分、缓存失效、严格原生标签及允许文件读取、真实历史GNN/MLP日志schema、初始快照、四次选模与保存后新模型排名、批次hash/非有限/超时/不完整valid/重载错误停止，以及来源/授权/缺失CPU或CUDA证据拒绝。

首轮40项检查通过后，补入原字节绑定和门槛完整性检查；冻结版本最终为 **50 passed / 15.66秒**，0失败、0跳过，1项PyTorch TypedStorage弃用警告。命令为 `python -m pytest tests/test_encoder_matched_control.py -q --basetemp=.local/e2-encoder-matched/pytest-freeze --junitxml=.local/e2-encoder-matched/frozen-tests.xml`；完整版本绑定与JUnit原字节SHA见 [机器质量记录](e2-encoder-matched-control-quality.json)。最后修正了进度日志在更新/验证/重载/存档各阶段的刷新，避免硬截止时沿用较早的完成步数。本机Python3.10.11、PyTorch2.0.1+cpu；这些工程结果不替代原2.5.1环境的初始化及抽批核验，也不能冒充CUDA门通过。

CUDA工程门需要独立L40分配，FP32原空图/新全实体/新唯一实体的points、logits、loss和全部参数梯度固定 `atol=2e-6, rtol=2e-5`；缓存绝对差门为1e-5。两个seed共130个原生数组私有归档：输入/查询/标签与回映、三路points/logits/loss、三路完整权重和参数梯度、直接/缓存全对分数及缓存排名；manifest绑定每数组dtype/shape/SHA。所有23个比较键必须齐全，空集合、缺项、放宽门槛或归档篡改拒绝。主管可用保存权重和输入在原空图模型CPU独立重算。

## 交接接口

静态检查无需torch导入：

```text
python -m acl_hct.encoder_matched_entry --config configs/e2_encoder_matched_control.json
```

真正执行统一加入 `--execute --phase PHASE --source-commit SOURCE40 --approval-record APPROVAL --quality-record QUALITY --output NEW_OUTPUT`。

- `PHASE=cpu_preflight`：增加 `--prepared-root PREPARED --reference-root REFERENCES`，不指定seed；一次检查两seed、原输入与参照、原历史初始化hash及全部1024个batch hash，无forward/backward或optimizer。
- `PHASE=cuda_fixture`：不提供真实prepared/参照或seed路径；一次归档两seed小图三路完整数值证据，输出 `cuda-quality.json`、`seed11/fixture.json/.npz` 和 `seed23/fixture.json/.npz`。
- `PHASE=science`：增加原输入/参照根、`--seed 11` 或23，以及 `--cpu-preflight-record FILE --cuda-fixture-record FILE`。参照根包含原字节不变的 `baseline-seed11/23-run.json`、`e2-capacity-seed11/23-run.json`。

审批记录必须含 `user_authorized`、非空 `user_message_reference`、`scope`、`execution_phase`、`source_commit`、`config_sha256`、`quality_record_sha256`、`quality_review_passed`、`entry_criteria_frozen` 和 `supervisor_released`。科学phase还要求 `cpu_preflight_review_passed`、`cuda_fixture_review_passed` 和两门的 `*_artifact_sha256`。门记录、完成的supervisor结果及完整数组绑定均核对，不能只提交一个passed标记。

只允许原2.5.1+cu124；GPU必须为独立Slurm分配内唯一可见NVIDIA L40，保留绑定、FP32、禁用AMP/TF32。实际资源仍未测量：工程分配300秒/worker240秒，两科学分配各900秒/worker840秒，总预留2100秒；整轮包含失败的2160秒上限由主管与实验任务的allocation账本核验。单次完整valid含编码不超过180秒。

下一步是主管独立接收源码/质量快照、发布固定版本并交实验任务做真实输入及CUDA门；通过后仅运行原登记的两个匹配对照。该工程交付未启动E3或追加其他训练。
