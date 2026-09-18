> 历史快照：以下状态、建议和授权只代表原文写作时点，不是当前执行指令。当前结论见 [报告总览](../../../README.md)。原始字节另存于归档 ZIP。

# 固定权重诊断入口：历史批次元数据修复

研究问题：两次真实固定权重诊断为何在科学观测前退出，能否恢复原登记入口？**原因已定位为历史MLP记录格式解析错误，尚无新的模型机制结论。** 两个作业均在历史文件的seed核验阶段失败，科学前向/反向及plain等价次数均为0。工程恢复已准备并通过离线回归；真实诊断目的尚未达到，不把退出算作已执行72+8矩阵。

原发布 `cd47f921b6fc1cfc5f47015e2f35f03f8f68bcc2` 的解析器优先选择 `history.config`，假定其中有单次运行的 `seed`。实际两份已发布MLP文件的config是包含 `seeds=[11,23]` 的总登记，单次运行身份记录在 `training_settings.seed` 及顶层 `seed`。两份文件原始SHA均匹配既定登记；在原函数的独立元数据复现中，两者均得到 `batch history configuration seed mismatch`。失败来自工程假设，不是数据身份或科学效应不通过。

修复仅改变新诊断输入模块及其测试：同一设置解析函数显式识别两seed登记与单次运行设置，返回经过核对的设置供loader与replay共同使用。所有出现的单次seed必须是整数并与指定seed一致；多seed登记必须包含它且列表有效，缺少单次设置/顶层身份立即拒绝。登记、config和单次设置中的batch_positives若冲突也拒绝。旧单seed config或training_settings格式继续通过，不静默换seed、换批次或选一个冲突记录。

固定科学配置、原checkpoint、原训练源码、769/770批、128×5监督、两层连续采样、72次科学加8次plain、等价容差、720秒入口及两卡各15分钟上限均不变。输入文件SHA校验和逐步历史batch hash门槛保留。本轮未读取真实prepared/checkpoint，未进行真实抽批流重放、模型前向/反向、训练或GPU；仅核对已有历史run JSON元数据，以及CPU人工小图回归。

测试命令：

```bash
python -m pytest tests/test_backbone_frozen_inputs.py -q
python -m pytest tests/test_backbone_frozen_diagnosis.py tests/test_backbone_frozen_inputs.py tests/test_backbone_frozen_readings.py -q
```

输入模块46项通过，新增两seed真实schema最小抽批回归及12种冲突/缺项拒绝检查；原32项输入与历史hash门槛保留。完整工程回归 **72 passed，0失败/错误/跳过，约31.07秒**，语法编译及diff空白检查通过；结果和新源码绑定见 [修复质量记录](../../../e2-backbone-frozen-input-repair-quality.json)。已有两份真实MLP的元数据另已通过新parser，正确解析seed11/23和batch_positives=128；没有跨版本冒充原2.5.1抽批重放。

旧 [工程质量记录](../../../e2-backbone-frozen-engineering-quality.json) 及主管复核保留其旧版hash，不覆盖或改称已核验修复代码。修复使用独立质量记录与新的source绑定；主管复核及准确发布版本登记后，由实验任务继续用户已批准的同一方案。真实失败证据和资源记录由实验任务保存，不删除失败或扩展实验。
