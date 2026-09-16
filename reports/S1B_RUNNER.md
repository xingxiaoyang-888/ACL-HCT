# S1b 有界训练 runner 交接

2026-09-16。本轮接通冻结协议B的训练、完整验证选择和状态保存。仅用离线20实体fixture运行CPU测试；没有真实数据训练或GPU作业。WordNet真实预处理由实验任务在固定2ca02d6上执行，不能把本文件当作其完成证据。

## 执行合同

`python -m acl_hct.train` 接收已准备目录、单一JSON配置、空输出目录、device及可选完整source commit。runner核对entity顺序/G_obs/训练query/feature hash、验证边隔离、4:1负正比和固定split_seed20260914；从不打开evaluator_test.json或evaluator_truth.json。valid正关系只进入评估器，不用于构造train负例。validate在完整G_obs上编码，与训练batch屏蔽后的图分开。

每优化step独立均匀选一组正查询（batch_positives），取其预生成4负例；仅复制受影响邻接行，统一屏蔽全部正query及reverse，再生成两层采样计划。batch RNG与sampling RNG分开并连续使用。两层编码/统一头/BCE/优化完整执行；记录损失、实际正负数、候选/选中消息数、fallback/clipping、消息半径及95%边界饱和比例。

唯一pilot配置 `configs/wordnet_b_pilot.json`：训练seed11、hidden/head128、c1、scaled radius1.2、fanout16/16、原模型none、128正query/step、10 steps、单进程2threads、1800秒总预算。验证probe固定16个query（split seed20260914置换选取），保留全实体父候选；probe ID/hash写日志，**从不选best checkpoint**。它只测成本/开发监测，不是完整valid MRR或训练收敛结论。

`evaluation=full_validation` 时必须完整完成所有valid query才可按query-micro filtered all-parent MRR选best；同分保留更早checkpoint。固定child-macro/Hits/候选规模及逐query秩也输出。评分/编码分别计时，时间不足标incomplete，绝不冒充完整指标。缓存只在一次eval内有效；权重更新即失效。

硬上限max_steps<=2000、max_seconds<=3600；evaluation budget不能超过总预算。deadline在step/child边界检查，因此一次操作可能轻微超时，日志明确这一点。外层Slurm仍须设不超过1小时。程序不提交、取消或重试作业。CUDA入口要求显式Slurm allocation、保留CUDA_VISIBLE_DEVICES且仅一张可见卡；不改绑定，不选host-global ID。CUDA TF32关闭，无混合精度。

## 保存、恢复与来源

初始状态、每save_every步及finally保存last.pt，完整验证改进才保存best.pt；JSON和checkpoint均先写本目录临时文件再replace。记录模型、Adam、completed_steps、CPU torch RNG、batch/sampling RNG、CUDA可见设备RNG、输入/源码hash、配置、选择指标。失败写error状态并保留last；不自动恢复数值失败。

显式resume到新空目录，要求相同输入、验证标签hash、源码和device type；只允许延长max_steps/max_seconds，其余科学配置不变。有此前selected best时必须保留原best.pt同目录文件，核验并复制到新输出。失败checkpoint需先分析，不能静默跳过失败batch续训。CPU fixture证明连续两步与一步后恢复的权重、sampling RNG和loss精确一致；不据此承诺GPU散射梯度的跨硬件逐bit一致。

无git archive下可用 --source-commit 声明来源；git不可用不伪称clean，归档CLI测试实际运行保存结果。训练入口文件本身及所有导入项目源码hash进入报告，避免只记录几何依赖而漏入口。计时区分step、mask/sampling、full编码、ranking、checkpoint及总用时；GPU显存只称PyTorch allocated/reserved。

## 测试与下一步

runner初版3项通过；加入CUDA授权条件后runner+S1b+ranking共17 passed / 23.97s。最后补无git训练CLI入口hash回归，runner 5 passed / 29.77s。测试还覆盖不读取test/truth、probe不产生best、完整valid选择、时间耗尽仍保存、prepared输入篡改拒绝、未分配GPU入口拒绝。

```powershell
$env:PYTHONPATH="src"
python -m pytest tests/test_training_runner.py tests/test_benchmark_preparation.py tests/test_ranking.py -q
# 实验任务在已审查CPU预处理之后、固定release的单L40 Slurm分配内运行：
python -m acl_hct.train --prepared data/processed/wordnet-b-v1 --output runs/wordnet-b-pilot --config configs/wordnet_b_pilot.json --device cuda --source-commit FULL_RELEASE_SHA
```

尚需父任务验收真实预处理与此runner，再由实验任务测10步、16-query全候选probe及checkpoint成本。根据测量登记完整valid频率和有界开发配置，不能因validation慢悄悄用probe替代选择。之后才能基于有效selected checkpoint做E2/E3；当前没有真实层级坍缩/竞争力/训练改善证据。
