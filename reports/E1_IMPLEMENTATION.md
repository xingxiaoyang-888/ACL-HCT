# E1 实现交接 — 2026-09-16

本阶段发布可运行的受控机制工具。仅执行了 `configs/e1_cpu_check.json` 的12个小案例（11个精确枚举、1个128次MC），全部状态ok；`configs/e1_s1.json` 的54个定向条件已冻结，但正式S1组尚未执行，交实验任务按固定commit运行。本地没有GPU作业或真实训练。

## 数学与统计契约

`src/acl_hct/mechanisms.py` 使用FP64，构造明确的二维或更高维有限总体，支持N/k/d/c/尺度/对称/非对称/反向及有限域内Lorentz boost。对称构型成对；非对称构型固定改变一侧尺度后中心化；反向条件只反转空间方向。生成器不因结果或越域重采样。

所有方法复用同一索引流，记录SHA256；小组合数阈值由配置预先固定，超过后固定MC次数，按chunk流式累积。正式配置阈值20000、MC备用4096、chunk256，N16/d2/c1下固定k=1/2/3/4/8/16、尺度0.05/0.15/0.7与三种构型。不是曲率/维度等全部参数的笛卡尔积。配置上限128个case、N128、单MC100000、总邻域2M；执行工具默认CPU，设备切换需实验任务按既定预算调度。

完整参照p固定。记录 `E[Log_p(Y)]` 的向量估计、范数、方差、MSE、半径变化、预测方向投影及余弦、配对MSE/半径差、步长、裁剪及回退。同点距离的不可微性不影响本工具的no_grad冻结统计。二阶oracle预测独立遵循既有NumPy参考；局部三阶oracle向量另存，均只用于诊断，不传给部署修正器。

MC报告均值向量的ambient协方差/MC标准误和标量标准误。**MC均值范数含噪声，不能称为精确偏差**；另报告扣除均值估计噪声的平方偏差估计，它允许为负，不裁成零。精确枚举MC标准误为0，不表示真实训练或模型不确定性为0。偏移范数不提供伪造的对称置信区间。图中标量误差条为1.96标准误的正态近似显示，非严格有限样本保证。

对照为完整参照、none、third_unclipped、third_protected、jackknife_protected。unclipped诊断通过1e100有限保护阈值实现，并断言无裁剪；不是把0.1版本当作未裁剪公式。所有方法保持现有局部公式；非法/越域显式记录，任何失败方法不输出基于剩余样本的选择性均值。工程域仍为原点缩放半径<=3。simple scaling和匹配零均值噪声明确列为尚未实现控制项。

## 正确性与小案例证据

- 新增机制测试5项通过：NumPy二阶方向、未裁剪第三矩枚举对照；对称/反向；chunk不改变MC索引流；独立协方差/平方偏差计算；全采样/低k回退；boost保持MSE；越域状态。
- 机制+既有几何/批量测试共58 passed / 9.16s。公共覆盖率脚本整理后机制+覆盖摘要测试6项通过（实际耗时见本次终端记录）。没有修改E0正在回归的几何/聚合/模型核心。
- 五点非对称独立枚举回归明确断言：偏差降低而MSE增加，避免选择性只报告偏差。
- 12个小案例JSON为 `e1-cpu-check.json`，记录原始配置、commit起点、dirty标记及规范化源码哈希；它在E1提交前运行，不能把其source_commit起点误认为纯9e610cb代码。
- 三类静态图在 `e1-cpu-figures/`；可只用结果JSON重建。方向图、偏差/方差/MSE图已目视检查，未裁图或手改数值。曲率4/维度4与非原点MC是定向正确性小例，不是正式参数扫描。

## 重现与正式组交接

```powershell
$env:PYTHONPATH="src"
python -m pytest tests/test_mechanisms.py tests/test_coverage_summary.py -q
python -m acl_hct.mechanisms --config configs/e1_cpu_check.json --output logs/e1-check.json
python scripts/plot_e1.py --input logs/e1-check.json --output-dir logs/e1-figures
# 以下正式组由实验任务在固定提交的受控资源执行
python -m acl_hct.mechanisms --config configs/e1_s1.json --output logs/e1-s1.json
python scripts/plot_e1.py --input logs/e1-s1.json --output-dir logs/e1-s1-figures
```

绘图可选依赖 `pip install -e '.[plots]'`，本机matplotlib3.10.8。全部单测离线。本工具没有自动调度/重试/参数扩展。

## 原图覆盖率证据的独立来源

父监督任务提供并批准发布 `S1_SAMPLING_COVERAGE.md` 与 `s1-sampling-coverage.json`；公共 `scripts/audit_sampling_coverage.py` 保留固定9e610cb解析器、原文件hash和结构核验，新增CLI本地路径与来源字段，摘要函数fixture通过，未重复全量审计。两个数据只接收父邻居且fanout>=8时没有采样删减；双向fanout8的直接影响节点比例分别为2.783%与4.593%。这不是机制无效证据；正式协议B训练图需重新审计，不能隐式增边或把等价full配置当独立实验。

## 继续工作

E1正式组尚待父任务交接实验任务；S1b接着补文本/路径schema、按子节点划分的协议B、训练实体文本拟合TF-IDF/SVD和两层原Lorentz-Mean/统一关系头。旧小图reachability helper不会冒充正式B。E2以后依赖有效协议与验证选择checkpoint；本阶段不能声称已建立真实层级坍缩或通用修正理论。


## 正式运行前补丁

父任务在任何正式S1结果观察前，按组合数和预期统计精度批准把正式配置枚举阈值从2048提高到20000。54条件几何参数完全不变，全部精确枚举，共138483个邻域（原计划59517的约2.33倍），低于2M预算；小尺度偏差O(scale^3)可能被4096次MC的O(scale/sqrt(R))误差淹没，这是预先提高统计精度的原因。保留12案例历史CPU输出和MC路径单测，不把其结果重标成正式精确组。

修复归档部署缺口：此前CLI末尾无条件调用git，在无.git的release计算后会失败。现CLI接受 `--source-commit FULL_SHA`，来源校验在运行前执行；没有git时明确记录git.available=false、commit/dirty=null，并将传入revision标为caller_declared，绝不冒充已验证clean checkout。没有声明时source_commit=null仍保存结果。可用源码仓库与声明不一致时在计算前报错。另记录规范化配置SHA256和所有实际导入项目源码模块的规范化LF SHA256（包含geometry/aggregation/__init__/mechanisms），独立于调用者声明。

回归在临时无git目录复制release源码，真实subprocess执行CLI保存小案例，再在该目录从JSON运行plot CLI生成三图；也核对正式配置54条件全部精确且总预算138483。归档运行命令增加 `--source-commit <本补丁完整提交SHA>`。E0批准发布的5个文件保持固定9e610cb历史来源，不能作为本补丁的远程验证。
