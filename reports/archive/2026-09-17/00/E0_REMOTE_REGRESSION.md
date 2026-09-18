> 历史快照：以下状态、建议和授权只代表原文写作时点，不是当前执行指令。当前结论见 [报告总览](../../../README.md)。原始字节另存于归档 ZIP。

# E0 远程回归验收

状态：**通过，资源已释放；专属文件等待父任务审查后由代码任务统一发布。** 本任务未执行 git add、commit 或 push，未修改几何公式及其他任务文件。

## 版本与环境

- 固定源码：`9e610cbc16acfeff0d0790c63a0a11d28c4b9cb6`。使用 `git archive` 导出独立 release，没有打包活动工作区。所有被导入 src 文件的 SHA256 已与本地固定归档逐一一致核对。
- 源码归档 SHA256：`6e16e890bcbebc9f745cb508c401ab0cb40aeea8ebce1ef670e68b9abb016dde`。
- 补充入口 SHA256：`f1e9e3692caa39b2b7362cb9d79325fed24212063bcbf7687d5debf3cf4b9b14`。入口部署在单独、按内容哈希命名的 harness 目录，没有覆盖 release。测试文件哈希也记录在 JSON。
- 远程沿用 ACL 独立 Python 3.11.7、torch 2.5.1+cu124、NumPy 1.26.4、pytest 8.4.2；pip check 通过。GPU 实测 NVIDIA L40，计算能力8.9，驱动580.65.06，单个可见设备，nvidia-smi 总显存49,140 MiB。这里是本轮实测环境，不能套用历史 smoke 的驱动/显存记录。
- 无新下载、全数据审计、真实数据训练或原图覆盖率重复计算；未操作 CVPR 环境/作业，未覆盖 Slurm GPU绑定。

## 实際运行证据

| 项目 | 结果 | 用时与资源 |
|---|---|---|
| 固定 release 原始 CPU pytest | 78 passed，无失败/跳过 | 测试11.71秒，Slurm占用15秒；2核、8GB、无GPU |
| 补充入口自身本地测试 | 9 passed | 7.53秒，Python3.10.11 / torch2.0.1+cpu |
| 补充入口本地全量预检 | 78个检查及三方法8步batched smoke通过 | 仅作入口预检，不冒充远程CUDA结果 |
| 真正 CUDA 回归 | 78个检查通过 | 检查4.745秒；张量实际位于cuda，拒绝自动退回CPU |
| CUDA batched smoke | none / third / jackknife均8步通过 | 各0.598 / 0.558 / 0.560秒，含小图评估 |
| GPU整段入口 / Slurm占用 | passed / COMPLETED，退出码0:0 | 入口8.807秒 / 作业16秒；1张L40、2核、8GB、30分钟上限 |

本轮GPU占用16秒，即0.004444 GPU小时。最终队列核对为空；JobID与原始调度、设备日志保存在 `.local/e0/`，公开报告不暴露服务器连接信息。没有重复提交或失败GPU作业。

## 检查口径

CUDA的78个检查与CPU的78项pytest是**不同集合**：CUDA入口包含36个混合邻域对照、36个非原点重合/近重合检查、6个模型批量/逐行参数梯度对照。每类均覆盖FP32和FP64、none/third/jackknife；几何案例覆盖c=0.1/1/4。

- 混合行实际k为 `[0,1,2,3,4,5,3,4]`，N为 `[0,7,7,3,8,9,6,4]`；包括mask空洞与NaN padding，验证无效位置梯度严格为零。
- 完整、低k和空行输出与同批原聚合严格一致；空行严格返回显式self。逐行和批量输出/梯度对照；同一CPU生成的FP64输入复制到各设备，CPU FP64逐行路径作为独立精度参考。
- 步长上限0.1与1e-5均覆盖；记录实际clipped/fallback行数，不通过改公式或删失败条件取得通过。
- 非原点重合与1e-7近重合包含Log/Exp重构、流形/切空间约束和有限反向梯度；模型检查比较批量/逐行参数梯度。
- 8步smoke采用固定源码现有batched入口。修正方法回退率70%、截断率0%。这不是竞争性基线、真实任务性能或吞吐量基准。

与CPU FP64参考相比，混合邻域FP32最大坐标绝对误差为2.762e-6，最大输入梯度绝对误差为3.312e-5，均来自jackknife、c=0.1、步长上限0.1；FP64两者均为4.441e-16。所有数值通过**事先固定的绝对+相对容差**：FP32输出atol=3e-6、梯度atol=2e-5、rtol=2e-4；FP64 atol=2e-11、rtol=2e-9。梯度最大绝对误差大于单独的atol，但满足组合判据；没有运行后放宽容差。逐案例误差与阈值均保留在JSON中。

检查段PyTorch峰值allocated为18,168,320 bytes、reserved为23,068,672 bytes；三方法smoke最大allocated为18,307,584 bytes（约17.46 MiB）、reserved为23,068,672 bytes（22 MiB）。这是PyTorch分配器统计，**不是整进程/整卡显存**；smoke会按方法重置峰值。计时包含验证同步、CPU参考和小图评估，不支持效率优越性结论。

## 证据与复现

- `reports/e0-remote-cpu.json`：固定源码、原始套件计数、环境、CPU作业状态与占用。
- `reports/e0-remote-gpu.json`：源码逐文件哈希、入口哈希、实际设备、78个检查的逐项误差、8步batched结果、显存、计时与最终状态。
- `scripts/remote_regression.py`、`tests/test_remote_regression.py`：本任务专属补充入口及离线测试。
- `.local/e0/`：固定归档、脚本、JobID映射、提交前后资源记录、原始JUnit/日志和不可变结果副本。

在固定release中先运行 `python -m pytest -q -p no:cacheprovider`。随后在已获批的单GPU分配中设置 `PYTHONPATH=/path/to/fixed-release/src`，运行：

```bash
python /path/to/harness/scripts/remote_regression.py --device cuda --source-root /path/to/fixed-release --output /path/to/results/e0.json
```

入口校验实际导入路径、记录所有源文件及入口哈希；调用者还需像本轮一样将这些哈希与固定git归档核对。缺少CUDA或可见GPU数量不是1时失败，不会静默转CPU。资源参数和Slurm原始命令保存在私有记录。

本机Windows tar曾对归档中文文档路径发出警告；本地预检改用Python仅导出固定归档中的src/scripts/tests，并与远程逐文件哈希核对。远程Linux解包未遇到此问题。该操作不修改源码，未影响回归数据。

## 交接与科学边界

E0就绪，可接收父任务指定的E1固定commit；本轮不自行扩展扫描或真实训练。原图覆盖率审计由父任务负责，不重复执行。真实数据仍未训练；历史合成案例中bias降低但MSE上升的边界结论保持不变。测试通过仅说明本轮有限案例在声明工作域和容差内一致，不证明普遍数值稳定性或层级坍缩假设。

本轮恰好交付五个专属文件，供父任务审查及代码任务显式stage：上述两个入口文件、两个JSON和本报告。其他源文件、测试、README、配置和STATUS均未修改。
