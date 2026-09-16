# 代码质量加固 — 2026-09-16

本轮完成实现、离线 CPU 回归和有上限的集成验证；没有提交 GPU 作业、修改服务器或真实数据训练。交付前 fetch 确认起点 HEAD/origin/main 均为 `948810d44cd77b5871f36862440fde2c5006d2c0`。父监督任务已有的 AGENTS.md 编辑权说明保留并纳入提交。本报告对应同次提交的源码；规范化 LF 的源码/测试 SHA256、依赖版本和测试结果在 `code-quality-validation.json`，避免把历史 GPU 结果归于新代码。

## 发现与修复

| 问题 | 证据与处理 |
|---|---|
| 非有限 max_step 契约不完整 | 首轮 `inf` 未拒绝；`nan` 原本通过下游检查报错。现入口统一要求有限正数，未声称 nan 曾静默成功 |
| 标量 point、缺失 generator | 首轮分别出现 IndexError、AttributeError；现在给出明确 ValueError，拒绝 bool 作为计数/曲率 |
| 坐标轴与 batch 广播混淆 | dot 统一核对坐标维度、dtype、device，仅 batch 维允许广播。加强后的有效点维度测试又发现 log 在校验前减法导致 RuntimeError，已前移检查；没有放宽测试 |
| WordNet 截断记录 | 首轮缺失指针计数产生 IndexError；现在检查词表/指针长度、负数和重复 synset，错误含源行号。保留 noun @/@i、重复边去重及多父关系语义 |
| 下载中断与证据时间 | 新下载使用可续传 .part，curl 成功且非空后才 rename；已有最终文件复用。失败/续传/成功发布由离线 mock 检验；不是本轮实际网络下载证据。audit 时间单独标识，不伪称下载时间；既有文件的 retrieval 时间仍不可恢复 |
| 诊断粒度与同步 | 旧 correct 返回 Python 标量且 batch 只汇总 any/max。保留该参考接口；新接口输出每节点设备张量，零步长报告真实 0，稳定反向仍保留范数保护 |

没有确认 FP32/FP64 重合点 NaN 缺陷。新增非原点重合/近重合反向检查均有限；不是全工作域或所有硬件的数值稳定性证明。

## 批量契约与接入

`correct_batched(sample, N, mask, c, method, max_step, self_points)` 接受 `[..., K, D]`，K>=1；布尔 mask 匹配 `[..., K]`；整数 N 广播到 batch 形状。每行实际 k 由 mask 计数，非空要求 1<=k<=N。空邻域必须 N=0 并提供 self_points，原样返回。masked padding 在几何运算前替换，包括 NaN padding；其输入梯度为零。

第三矩和 cached-sum jackknife 公式不变，仅按 mask 做逐行统计。k=N、k<3 保持逐邻域未修正结果；不读取未采样表示或层级标签。返回 detached 设备张量 fallback/empty/full/small_sample/clipped/raw_step/step/k/N，原因标记允许重叠。none 的空行也记 fallback；原参考接口不直接支持空行。

TinyGNN 的 `batched=True` 为可选路径，默认仍是参考路径。两条路径使用同一 CPU generator、相同排序去重及无放回抽样序列；空邻域不消耗随机数。新 smoke `--batched` 仅每步取回两个汇总计数，不逐节点取回 GPU 统计。输入验证仍有固定数量的 batch 级同步；采样/索引组装仍有 Python 循环。

## 正确性证据

最终 `python -m pytest -q`：**78 passed / 36.37s**。Python 3.10.11、PyTorch 2.0.1+cpu；其余版本见 validation JSON。测试默认离线，无真实数据下载。

- 原有 NumPy 独立枚举参考、对称/非对称邻域、采样频率/确定性/无重复、完整采样、泄漏路径检查保留通过。
- 新批量对照独立调用逐邻域 correct，覆盖 FP32/FP64、c=0.1/1/4、混合 N/k、k=1/2、完整样本、mask 空洞、NaN 填充及裁剪；检查值、输入梯度、fallback/clipped 和 padding 梯度。
- FP64 gradcheck 使用有限差分验证第三矩及 jackknife；小模型参数梯度与两步训练损失对照通过。没有把所有分布上偏差或 MSE 改善设为断言。
- 非原点重合/近重合、batch 广播、维度/计数/mask/邻居索引非法输入，以及 WordNet 多父/instance-hypernym/截断 fixture 和下载失败 mock 已覆盖。
- 40 节点、三种方法各八步批量 CPU smoke 已完成，见 `code-quality-cpu-smoke.json`；修正回退率 70%，裁剪率 0%。仅合成训练诊断，无 held-out 性能含义。

失败轨迹：首批契约测试在旧代码上 4 failed/2 passed；修复后完整 27 passed。加入批量和数据测试后 72 passed/14.10s。加强维度测试后 1 failed/77 passed，修正 log 校验顺序后最终 78 passed。独立 smoke 首次未设置 PYTHONPATH，ModuleNotFoundError；设置为 src 后成功，最终源码再运行成功。没有以改容差或硬编码输出消除失败。

## 效率证据（单独解释）

`code-quality-cpu-benchmark.json` 是串行 CPU 微基准：2 threads、32 行、k=8、N=16、4 spatial dims、FP64，含 from_spatial、前向/反向、验证和诊断，每路径一次预热、三次测量。最终测量在其他本任务测试/作业结束后执行。

| 方法 | 参考中位秒 | batch 中位秒 |
|---|---:|---:|
| none | 0.040589 | 0.006539 |
| third | 0.211398 | 0.010775 |
| jackknife | 0.205136 | 0.011160 |

这只说明这组固定 CPU 输入下减少 Python/校验调用的效果。参考路径包含逐行标量诊断，新路径保留设备诊断；没有把它解释为纯算术加速。三次测量、固定执行顺序、普通桌面环境不能支持正式效率或 GPU 结论。batch padding 的内存是 O(B*Kmax*D)，不等于稀疏图可扩展性证据。

## 可复现命令

在仓库代码根目录执行（PowerShell）：

```powershell
$env:PYTHONPATH="src"
python -m pytest -q
python -m acl_hct.smoke --device cpu --batched --output reports/code-quality-cpu-smoke.json
python scripts/benchmark_aggregation.py --output reports/code-quality-cpu-benchmark.json
```

微基准硬限制 nodes<=128、repeats<=10，CPU only；smoke 原有 steps<=30 上限保留。不要为重现本报告提交 GPU 作业。

## 剩余限制

- 新代码尚未远程 CPU/CUDA 回归。历史 21 项远程测试与 24 秒 L40 smoke 仅适用于原报告的 abd3d43，不是这次修改的证据；后续短时远程回归由父任务协调。
- 原点缩放半径<=3 的工程域未放宽；高维/高曲率模型参数仍可能越界并报错。距离在重合点不可微，未改其数学定义。
- 候选修正仅为固定、局部、等权、均匀无放回采样假设下的局部算子。原非对称例子降低偏差但增加 MSE 的事实仍保留；不主张通用理论或训练收益。
- MeSH 歧义隔离和小图 split 协议未改变；本轮未重新审计完整官方原始数据，也没有正式大图划分/真实层级坍缩证据。
- 下载 mock 不模拟真实服务器 Range 支持；失败留 .part，现有最终文件不自动覆盖。旧版留下的半成品需显式改名为 .part 后再续传；archive/audit 仍决定文件是否可用，未引入可信上游签名。
