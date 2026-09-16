# STATUS — 2026-09-16

- 实现：Lorentz FP32/64、完整/采样等权聚合、确定性无放回采样、三阶候选及有限总体 jackknife 比较接口、小型 GNN、合成树诊断和官方数据解析已实现。
- 本地测试：Python 3.10.11 / torch 2.0.1+cpu，19 passed in 24.53s。含零点/重合点梯度、有限差分 gradcheck、FP64/FP32 参考、采样性质、NumPy 子集枚举对照、解析/泄漏、小规模复现。
- 远程部署：ACL 独占目录与 Python 环境已建立。用户授权只读复用 CVPR 安装包，正在离线安装 torch 2.5.1+cu124 至 ACL；原环境不修改。环境已成功安装且 pip check 通过，Python 3.11.7 / torch 2.5.1+cu124 / numpy 1.26.4 / pytest 8.4.2。
- 数据：Princeton WordNet 3.0 与 NLM MeSH 2026 官方压缩文件已在本机下载成功并上传；服务器直连超时，未绕过官方来源。等待调度 CPU 审计。
- 调度：已提交首个 CPU 作业（2核、8GB、15分钟上限），私有 JobID 记录在 `.local/jobs.tsv`。GPU 未提交。
- 异常：SFTP 递归首次上传因目标子目录 canonicalization 失败，改为显式源码 tar 包传输。ACL 自己的 cu118 下载被主动停止，改用已下载的 cu124 wheels。
- 科学结论：仍无真实网络证据；合成测试通过不等于候选修正有效或层级坍缩成立。
- 下一步：完成数据和环境准备，受控 CPU 检查，再单 L40 smoke，保存实际资源与结果，审查后发布。

- 本地新增重合原点修正反向传播测试通过（累计 20 项）；8 步/方法的本地 CPU smoke 已完成，结果见 `reports/local-smoke.json`。非对称邻域枚举中修正降低偏差但提高 MSE，不作为方法全面有效的证据。

- 调度异常：初次脚本执行因 CRLF 换行失败，未创建作业；已转换 LF。Git 属性已约束 shell/Slurm 文件 LF。

- 远程 CPU 首次运行：20 passed in 10.59s；WordNet 82,115 节点/84,427 边、SHA256 与本机一致。MeSH 官方文件触发重复树位置校验，作业 FAILED/1，未进入训练 smoke。正在核查数据并补测试，GPU 暂停提交。
