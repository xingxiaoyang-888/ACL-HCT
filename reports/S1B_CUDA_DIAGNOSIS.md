# S1b CUDA失败：只读对比与下一次最小验证方案

两次pilot均未进入训练；初始只读排查之后，监督任务批准并完成一次独立节点隔离诊断，最小CUDA验证通过，未启动训练。当前根因未定位。比较依据是ACL自有原始脚本、E0结果、两次错误日志、当前ACL环境包清单，以及调度器只读状态；没有连接未分配计算节点或读取CVPR私有内容。

## 已确认的差异与一致项

| 项目 | 成功的E0 | 两次pilot失败 |
|---|---|---|
| 固定计算源码 | 9e610cbc16acfeff0d0790c63a0a11d28c4b9cb6 | fce017259a6bdc513f21b31304f096239e357a75 |
| Python命令路径 | 同一个ACL独立.venv/bin/python | 同左，均为显式绝对路径 |
| Python/torch/CUDA runtime | 3.11.7 / 2.5.1+cu124 / 12.4 | 当前ACL环境对应同版本；失败栈明确来自此.venv的torch |
| 分配节点 | 成功节点 | 两次均为另一个相同节点 |
| 分区与GPU请求 | L40，gpu:l40:1 | 相同 |
| NVIDIA驱动 | 580.65.06 | 相同 |
| nvidia-smi总显存 | 49,140 MiB | 46,068 MiB |
| GPU执行上下文 | srun内实际CUDA回归78检查及8步smoke通过 | 第一次batch初始化失败；第二次srun内初始化失败 |
| 主机资源/上限 | 2CPU / 8GiB / 30分钟 | 2CPU / 16GiB / 60分钟 |
| OMP_NUM_THREADS | 2 | 2 |
| OPENBLAS_NUM_THREADS、MKL_NUM_THREADS | 提交脚本未显式设置 | 均显式设置2 |
| PYTHONPATH | 当前固定release/src绝对路径 | 当前固定release内的相对src |
| CUDA_VISIBLE_DEVICES | 脚本未覆盖，实际值未记录；可见设备数1 | 脚本未覆盖；第二次实际记录为0，可见设备数1 |
| LD_LIBRARY_PATH | 未记录，脚本未修改 | 未记录，脚本未修改 |

当前解释器链接解析到共享Python3.11，ACL venv配置为include-system-site-packages=false；当前torch/version.py与E0记录一致。比较E0 pip freeze与当前ACL pip freeze：没有移除或改变已有版本，仅新增joblib1.5.2、scikit-learn1.5.2、scipy1.14.1、threadpoolctl3.6.0。当前快照不是失败时点的完整环境镜像；失败脚本的pip freeze位于初始化检查之后，未执行。E0未保存实际sys.executable/torch.__file__，因此历史路径一致性依据提交命令、版本和后续失败栈，不能称所有二进制逐字节一致。

两个失败的完整错误栈均在torch.cuda.get_device_properties → _lazy_init → torch._C._cuda_init处报 `RuntimeError: No CUDA GPUs are available`。第二次device_count断言通过之后初始化失败，小张量检查没有机会执行。无训练模型导入或真实数据运算参与失败点。

调度器当前显示失败节点仍为MIXED，并记录在第二次失败之后有SlurmdStartTime更新；这只是查询时点的状态，不能解释历史失败原因，也不证明已经修复。两个分区当时都up且存在idle节点，不代表本账户立即可用，也不自动授权改为A800。总显存差异值得保留，但不能据此推断ECC、硬件损坏或设备映射原因。

## 已审阅的最小验证方案（下文保留原计划）

1. 使用ACL独立诊断脚本，先记录脚本SHA256；申请单张L40、2CPU、4GiB、最多5分钟，只诊断，不自动调用训练。优先排除连续失败节点，以调度器分配另一张L40作为节点隔离对照；保留Slurm原始绑定，不指定主机全局GPU号。
2. 在同一个srun步骤内按白名单记录SLURM_JOB_ID、SLURM_STEP_ID、SLURM_JOB_GPUS、SLURM_STEP_GPUS、CUDA_VISIBLE_DEVICES、CUDA_DEVICE_ORDER、LD_LIBRARY_PATH、LD_PRELOAD、PYTHONPATH、CONDA_PREFIX及线程变量。私有日志保存路径和节点名，公开报告仅保留比较结论，不整份dump环境。
3. 记录sys.executable及realpath、sys.prefix、torch.__file__、torch版本、torch.version.cuda、torch._C路径与关键文件SHA256。记录nvidia-smi的设备UUID、PCI地址、驱动与总显存，并读取本进程cgroup及可见设备文件权限；不修改设备、权限或驱动。
4. 用相互独立的子进程诊断CUDA driver初始化返回码和错误名、PyTorch device_count/is_available、get_device_properties，以及4元素CUDA张量求和并synchronize。错误保留完整栈；小张量结果应为4。读取/proc/self/maps中实际加载的libcuda/libcudart路径，有助区分已安装包与实际加载库。每个子检查设短超时，失败不丢后续诊断记录。
5. 若另一L40通过，只能说明该分配下当前环境可执行CUDA，支持节点/分配条件相关的假设，仍不宣称根因已定位。然后单独决定是否重新执行原10步pilot；保持fce0172源码、2ca02d6 prepared数据与原配置。
6. 若仍失败，先审查绑定、库路径、driver错误码等证据；必要时再讨论单张A800最多5分钟的硬件隔离对照。A800诊断不是L40性能证据，不能混入原pilot测速。不得自动连续重提或更改共享CUDA环境。

当前10步planning/total、full编码、16-query排名、峰值训练显存均不可提供；两次合计10GPU秒仅为分配成本。完整两次经过及四个pilot交接文件见S1B_PILOT.md，本文件为第五个公开交接文件。私有原始对比保存在.local/pilot/readonly-comparison.txt、environment-current.txt及两次evidence目录。

## 单次节点隔离诊断实测结果

监督任务明确批准后，按上述资源边界排除连续失败节点，分配到另一台L40节点（也不同于历史E0节点）。诊断脚本固定SHA256为 `a8811e2c6e6243e89b8b2e48c17ad5995ceafc5083ed993d1ba08c2a1b1865a5`，原文与launcher保留在私有目录，上传前后哈希一致。没有运行训练源码或改变其配置。

- 单L40、2CPU、4GiB、5分钟上限；COMPLETED/0:0，实际分配11秒，0.003056 GPU小时，队列核对已释放。
- nvidia-smi：NVIDIA L40、驱动550.54.14、总显存46,068 MiB。PyTorch total_memory为47,576,711,168 bytes，compute capability8.9。
- `cuInit(0)` 返回0，错误名CUDA_SUCCESS；driver API版本12040，driver device count=1。
- PyTorch device_count=1、is_available=true、get_device_properties成功。
- CUDA四元素全1张量求和=4.0，synchronize通过；此最小检查的peak allocated=1,024 bytes、reserved=2,097,152 bytes，**不是训练显存测量**。
- 五个独立子检查全部退出0，无超时、无stderr。诊断Python总墙钟10.325725秒；各子进程包含启动/import/文件hash等开销，不能当成GPU内核时间。
- 同一ACL解释器与torch路径、Python3.11.7/torch2.5.1+cu124/CUDA12.4实测确认。LD_LIBRARY_PATH、LD_PRELOAD、CONDA_PREFIX均未设置；CUDA_VISIBLE_DEVICES由Slurm传入0，未修改。实际libcuda来自系统550.54.14驱动，libcudart来自ACL venv。具体路径、设备UUID、cgroup和权限信息仅保存在私有原始JSON。

两次失败加此次诊断累计21GPU秒，即0.005833 GPU小时；此前E0与其他作业不计入这个小计。本次诊断自身占11GPU秒，必须与两次失败的10GPU秒分开。

可确认的结论是：当前ACL环境在另一L40分配上可以初始化并执行CUDA，支持先前失败与节点/分配条件有关的假设。新节点驱动也不同，不能隔离节点因素与驱动因素，更不能断言580.65.06不兼容——历史E0已在该驱动版本成功。新节点与失败节点的nvidia-smi显存总量相同，故总显存差异本身不足以解释失败。

下一步可在单独的新Slurm分配中执行原10步pilot，并继续排除连续失败节点；先验小张量通过才进入训练。但本次授权仅限诊断，**未自动提交训练或A800对照**。本次机器记录为 `reports/s1b-cuda-diagnostic.json`；连同此前五文件共六个公开交接文件，历史失败JSON保持不变。
