# E2 径向/正交分量：最小实现与工程验收

2026-09-17。研究问题：旧独立均值移除 O 为什么改善检索却恶化父子顺序，移除不同方向分量能否解释该分歧？当前结论是工程实现和合成 CPU 门通过；B 的真实 CUDA 数值门与原四个科学分片尚未执行。科学目的仍为 **insufficient evidence**，不能把工程通过写成层级恢复。

## 固定范围

仅新增 R（径向均值移除）与 T（正交切向均值移除），复用旧 source `293ca0beaa50dd9bedacf6f06001ec3b73b2b9b6` 的两模型、原32份 S、原 p/b/floor、固定面板/根、节点和验证查询、冻结权重。新配置为 `configs/e2_radial_component_pilot.json`，canonical SHA 为 `576b9d2984b2ce10c5e3c0bae07a570fcf95e57c10db6b76aca16740e25d351d`。旧32模块与已发布A模块保持不变。

FP64投影与结构评价固定在CPU，原FP32关系头在CUDA。合成门另保存完整CUDA FP64参考数组与CPU参考对照。科学路径不加载特征或checkpoint，不构建GNN，不重forward、不采样、不训练、不重估校准。关系头采用meta构造，加载原四个头参数，复用原score与cached ranking实现；科学路径不消耗新的CPU/CUDA随机流。新配置的O/Q文字说明已在新结果前改成R/T，数值阈值保持旧值。

`radial_components.py` 在固定p处取向外单位切向量，使用Lorentz内积投影。根/近根方向不可分辨时，两个新条件均直接保留S，保存未施加b。分解与范数恒等只用于方向可分辨节点。该几何公式采用标准Lorentz log/inner定义，可参照 [Geoopt Lorentz math](https://github.com/geoopt/geoopt/blob/master/geoopt/manifolds/lorentz/math.py)；本实现的测试另用NumPy闭式投影复算。

有限T不保证全局半径不变，R/T效果不要求加起来等于O。科学档案还保存相对S的实际径向变化与固定根切空间中的方向夹角；任一半径低于固定数值门时，角度用NaN数组值和显式mask标记，仍保留全部主要评价节点。acos的[-1,1]舍入保护仅用于辅助角度，不改变任何干预。

## 必要本地验收

- 数学层15项通过（独立NumPy投影、切向/单位/正交/范数恒等、纯分量、零场、根/近根/阈值边界、原一次FP32转换门和非法输入）。数学源与这一批证据沿用已审查原件。
- 集成33项通过，13.87秒：独立NumPy头打分与Exp复算、四组合成完整数组、meta头不消耗RNG、F逐行/候选数量核对、转换/新点/原S身份漂移拒绝、四个八重复合成存档链路、F后预算不足零R/T退出、固定18项统计与零方差约定、源码/质量/配置/阶段/用户授权门、独立fixture验收门。
- 四片端到端测试使用N=16的工程存档，并显式mock原校准数据入口；完整真实校准/节点/查询门保持原A实现。测试输出保留 `engineering_fixture_only=true`，不能进入科学CPU合并。真实L40/完整82115节点验收尚待主管放行，未把测试替代科学证据。

复现命令：`python -m pytest tests/test_radial_recovery.py -q`。数学层已通过的命令为 `python -m pytest tests/test_radial_components.py -q`。测试不联网、不下载、不用GPU。机器可读质量原件为 `reports/e2-radial-component-quality.json`；私有JUnit原件单独保存。

## 执行门和交接

入口 `python -m acl_hct.radial_entry` 默认只输出静态门槛；实际执行必须指定 `--execute --phase cuda_fixture|science`，由stdlib supervisor先启动240/840秒总worker期限，再进入tensor导入、源码/发布/输入验证、打分与写档。允许4个独立已放行科学片；运维先检查用户授权CVPR分配的安全独立ACL step容量，保持Slurm绑定，本地代码不申请GPU。

两阶段共同参数：`--config`新B配置、`--protocol`当前协议、`--approval`主管阶段放行原件、`--quality`本质量原件、`--source-commit`已发布40位commit、`--output`新空输出。fixture禁止旧输入与片号。science另需 `--old-config`原物理配置、`--bindings`原私有清单、`--shards`原四根目录（清单顺序）、`--seed 11|23 --repeat-start 0|8 --fixture-record`完整fixture run与 `--fixture-review`主管独立复算原件。

放行schema：`status=approved_B_cuda_fixture|approved_B_science`，`protocol=E2-radial-components-v1/B`、匹配 `execution_phase/source_commit/config_canonical_sha256/new_source_lf_sha256`，`user_authorized/quality_review_passed/supervisor_released=true`、`GPU_requested=1`，并绑定 `quality_raw_sha256/config_raw_sha256/protocol_lf_sha256`。science再绑定 `bindings_raw_sha256/fixture_raw_sha256/fixture_review_raw_sha256`。独立fixture review必须为 `independent_fixture_review_passed`，匹配fixture原始hash、commit、config与4个新模块hash；不能用worker自检代替。公开质量通过不自行释放科学任务。

每科学片先检查四片small JSON与原32源，再仅读取本片原完整NPZ。用保存native_F重放所有查询：完整rows、顺序、rank与candidate必须逐行完全一致。保存F后计算 `16×完整F实测秒数+180`；大于剩余worker时间则保留F、零R/T、资源失败退出。科学条件/样本ID不改变。单次完整排名上限180秒；不完整或错误排名归档为失败，不进入合并。

随后每个原S生成R/T，沿用绝对1e-4、resolved相对1%、零分量精确native复制门。新档案仅保存新点、排名、结构、转换和有限移动辅助，引用旧档案而不复制旧4.53GB。完整archive reader校验NPZ字节与全部数组shape/dtype/data SHA；一次仅驻留原一个repeat。片末验证冻结头/输入/RNG/零梯度，8条完整观测才可标记complete。

科学成功后，在原CPU环境执行 `python -m acl_hct.radial_analysis --config B_CONFIG --old-config OLD_CONFIG --bindings BINDINGS --original-shards OLD_ROOT_1 OLD_ROOT_2 OLD_ROOT_3 OLD_ROOT_4 --shards NEW_ROOT_1 NEW_ROOT_2 NEW_ROOT_3 NEW_ROOT_4 --output NEW_SUMMARY`。这次CPU读档应由运维放入适当有界CPU allocation；不是登录节点计算。它验证新点与完整cast数组由同一个原S逐位复现、组件/退化掩码、F保存rows、结构、排名派生指标及辅助诊断，再合并固定R-S/T-S/R-T×3指标×2模型=18家族；n=16、SE=样本SD/4、df=15、边际Student95%、双侧p、全18 Holm；零观测方差为inferable=false、CI=null、p占位1。

## 判读与停止边界

本轮复用了此前已观察的样本，只能称探索性配对机制试验；条件MC区间不包含校准估计误差，不代表新模型/数据泛化。原O/C/Q与固定分组仅为辅助解释，不能选择有利p值或把方向效果变成可加因果损伤百分比。

完成当前A/B、独立数值与统计验收、结论及质量交付后停止新增实验，先与用户讨论结果和下一步。不得由本工程包自动进入新模块、训练、更多样本、调参或后续阶段。
