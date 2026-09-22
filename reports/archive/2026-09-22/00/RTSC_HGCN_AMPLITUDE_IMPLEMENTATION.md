# HGCN 幅度修订 v1：代码与数值质量记录

2026-09-22。本页只记录本轮独立源码的实现与本机检查；科学结果须等待原服务器 CPU/CUDA 质量门、丢弃权重探针、四组训练及五条件最终评估。

## 研究问题与当前状态

问题：将旧版三候选方向按物理范数平滑校准、改用软限幅和原始幅度正则后，能否在两个固定 HGCN 上获得更大的检索收益，同时保持层级改善？关系损失能否产生额外收益？**目前只有工程实现与本机质量检查，科学目的尚无结果。**旧阶段 A 的正结果和约 5.3% MRR 损失恢复率保持原记录，不在本轮改写。

前瞻协议为 [RTSC_HGCN_AMPLITUDE_V1.md](../docs/operations/RTSC_HGCN_AMPLITUDE_V1.md)。本轮配置为 [hgcn_rtsc_amplitude_v1.json](../configs/hgcn_rtsc_amplitude_v1.json)，canonical SHA256 为 `d48083a0fa4e36a4be97d5837327bf0be7b68c1ea90abddb478a73a07af1d544`。当前实现是待主管独立审查的候选，不是服务器执行许可。

## 新版实现及隔离

- 新模块只在独立源码中定义。旧 `hgcn_tangent_correction.py`、`hgcn_rtsc_stage_a.py`、旧配置和检查点未修改；新入口会对照原阶段 A 源提交核验五个旧源码文件的规范化哈希，并以旧绑定加载四臂中两个旧选中检查点。
- 每个方向以聚合点 Riemann 范数和物理 `epsilon=1e-8` 平滑单位化；固定 `s=4`，按 `r=(4 tau)q sum_j a_j u_j` 构造原始步长，以 `rsqrt(1+||r||²/tau²)` 做软限幅。优化项取逐节点、逐层平均 `||r||²/tau²`；截断后平方只单列诊断。控制器仍为两层共454参数，零输出初始化。
- 初始化、训练批次、两层训练图和选模图继续用旧命名流；最终图改用本轮独立命名流。正式最终入口对同一新图配对 S、旧两臂和新两臂，每片60条件；四个完整邻域模块控制加 S 的原始完整邻域控制都需独立合格。
- 新版最终汇总把新旧同臂的 MRR 单侧优效及层级单侧 `−0.002` 非劣纳入统一 Holm24；新臂相对 S 用双侧 Holm24，关系相对任务用双侧 Holm12。16个整图采样重复是统计单位，零方差一律 `p=1`、区间未定义且不拒绝。

## 本机核验

在工作树 `F:\ACL\_HGT`，固定本地 HGCN checkout `.local/mature-hgcn-validation/upstream`，使用 `PYTHONPATH=src` 运行。

1. 新模块、协议与短训练测试：`python -m pytest -q tests/test_hgcn_rtsc_amplitude.py tests/test_hgcn_rtsc_amplitude_protocol.py tests/test_hgcn_rtsc_amplitude_train_loop.py`。覆盖物理范数公式、零点/近零、FP32 对量化后 FP64 的值和梯度、FP64 有限差分、软限幅的 q 响应及 raw 正则梯度、两层零初始化/冻结/完整与低 k 恒等、同旧版 RNG、真实两步优化器探针、四绑定检查点加载与60条件配对。
2. 新质量函数在本机 CPU 原始上游环境运行，通过12个物理近零/边界 FP32↔FP64 值及梯度对照、非零系数后的完整/低 k 恒等、边界整模块有限梯度及 q/正则梯度。此结果不能替代服务器调度 CPU/CUDA 质量检查。
3. 新版静态入口返回准确配置哈希并要求独立 release。已从接受的 `aadbecaa...` 提交逐个核对五个旧 Stage A 源码文件，当前规范化字节相同。
4. 全库回归在加入最终诊断汇总的小修订前为 **643 passed，6 skipped**；修订只扩展了描述性恢复率至三个预算，之后新版与旧阶段 A 合并定向回归为 **35 passed**。旧官方 PyTorch 产生弃用警告，无本轮失败。

本轮未执行科学训练、最终评价或占用 GPU。下一执行关口为主管对精确源码/配置及本机证据独立审查，再在原服务器依次放行 CPU、CUDA 与丢弃权重探针；不能从本机通过直接推断方法收益。
