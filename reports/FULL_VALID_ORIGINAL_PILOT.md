# 原10步pilot checkpoint：只读完整验证

完整验证已完成，覆盖8,456条valid查询、8,210个child和82,115个候选实体（排名时按协议过滤self及其他真父）。原checkpoint前后SHA256一致；没有训练、写入best或改变历史模型选择。

评估来源8b9f8bc523a04147f9612090c0f4e6173d00f369；训练来源仍为fce017259a6bdc513f21b31304f096239e357a75；prepared复用2ca02d6。原首个成功pilot的last SHA256为038fd4bc72c71dd544da48453e24349dd9018035b662cf2e5c2799cb9615638a，未换成效率对照中新版的last。两类源码哈希分别核对归档，manifest与完整valid哈希匹配。

## 门槛与历史失败

旧8408af6在远程无Git归档中fixture来源缺失，CPU9失败/1通过，该记录独立保留在full-valid-cpu-gate-8408af6.json；这不证明显式来源参数下生产评估失败。新8b9f8bc仅修测试fixture，production源码未改。新版本远程同一目标10项fixture在10.45秒通过、无skip；随后实际checkpoint来源及输入核验通过。CPU分配16秒，不能追认旧版本测试通过。

## 完整开发指标

| 指标 | 实测值 |
|---|---:|
| query-micro MRR | 0.001850345392 |
| child-macro MRR | 0.001897378366 |
| Hits@1 | 0.000354777673 |
| Hits@3 | 0.000354777673 |
| Hits@10 | 0.006031220435 |

ranking状态complete，8456/8456条全部完成；所有逐查询秩保留于原始JSON。评估为两层完整G_obs编码及filtered all-entity候选，同分采用精确计算得分的平均秩；不读取test/truth用于本评估。这些是10步last的开发指标，不能称完整valid选出的best、收敛或有竞争力基线；selection_performed=false。

## 耗时与资源

| 阶段 | 秒 |
|---|---:|
| checkpoint/来源/prepared核验 | 2.975405 |
| 模型构建/载入/迁移 | 0.191112 |
| 两层完整图编码 | 3.276722 |
| 排名（外层同步计时） | 24.367800 |
| 评估总墙钟 | 30.816645 |

ranking内部计时24.356897秒与外层计时边界不同，原样保留。编码加排名27.644522秒，显著小于已冻结基线每次180秒评估预算；这一功能、完整性与成本门槛通过，MRR高低不作为是否继续的条件。

GPU单L40、2CPU、16GiB、外层15分钟，内部600秒，threads2、candidate chunk4096。实际驱动580.65.06，nvidia-smi总显存46,068MiB。CUDA前检通过；FP32、TF32关闭、无AMP。PyTorch peak allocated=520,022,016bytes（495.932MiB），reserved=629,145,600bytes（600MiB），是分配器峰值不是整GPU进程占用。Slurm COMPLETED/0:0，占40GPU秒=0.011111 GPU小时，资源已释放；没有与旧新测速并发。

## 交接与后续

三个新文件：reports/FULL_VALID_ORIGINAL_PILOT.md、reports/full-valid-original-pilot.json（原始逐字节复制）、reports/full-valid-original-pilot-metadata.json。另附旧失败记录reports/full-valid-cpu-gate-8408af6.json。原始JUnit、checkpoint前后哈希、调度和运行日志保留私有目录；不自行stage/commit/push。

依据既有条件授权，成本门槛通过后另行提交固定b6269f24e33602c10c39ee6b9bac57fe0901a405的seed11/23各一个独立单卡开发训练，从头初始化；split、features和固定负例均不重做。后续训练结果另记，不提前当作已完成。
