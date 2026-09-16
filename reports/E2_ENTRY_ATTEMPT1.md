# E2入口尝试：来源门槛失败，未取得科学观测

研究问题：真实训练模型固定邻域中的采样偏移能否可靠分辨，是否足以支持后续层级损伤研究？结论：本次没有获得可回答该问题的观测。目的未达到，因为两个seed均在完整前向及采样之前被基线报告原始字节哈希门槛拒绝；这不是模型数值失败、偏移不存在或层级能力不足的证据。

执行源码固定8e4209a23f69993c416ee1c28b857f02673363cb、配置规范SHA052e30ec2f243ce9ef0d609696c7483ff390785417703e587dfe8458298014e9。launcher错误地使用Git归档中的基线报告副本；提交前仅检查规范LF哈希，而入口要求原始文件字节SHA。归档副本含不同换行字节，因而正确触发baseline report hash mismatch。执行准备遗漏由实验任务承担，不放宽来源门槛。

只读核对确认远端原训练run.json的raw哈希分别为64305aefdfecb77bcdeffd94da912f1ae4b1d1683abf2c4aa706973d667fbe94和4de0b6a51555b2b16f17c4f003cb7b1fcce8278c00b4b6658b291f04566f329b，正好等于固定配置。归档副本规范LF哈希也一致，但这不能代替原始字节身份核验。

两单L40的CUDA前检通过，驱动580.65.06、46,068MiB；每作业2CPU/16GiB。两个Slurm作业分别8秒和7秒FAILED/1:0，共15GPU秒=0.004167GPU小时，资源均已释放。没有完整前向、完整valid复现或局部16次采样，不计算科学均值；未自动重试、修改配置/阈值或进入E3。

若用户明确批准恢复尝试，具体修复是将baseline-report指向已保留的原训练run.json，并在申请GPU前检查raw哈希；固定源码和科学范围不变。此为提案，尚未执行。原E2整体并未完成，原路线其余缺口也未因此解决。

交接四文件：本报告、e2-entry-seed11-attempt1.json、e2-entry-seed23-attempt1.json、e2-entry-attempt1-metadata.json，均在reports。两个失败JSON原样保存；原日志和分配证据私有保留，未git提交。

## 只读字节链定位（未重试）

已将Git blob、本地原报告、本地source.tar成员、远端tar成员、远端release文件与原训练run逐项比较。Git blob和本地/远端原报告均全LF且raw哈希符合配置；本地tar成员每文件已有265,625个CRLF。上传前后整个tar SHA相同，远端tar成员与解包后的release字节完全一致。实际launcher参数为release中的reports/baseline-seed<seed>-run.json，其文件hash与错误相符。

因此转换发生在本地archive导出阶段，不是Git blob被改、不是真实训练报告漂移，也不是scp或解包转码。本机Git系统设置core.autocrlf=true，归档.gitattributes为* text=auto，JSON未显式eol=lf。对同commit的两个报告做了三个仅文件导出对照：默认git archive重现CRLF；单命令core.autocrlf=false仍为CRLF；单命令同时core.autocrlf=false和core.eol=lf得到全LF，raw哈希完全匹配。没有修改全局或仓库配置，没有运行科学计算。

可审查修复为：后续获批执行前用原训练run.json作为报告信任锚，并逐字节预检；若使用归档报告，则必须显式LF导出且raw哈希通过。不能用规范LF哈希替换入口的raw约束。三个导出及完整字节链证据保存在.local/e2-pilot/archive-conversion-experiment.json，实际launcher原文保留。该定位的三批命令工具墙钟约2.0、3.7、1.0秒（含本地/SSH文件检查与传输），无新增调度或GPU卡时；人工分析时间未作为计算耗时。仍未重试。
