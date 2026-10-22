# 实施状态（2026-10-22）

- 运行已完整迁至服务器：Git项目、CPU runner、ALFWorld数据/评估器、GPU模型服务、历史与当前原始日志均在服务器。工作站只保留源码/SSH/报告；已校验备份的本地环境、数据、Q8历史权重及安装缓存已清理。
- CPU runner为独立Python3.10.22 prefix，GPU服务为原有Conda **theory** / Python3.11.15 / vLLM0.10.2；base history与pip freeze再次核对未变。迁移发现并修复宿主glibc兼容问题；原失败审计保留。
- 默认入口 `reference-run` 直接加载固定作者ReAct函数、示例和完整历史，保留固定Qwen3-8B BF16；原始completion接口采用作者100token/温度0/newline stop。现代ALFWorld固定语法别名和骨干替换均显式记录，不称论文全条件复现。
- 最终19项离线检查通过；direct真实恢复审计10任务/40检查通过；官方Gym公开输出对照10任务/40检查通过。native/source/task/model hash不匹配会阻止运行。
- 冻结dev12全部完成，成功7/12、覆盖5类，5个49轮预算耗尽；406次调用，最终未确认0。通过预先基础能力门槛；未测试134 unseen。见[结果](experiments/results-20261022.md)。
- 实测停止本机SSH master期间，远程任务继续从10完成至11；runner不依赖本机Python、模型端口转发或代理。控制连接已恢复为systemd唯一master，普通客户端只复用。

## 研究边界与下一门槛

原版ReAct没有独立持久记忆M，当前通过仅表示基础能力可用。已审阅官方Reflexion四个源码文件并记录固定commit/hash；它是下一记忆实现候选，尚未安装或运行。先冻结其作者算法、8B模型接口、反思/尝试预算与原生成功判定，验证基础能力和等内容sham，再决定配对注入与恢复协议。不能把跨尝试反思当成逐动作记忆写入。

错误注入、R0–R3四恢复臂、参数估计、Dependency-Workflow生成器及长度外推均未实现；理论未获验证。当前不再按旧自建记忆矩阵扩大实验。见[方法修订](experiments/reference-alignment-20261022.md)、[当前手册](LINUX_RUNBOOK.md)和[部署记录](deployment/README.md)。

## 历史记录

原零样本/短历史pilot只完成27/48且成功0、sham触发0；不是ReAct/RIR或因果证据。10月19日自建公开ReAct chat适配base6/12、memory5/12、sham5/12，仪器化与sham门槛未过；有限串行36请求检查一致不能消除协议差异。保留[历史结果](experiments/results-20261019.md)和原始远程journals，不续跑、也不把其成绩改写为本轮作者基线。

远程Codex CLI与实验runtime分开管理；10月18日登录/READY是历史检查。本次只验证实验离线运行独立性，不宣称远程CLI可脱离工作站代理联网。
