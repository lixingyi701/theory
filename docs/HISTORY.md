# 历史与分支沿革

2026-10-22 将研究分支 `fix/alfworld-sham-inreceptacle-20261018` 的工作整合到 main，统一后续研究入口。整合前 main 为 `143f928`，研究分支为 `072bf7d`；后者包含前者的提交历史。两条分支在整理后对齐到同一提交，修复分支作为阶段历史入口保留。

## 研究材料

- [当前研究协议](research/agent_error_recovery_model_and_protocol.md)：今天确认的研究方向与实验定义。
- [全分支审核](research/review-and-revised-plan-20261022.md)：近邻、架构和路线选择依据。
- [早期模型协议](research/agent_error_recovery_model_and_protocol.historical.md)：四状态模型、旧逐步记忆及早期实验矩阵。
- [五篇论文精读](research/agent_error_recovery_five_paper_review.md)：早期相关工作材料。

## 实验与环境

- [10 月 22 日作者 ReAct 结果](experiments/results-20261022.md)：当前工程基础。
- [10 月 19 日历史结果](experiments/results-20261019.md)：早期 Agent、sham 与服务诊断。
- [部署过程归档](deployment/history-20261022.md)：本地至远程迁移、ABI 修正和旧启动方式。
- `docs/deployment/*.json`、`docs/experiments/*.json`、`repro/`：版本、任务分区与脱敏结果凭据。

历史材料用于解释已有结果和复现早期运行。当前启动方式由执行手册维护，后续实验按当前协议和路线图推进。
