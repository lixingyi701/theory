# Agent 恢复后的可靠性

本项目研究会反思、会重试的 LLM Agent：一次恢复之后，保留下来的错误经验、交互状态和剩余预算，怎样影响后续复发与任务完成？我们通过同一任务状态下的配对干预，估计错误传播与反思写入的规律，再预测新任务和未参与拟合的恢复条件下的结果。

具体流程是：将公开 Reflexion 程序接到固定的 Qwen3-8B 上，在 ALFWorld 中执行任务；在原生反思中改错一个有证据的事实，从同一状态分别保留错误、纠正目标事实或丢弃新增反思；记录错误进入行动、返工与任务完成的过程，检验恢复后状态能否支持跨条件预测。

## 当前主线

1. [研究问题、机制与实验协议](docs/research/agent_error_recovery_model_and_protocol.md)
2. [下一步实验计划](docs/EXPERIMENT_ROADMAP.md)
3. [ACM TIST 专刊与投稿计划](docs/submission/TIST_SPECIAL_ISSUE.md)
4. [实现状态与已有结果](docs/IMPLEMENTATION_STATUS.md)
5. [服务器执行手册](docs/LINUX_RUNBOOK.md)

目标为 ACM TIST 专刊 **LLM-Driven Agentic AI in Intelligent Systems: Theory, Technologies, and Applications**，征稿截止日期为 **2027-01-31**。原始[征稿 PDF](docs/submission/ACM-TIST-CfP-SI-LLM-Driven-Agentic-AI-Intelligent-Systems.pdf)随仓库保存。

## 已有基础

固定作者 ReAct＋Qwen3-8B 已在开发集完成 12 个任务，成功 7 个，覆盖 5 类；恢复审计与官方环境接口对照各通过 10 任务／40 检查。完整结果见[开发实验报告](docs/experiments/results-20261022.md)。下一实施单元是作者 Reflexion、命题级干预与科学分叉。

代码提供任务清单、环境审计、作者 ReAct 执行入口、持久请求日志与中断续跑。`--resume` 用于继续既有运行；配对研究需要独立的未来调用和分支日志。Reflexion、错误注入、恢复分叉和预测模型仍需实现。

全部实验在远程服务器运行，工作站用于编辑、Git 和 SSH。当前部署见[环境说明](docs/deployment/README.md)。离线检查从仓库根目录运行：

```bash
bash scripts/offline_checks.sh
```

## 分支与文档

`main` 是研究和实现的统一主线，已承接 `fix/alfworld-sham-inreceptacle-20261018` 的全部工作。该修复分支完成阶段任务，保留为历史入口；后续开发从 main 创建短期分支，完成后回到 main。

今天的[完整审核](docs/research/review-and-revised-plan-20261022.md)提供近邻比较与设计依据。早期模型、部署过程和失败诊断由[历史索引](docs/HISTORY.md)统一导航。阅读当前方案时，从上面的五个入口开始。
