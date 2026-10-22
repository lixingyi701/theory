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

固定作者 ReAct＋Qwen3-8B 在开发集成功 7/12，作为工程参照。Reflexion 已统一 Qwen 非思考模板、动作行和公开放置语法；保持采样不变，dev12 三次内成功从 8/12 提高到 10/12。旧接口的 24 任务 pilot 已完成，成功 14 个。当前正在比较两套采样配置并进行逐请求接口审计，见[接口规范化](docs/experiments/interface-normalization-20261023.md)和[实测记录](docs/experiments/reflexion-results-20261022.md)。

代码提供任务清单、环境审计、作者 ReAct／Reflexion 执行入口、持久请求日志与中断续跑。Reflexion 已完成首组真实五分支 pilot；[实验记录](docs/experiments/reflexion-results-20261022.md)包含结果和下一步调整。[运行协议](docs/experiments/reflexion-protocol-20261022.md)说明作者配置、干预与预算，机制校准和冻结验证将依据独立任务的事件覆盖继续推进。

全部实验在远程服务器运行，工作站用于编辑、Git 和 SSH。当前部署见[环境说明](docs/deployment/README.md)。离线检查从仓库根目录运行：

```bash
bash scripts/offline_checks.sh
```

## 分支与文档

`main` 是研究和实现的统一主线，已承接 `fix/alfworld-sham-inreceptacle-20261018` 的全部工作。该修复分支完成阶段任务，保留为历史入口；后续开发从 main 创建短期分支，完成后回到 main。

今天的[完整审核](docs/research/review-and-revised-plan-20261022.md)提供近邻比较与设计依据。早期模型、部署过程和失败诊断由[历史索引](docs/HISTORY.md)统一导航。阅读当前方案时，从上面的五个入口开始。
