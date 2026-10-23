# Agent 恢复后的可靠性

本项目研究会反思、会重试的 LLM Agent：一次恢复之后，保留下来的错误经验、交互状态和剩余预算，怎样影响后续复发与任务完成？我们通过同一任务状态下的配对干预，估计错误传播与反思写入的规律，再预测新任务和未参与拟合的恢复条件下的结果。

具体流程是：将公开 Reflexion 程序接到固定的 Qwen3-8B 上，在 ALFWorld 中执行任务；对原生反思中的事实或操作前提进行定点干预，从同一状态分别保留错误、纠正指定内容或丢弃新增反思；记录错误进入行动、返工与任务完成的过程，检验恢复后状态能否支持跨条件预测。

## 当前主线

1. [研究问题、机制与实验协议](docs/research/agent_error_recovery_model_and_protocol.md)
2. [下一步实验计划](docs/EXPERIMENT_ROADMAP.md)
3. [ACM TIST 专刊与投稿计划](docs/submission/TIST_SPECIAL_ISSUE.md)
4. [实现状态与已有结果](docs/IMPLEMENTATION_STATUS.md)
5. [服务器执行手册](docs/LINUX_RUNBOOK.md)

目标为 ACM TIST 专刊 **LLM-Driven Agentic AI in Intelligent Systems: Theory, Technologies, and Applications**，征稿截止日期为 **2027-01-31**。原始[征稿 PDF](docs/submission/ACM-TIST-CfP-SI-LLM-Driven-Agentic-AI-Intelligent-Systems.pdf)随仓库保存。

## 已有基础

固定作者 ReAct＋Qwen3-8B 在开发集成功 7/12，作为工程参照。Reflexion 已统一 Qwen 非思考模板、动作行和公开放置语法；保持采样不变，dev12 三次内成功从 8/12 提高到 10/12。Qwen 推荐采样也为 10/12，按预设同分规则选为后续配置。当前 33 项离线检查通过，模板／采样契约与模型输出合规分别审计。见[接口规范化](docs/experiments/interface-normalization-20261023.md)。

冻结接口后，24 个新任务的首尝试完成 16 个任务，并产生 8 条失败反思。其中四个目的地绑定已完成五条件分叉，共 1067 次真实调用，各条件成功 2/4。错误条目进入可见记忆后，尚未观察到按错误目的地执行的动作。下一步扩展有公开证据且不被当前任务直接重述的事实／操作前提，获得采用事件后校准机制，再冻结参数验证跨恢复条件预测。[实验记录](docs/experiments/reflexion-results-20261022.md)和[校准细则](docs/experiments/calibration-plan-20261023.md)说明事件定义与采样方向。

[操作前提扩展](docs/experiments/precondition-pilot-20261023.md)完成另一组 24 个新任务，首次成功 17 个，并从 7 条失败反思中预先固定 3 个合格目标。五条件分叉共 1084 次请求：原始错误成功 0/3，定点纠正与丢弃各 1/3。错误顺序会被反思重新写入，纠正的效果也随任务而变。另一个开发杯子案例揭示了错误放置被环境拒绝后才成功的路径，因而采用、执行和结果分别测量。

当前进行[受控记忆状态校准](docs/experiments/calibration-plan-20261023.md#当前执行的受控状态开发试验)：在四个已知开发任务上，用同质窗口拟合数量／位置模型，冻结参数后预测两种混合窗口。正式新任务与多步预测仍按后续阶段开展，134 holdout 保持封存。

全部实验在远程服务器运行，工作站用于编辑、Git 和 SSH。当前部署见[环境说明](docs/deployment/README.md)。离线检查从仓库根目录运行：

```bash
bash scripts/offline_checks.sh
```

## 分支与文档

`main` 是研究和实现的统一主线，已承接 `fix/alfworld-sham-inreceptacle-20261018` 的全部工作。该修复分支完成阶段任务，保留为历史入口；后续开发从 main 创建短期分支，完成后回到 main。

今天的[完整审核](docs/research/review-and-revised-plan-20261022.md)提供近邻比较与设计依据。早期模型、部署过程和失败诊断由[历史索引](docs/HISTORY.md)统一导航。阅读当前方案时，从上面的五个入口开始。
