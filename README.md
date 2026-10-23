# Agent 恢复后的可靠性

本轮采用[固定目标规则与其余文字的四组诊断](docs/experiments/residual-context-design-20261024.md)：12 个既定任务、两个配对种子，新增 72 次尝试并复用 24 条已审核控制。先检验其余文字是否改变成功及纠正收益，再确定多轮诊断所需状态；134 holdout 继续封存。

本项目研究会反思、会重试的 LLM Agent：纠正错误命题或删除含错反思后，后续反思与有限窗口更新，怎样影响给定尝试次数内的成功概率和平均尝试次数？主线是建立可分析的过程模型，推导恢复有效与失效的条件，再用受控仿真和真实 Agent 检验这些条件及其预测。

具体流程是：先描述“记忆读取—行动执行—失败回写”的过程，分析局部纠正、整条删除、错误生成和条目移出如何影响后续成功；在受控模拟中改变这些机制参数；再将公开 Reflexion 接到固定 Qwen3-8B，在 ALFWorld 的同一任务状态下保留、纠正或丢弃反思，估计必要参数并检验冻结后的新任务与预算预测。

## 当前主线

1. [研究问题、机制与实验协议](docs/research/agent_error_recovery_model_and_protocol.md)
2. [理论框架与基础推导](docs/research/recovery_theory_framework.md)；[有限记忆的主命题与证明](docs/research/finite_memory_recovery_analysis.md)
3. [有限记忆仿真结果](docs/experiments/finite-memory-simulation-20261023.md)；[采样前冻结设计](docs/experiments/finite-memory-simulation-design.md)
4. [中间位置与长反思前瞻结果](docs/experiments/middle-context-20261023.md)；[多轮条件预测](docs/experiments/agent-calibrated-multistep-forecast-20261023.md)；[新任务校准](docs/experiments/state-representation-20261023.md)；[后续路线](docs/EXPERIMENT_ROADMAP.md)
5. [ACM TIST 专刊与投稿计划](docs/submission/TIST_SPECIAL_ISSUE.md)
6. [实现状态与已有结果](docs/IMPLEMENTATION_STATUS.md)
7. [服务器执行手册](docs/LINUX_RUNBOOK.md)

目标为 ACM TIST 专刊 **LLM-Driven Agentic AI in Intelligent Systems: Theory, Technologies, and Applications**，征稿截止日期为 **2027-01-31**。原始[征稿 PDF](docs/submission/ACM-TIST-CfP-SI-LLM-Driven-Agentic-AI-Intelligent-Systems.pdf)随仓库保存。

## 已有基础

固定作者 ReAct＋Qwen3-8B 在开发集成功 7/12，作为工程参照。Reflexion 已统一 Qwen 非思考模板、动作行和公开放置语法；保持采样不变，dev12 三次内成功从 8/12 提高到 10/12。Qwen 推荐采样也为 10/12，按预设同分规则选为后续配置。当前 45 项离线检查通过，模板／采样契约与模型输出合规分别审计。见[接口规范化](docs/experiments/interface-normalization-20261023.md)。

冻结接口后，24 个新任务的首尝试完成 16 个任务，并产生 8 条失败反思。其中四个目的地绑定已完成五条件分叉，共 1067 次真实调用，各条件成功 2/4。错误条目进入可见记忆后，尚未观察到按错误目的地执行的动作。随后已开展操作前提扩展，补充相关动作尝试、执行生效与新反思事件，结果见下一段。[实验记录](docs/experiments/reflexion-results-20261022.md)和[校准细则](docs/experiments/calibration-plan-20261023.md)说明事件定义与采样方向。

[操作前提扩展](docs/experiments/precondition-pilot-20261023.md)完成另一组 24 个新任务，首次成功 17 个，并从 7 条失败反思中预先固定 3 个合格目标。五条件分叉共 1084 次请求：原始错误成功 0/3，定点纠正与丢弃各 1/3。错误顺序会被反思重新写入，纠正的效果也随任务而变。另一个开发杯子案例揭示了错误放置被环境拒绝后才成功的路径，因而相关动作尝试、实际生效和任务结果分别测量。

[首次冻结预测](docs/experiments/state-prediction-20261023.md)已完成：四个已知开发任务、28 个同质窗口校准单元、8 个混合窗口验证单元。混合窗口全部失败，数量和位置模型的 Brier 为 0.1438、0.1648，总体均值为 0.1111。这些结果用于约束模型结构；现已推导有限窗口的期望尝试数公式、预算内增益的非单调反例，以及起始记忆差异的影响上界；三组机制仿真也已完成，现已按错误生成／重述、位置线索与条目年龄完成首批真实校准。134 holdout 保持封存。此前该批次新增 2959 次真实调用，累计 Reflexion 调用 12369 次，响应全部保存。

有限记忆推导使用精确分数核对了完整窗口与压缩状态递推，以及独立线性方程解；所有比较相等，记录见[代数核对](repro/theory-20261023/finite-memory-analytic-checks.json)。随后已按冻结设计完成 26 万条逐步模拟轨迹、1125 个二值参数条件和 90 个信息／位置条件；260 项成功概率与 1920 项槽位差异检查均在预设容差内。仿真进一步显示恢复排序可能随预算先反转、再转回；这些是人工机制结果，真实 Agent 校准结果见下一段。三进程正式计算约 8.73 秒，无新增 LLM 调用，完整结果与曲线见[仿真报告](docs/experiments/finite-memory-simulation-20261023.md)。

[首批真实 Agent 定向校准](docs/experiments/directed-calibration-20261023.md)已完成：四个已知开发任务、104 个起始条件、150 次真实尝试，三卡约 48.93 分钟，新增 5984 次调用。实际通用计划与空记忆表现不同；经公开观测核验的位置线索在两次内成功计数上未获优势；本批尚未分辨稳定的位置收益方向。这些结果用于设计下一批受控短反思比较。

[新基础任务校准](docs/experiments/state-representation-20261023.md)完成 12 个新任务、108 次尝试：四卡约 21.50 分钟，3779 次调用，成功 45/108。预定比较选择错误剩余保留轮数表示，整任务留出五类 Brier 为 0.6132，常数为 0.7206，丰富表示为 0.6180。补充的错误存在性基线为 0.6034，尚无稳定年龄收益；参数、40 个状态预测和未覆盖位置均已保存。截至该批累计调用 22132 次，全部响应保存。

[中间位置与长反思检验](docs/experiments/middle-context-20261023.md)完成 48 次前瞻尝试，双卡约 22.68 分钟，新增 1740 次调用且无参数重拟合。中间位置优于原冻结常数基线；原生长反思未达到预测优势门槛，成功概率预测反而变差。当前先分别检查成功模型和反思更新模型，再固定目标规则文字，检验周围文字的影响。[真实参数的多轮递推](docs/experiments/agent-calibrated-multistep-forecast-20261023.md)已给出条件预测和单步误差分解，作为后续机制诊断。累计调用 23872 次，全部响应保存，134 holdout 继续封存。

全部实验在远程服务器运行，工作站用于编辑、Git 和 SSH。当前部署见[环境说明](docs/deployment/README.md)。离线检查从仓库根目录运行：

```bash
bash scripts/offline_checks.sh
```

## 分支与文档

`main` 是研究和实现的统一主线，已承接 `fix/alfworld-sham-inreceptacle-20261018` 的全部工作。该修复分支完成阶段任务，保留为历史入口；后续开发从 main 创建短期分支，完成后回到 main。

此前的[完整审核](docs/research/review-and-revised-plan-20261022.md)提供近邻比较与设计依据。早期模型、部署过程和失败诊断由[历史索引](docs/HISTORY.md)统一导航。阅读当前方案时，从上面的研究协议和理论框架开始。
