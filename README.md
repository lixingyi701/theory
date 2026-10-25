# Agent 恢复后的可靠性

[近邻方法比较](docs/research/related-work-positioning-20261025.md)已按九项原文整理，补入More Memory, Worse Agents等直接相关工作。[已有更新诊断](docs/experiments/update-diagnostic-20261025.md)完成134条首轮失败反思的定向标注：删除组七条第二轮成功轨迹均仍可见旧错误，但没有先将目标物放入微波炉。将实际反思类别代入后，模型仍预测约3条成功、实际7条，提示继续检查成功核在原生更新之后的适用性。

[四次预算设计](docs/experiments/four-attempt-validation-design-20261025.md)已选81个新任务、27场景；主要样本比较C/N起点的完整更新预测56.44%与固定起点预测79.72%，另用27任务检查保留／纠正／删除。规划486轨迹、最多1944尝试，种子跨条件使用；四服务耗时估计约5.7小时。任务清单与精度规划完成，真实采集尚未启动，本阶段新增模型调用0。

[冻结参数的新任务验证](docs/experiments/shared-validation-20261025.md)已完成54任务、九场景、378条轨迹与700次尝试。两次预算Brier由相同任务混合常数的0.18151降至0.14781，改善0.03369，场景t95%区间[0.01360,0.05379]；零参数重拟合。起始记忆提供了预测区分，完整更新相对固定初始记忆的额外收益尚未分辨，纠正相对整条删除的优势也未确立。29159次调用全部响应，累计93186次，本批推理服务已释放。原134任务继续封存。

此前[共同机制校准](docs/experiments/shared-calibration-20261025.md)完成36任务、18场景的252条轨迹与463次尝试，固定了本次使用的共同参数。新验证保持相同七种起点，任务未参与拟合，场景与当前校准分区分离。接下来集中整理方法差异、检查已有反思中的更新误差，再设计四次预算的冻结比较以区分更新机制。

[原生反思目标语句干预](docs/experiments/native-clause-20261025.md)完成 108 次完整尝试和 108 次固定失败输入反思调用。纠正成功 19/36，原错误 8/36，主要差值 +30.56 个百分点，调整后 97.5% 区间 [5.56,55.56]；删句成功 18/36，尚未分辨纠正相对删句的优势。固定输入下 E 类反思从 35/36 降至 12/36，同时保留 10 条纠正组含糊／冲突 A。后续共同校准与冻结预测见上方进展。新增 3758 次请求全部响应，累计 44939 次；临时 GPU1/2/3 已释放，134 个留出任务继续封存。

[正确规则与空记忆诊断](docs/experiments/correctness-diagnostic-20261024.md)已完成 180 次新尝试。已有错误时，最新正确规则使成功从 2/36 变为 3/36，主要差值 +2.78 个百分点，95% 区间 [−5.56,+11.11]；正确规则与空记忆均为 21/36，通用计划为 13/36。首请求核对确认，非空记忆还带入标题和条目编号，随后开展了上面的原生语句干预。新增 6590 次请求全部响应，累计 41181 次，134 个留出任务继续封存。

[有序窗口与收益分解](docs/research/recovery_state_sufficiency_analysis.md)已给出状态压缩条件、两次预算收益分解、任务混合关系与误差传播界。实际长度和有序条目保留为过程结构，是否能进一步压缩由定向实验检查。[失败经历与反思更新](docs/research/failure_feedback_and_reflection_update.md)进一步拆分成功、失败经历分布和 reflector 的响应，并给出对应预测误差界。系统相关工作比较留到当前实验问题解决后集中整理。

[外部审核回应与研究调整](docs/research/review-response-and-research-adjustment-20261024.md)逐项核验了当前建模、统计和近邻工作。主线继续是有限反思记忆的恢复收益及其可预测条件；定向检查、共同参数校准与两次预算的新任务冻结验证已完成，当前进入证据整合与更新机制的定向分析。

[多轮过程诊断](docs/experiments/multiround-diagnostic-20261024.md)已完成 72 条轨迹、189 次尝试。重复错误起点两次预算中，纠正成功 0/12、删除 1/12，冻结的纠正优势未复现；纠正组起始旧错误退出后，12 条新反思中有 10 条继续表达该错误。该结果推动了后续的共享状态修订与最小同状态诊断。新增 7918 次请求全部响应，累计 34591 次；134 个留出任务继续封存。

[固定规则的文字诊断](docs/experiments/residual-context-20261024.md)已完成 72 次新尝试，复用 24 条旧对照，共 96 格。比较通用计划与删去目标命题后的反思文字，两种背景下含错条件均成功 1/24，纠正后均为 13/24；纠正收益均为 50 个百分点，背景对收益的平均影响尚未分辨。双卡新增 2801 次调用，约 35.38 分钟，累计 26673 次请求全部响应。该批随后用于上面的多轮过程诊断。

本项目研究会反思、会重试的 LLM Agent：纠正错误命题或删除含错反思后，后续反思与有限窗口更新，怎样影响给定尝试次数内的成功概率和平均尝试次数？主线是建立可分析的过程模型，推导恢复有效与失效的条件，再用受控仿真和真实 Agent 检验这些条件及其预测。

具体流程是：先描述“记忆读取—行动执行—失败回写”的过程，分析局部纠正、整条删除、错误生成和条目移出如何影响后续成功；在受控模拟中改变这些机制参数；再将公开 Reflexion 接到固定 Qwen3-8B，在 ALFWorld 的同一任务状态下保留、纠正或丢弃反思，估计必要参数并检验冻结后的新任务与预算预测。

## 当前主线

1. [研究问题、机制与实验协议](docs/research/agent_error_recovery_model_and_protocol.md)
2. [理论框架与基础推导](docs/research/recovery_theory_framework.md)；[有限记忆的主命题与证明](docs/research/finite_memory_recovery_analysis.md)
3. [有限记忆仿真结果](docs/experiments/finite-memory-simulation-20261023.md)；[采样前冻结设计](docs/experiments/finite-memory-simulation-design.md)
4. [原生目标语句干预](docs/experiments/native-clause-20261025.md)；[正确规则与空记忆诊断](docs/experiments/correctness-diagnostic-20261024.md)；[多轮过程诊断](docs/experiments/multiround-diagnostic-20261024.md)；[固定规则的文字诊断](docs/experiments/residual-context-20261024.md)；[中间位置与长反思前瞻结果](docs/experiments/middle-context-20261023.md)；[多轮条件预测](docs/experiments/agent-calibrated-multistep-forecast-20261023.md)；[新任务校准](docs/experiments/state-representation-20261023.md)；[后续路线](docs/EXPERIMENT_ROADMAP.md)
5. [ACM TIST 专刊与投稿计划](docs/submission/TIST_SPECIAL_ISSUE.md)
6. [实现状态与已有结果](docs/IMPLEMENTATION_STATUS.md)
7. [服务器执行手册](docs/LINUX_RUNBOOK.md)

目标为 ACM TIST 专刊 **LLM-Driven Agentic AI in Intelligent Systems: Theory, Technologies, and Applications**，征稿截止日期为 **2027-01-31**。原始[征稿 PDF](docs/submission/ACM-TIST-CfP-SI-LLM-Driven-Agentic-AI-Intelligent-Systems.pdf)随仓库保存。

## 已有基础

固定作者 ReAct＋Qwen3-8B 在开发集成功 7/12，作为工程参照。Reflexion 已统一 Qwen 非思考模板、动作行和公开放置语法；保持采样不变，dev12 三次内成功从 8/12 提高到 10/12。Qwen 推荐采样也为 10/12，按预设同分规则选为后续配置。当前 59 项离线检查通过，模板／采样契约与模型输出合规分别审计。见[接口规范化](docs/experiments/interface-normalization-20261023.md)。

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
