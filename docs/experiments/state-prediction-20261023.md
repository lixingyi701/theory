# 恢复状态的首次冻结预测

本轮完成四个已知开发任务上的受控记忆窗口试验：先运行同质窗口，拟合并上传参数与具体预测，再执行未参与拟合的混合顺序。它检验单次剩余尝试的成功概率，后续再扩展到新任务和多步恢复。

## 固定的比较

E 是包含指定错误操作前提的原生反思，C 只纠正该前提；其他建议和竞争错误保留。每种状态均执行完整原生 actor，失败后运行原生反思更新器。结局为最多 49 次决策内的 native won。

- 校准：空窗口、E、EE、EEE、C、CC、CCC，共 28 个任务—状态单元；12 个来自已核对身份的已有真实尝试，16 个新执行。
- 验证：ECC 与 CCE，每任务各一次，共 8 个新执行单元；两种窗口内容多重集相同，仅错误位置不同。
- 四个基础任务由一个已有开发诊断和三个预先核对的新目标组成；同一条反思的重复属于主动构造的记忆状态。独立任务数为 4。

## 校准与实际结局

表中 1 表示成功，0 表示失败。前七列参与拟合，最后两列在参数冻结后执行。

| 基础任务 | 空 | E | EE | EEE | C | CC | CCC | ECC | CCE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cup 开发诊断 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 0 | 0 |
| egg → diningtable | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mug → coffeemachine | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| egg → sidetable | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |

数量模型有 3 个系数，位置模型有 7 个系数，均使用正则 logistic；正则强度按基础任务留一交叉验证选择。所选强度分别为 10.0 与 10.0。总体均值使用 Beta(1,1) 平滑。任务难度诊断基线单独预注册，只使用每个任务七种同质状态的平滑成功率。

## 留出组合的预测误差

| 预测模型 | Brier ↓ | Log loss ↓ |
|---|---:|---:|
| 总体均值 | 0.1111 | 0.4055 |
| 条目数量 | 0.1438 | 0.4767 |
| 窗口位置 | 0.1648 | 0.5204 |
| 已知任务难度（诊断） | 0.1759 | 0.5139 |

八个混合窗口全部失败，数量和位置模型均未优于总体均值。当前结果没有显示加性状态模型的组合外推优势。验证没有成功样本，尚不能评估成功事件的区分能力；总体均值在这组全失败结果上得分最低，也不能据此认定它能推广到新的恢复条件。

以下为调用前已保存的成功概率及实际结局。

| 任务 | 顺序 | 数量 | 位置 | 总体均值 | 任务难度 | 实际 |
|---|---|---:|---:|---:|---:|---:|
| cup 开发诊断 | ECC | 0.3792 | 0.4310 | 0.3333 | 0.6667 | 0 |
| cup 开发诊断 | CCE | 0.3792 | 0.3793 | 0.3333 | 0.6667 | 0 |
| egg → diningtable | ECC | 0.3792 | 0.4310 | 0.3333 | 0.1111 | 0 |
| egg → diningtable | CCE | 0.3792 | 0.3793 | 0.3333 | 0.1111 | 0 |
| mug → coffeemachine | ECC | 0.3792 | 0.4310 | 0.3333 | 0.2222 | 0 |
| mug → coffeemachine | CCE | 0.3792 | 0.3793 | 0.3333 | 0.2222 | 0 |
| egg → sidetable | ECC | 0.3792 | 0.4310 | 0.3333 | 0.4444 | 0 |
| egg → sidetable | CCE | 0.3792 | 0.3793 | 0.3333 | 0.4444 | 0 |

## 轨迹与下一关

同一条错误反思可见，不代表错误动作必然被执行。开发杯子的一条 E 失败、两条 E 成功、三条 E 再次失败；两条 E 的成功轨迹没有尝试放置目标杯子，而是直接手持加热。单条 C 和两条 C 的成功均发生在错误放置被拒绝之后，三条 C 则没有尝试这一放置。因而记忆条目、动作尝试、接受／拒绝和任务结果分别记录。

八个混合窗口都接受了目标物体的错误放置。五个单元随后用规范的 heat-with 命令加热已放下的目标物体而失败；一个转向错误的 stoveburner，另两个继续尝试保留在反思中的非规范 heat-in／use 拼写。后两类没有计入规范宏的错误链，但仍保留它们已经放下目标物体的证据。

原生反思还会重新生成错误前提。校准的 19 个失败单元中，人工逐项复核得到 16 条明确重述目标错误、2 条未明确涉及目标前提、1 条目标放置建议含糊；后两类均保留其他错误的可能。它们来自四个基础任务，作为回写事件记录，未加入本轮已冻结的预测拟合。

后续先推进[机制理论与受控仿真](../research/recovery_theory_framework.md)，再由未识别参数决定独立任务、EC／CE 或更新事件的补采。起点任务特征、动作接受和反思再生成如何进入模型，须先有明确假设与推导。下一次正式验证使用新的基础任务，新增种子用于任务内稳定性。134 holdout 保持封存。

## 成本与复现

校准新增 581 次请求，验证新增 360 次；复用的 12 个单元不再次计费。当前追加批次共 2959 次调用，累计 Reflexion 调用 12369 次，响应全部保存。最终 33 项离线检查通过。

源代码位于 `scripts/`，状态协议、参数、逐单元预测、实际结果、接口／动作审计与时间线工件位于 `repro/reflexion-20261023/`。参数与预测的公开提交为 [4da94df](https://github.com/lixingyi701/theory/commit/4da94df13ba766ca34f40c6c6be5ff1c1856b567)；关闭记录核对其发布时间早于第一条混合窗口请求。

- 协议：[主要设计](../../repro/reflexion-20261023/state-prediction-protocol.json)、[任务难度补充](../../repro/reflexion-20261023/state-task-baseline-protocol.json)。
- 冻结：[拟合参数](../../repro/reflexion-20261023/state-calibration-parameters.json)、[逐单元预测](../../repro/reflexion-20261023/state-frozen-predictions.json)、[任务难度预测](../../repro/reflexion-20261023/state-task-baseline-freeze.json)。
- 结果：[校准单元](../../repro/reflexion-20261023/state-calibration-summary.json)、[验证单元](../../repro/reflexion-20261023/state-validation-summary.json)、[主要评估](../../repro/reflexion-20261023/state-prediction-evaluation.json)、[诊断基线评估](../../repro/reflexion-20261023/state-task-baseline-evaluation.json)。
- 机制：[校准行为](../../repro/reflexion-20261023/state-calibration-events.json)、[验证行为](../../repro/reflexion-20261023/state-validation-events.json)、[回写复核](../../repro/reflexion-20261023/state-calibration-reflection-review.json)。
- 执行：[批次账本与时间线](../../repro/reflexion-20261023/precondition-campaign-closeout.json)、[采集代码](../../scripts/run_reflexion_state_prediction.py)、[拟合代码](../../scripts/fit_reflexion_state_prediction.py)、[评估代码](../../scripts/evaluate_reflexion_state_prediction.py)。
