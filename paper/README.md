# 论文内容初稿

当前正文：[manuscript.md](manuscript.md)。中文工作稿，更新于2026-10-26，用于确认论文论证、理论与证据的衔接。题目与英文题目均为工作标题，后续在同一文件迭代。

主线是有限反思窗口中的错误延续如何影响一次记忆操作后的恢复收益。正文包含理论推导、受控模拟、原生语句干预、共同校准、54新任务两次预算验证及已完成的81新任务四次预算验证。摘要、实验和结论已同步保留四次主要比较未获支持的结果；完整更新的预测优势尚未建立。

## 主张与现有来源

| 正文内容 | 来源 | 已支持的解释 |
|---|---|---|
| 有限窗口、期望尝试数、预算反例、位置影响界 | [有限记忆理论](../docs/research/finite_memory_recovery_analysis.md) | 指定模型条件下的数学结论 |
| 状态压缩、恢复收益分解、任务异质性 | [有序状态分析](../docs/research/recovery_state_sufficiency_analysis.md) | 状态条件及预算递推关系 |
| 成功／失败经历／反思响应与误差分解 | [失败反馈分析](../docs/research/failure_feedback_and_reflection_update.md) | 需要分别估计的机制对象 |
| 26万条模拟与解析参照 | [受控模拟](../docs/experiments/finite-memory-simulation-20261023.md) | 人工指定机制的实现与数值一致性 |
| 原生语句干预 | [干预结果](../docs/experiments/native-clause-20261025.md) | 既定开发任务上的目标表达作用，固定失败输入与自然更新分开 |
| 旧错误退出与新反思重述 | [早期多轮诊断](../docs/experiments/multiround-diagnostic-20261024.md) | 旧条目身份与后继语义是不同事件，早期排序预测偏差保留 |
| 36任务共同参数 | [共同校准](../docs/experiments/shared-calibration-20261025.md) | 参数估计与模型选择，不充当独立验证 |
| 校准计数与组内拟合检查 | [计数勘误](../docs/experiments/shared-calibration-count-correction-20261026.md)、[成功标定审计](../repro/reflexion-20261025/success-calibration-audit-20261026.json) | 58次成功、405次失败写入、402种文本；原拟合输入正确，C/N首轮低估已存在于校准数据 |
| 后续成功侧开发诊断 | [正则化与后继状态检查](../docs/experiments/success-regularization-diagnostic-20261026.md) | 优化核对一致，弱正则化有小幅开发留出收益；零正则化有发散方向；多条原生反思状态已由下项补充标签。未替换正文的冻结验证结论 |
| 后继状态与校准覆盖补充 | [106文本与217次尝试诊断](../docs/experiments/successor-support-diagnostic-20261026.md)、[覆盖命题与证明](../docs/research/calibration_support_and_multistep_prediction.md) | 首轮改善未统一延续；区域内一致不保证后续预算一致。新命题与诊断待正文整合，原冻结结论保留 |
| 新成功函数的开发选择 | [680次尝试的六维／七维比较](../docs/experiments/successor-development-calibration-20261026.md) | 按原简约规则选七维，增量Brier0.00409，后继偏差仍保留；属于开发证据，待新任务评价后整合 |
| 下一次独立检验 | [100新任务冻结设计](../docs/experiments/successor-validation-design-20261026.md) | 参数、任务与预测已准备，尚未GPU采集，不写入正文为已完成实验 |
| 54新任务两次预算预测 | [冻结验证](../docs/experiments/shared-validation-20261025.md) | 主要Brier改善；更新机制增量与纠正／删除排序仍未分辨 |
| 81新任务四次预算预测 | [结果与诊断](../docs/experiments/four-attempt-validation-20261025.md) | 完整更新Brier0.22821，固定起点0.21862，主要损失差−0.00960，区间跨零；首轮标定偏差与后续条件曲线分别诊断 |
| 条件更新后的成功残差 | [定向诊断](../docs/experiments/update-diagnostic-20261025.md) | 已知汇总后的局部解释，不重新拟合 |
| 九项近邻工作的对象与证据 | [相关工作定位](../docs/research/related-work-positioning-20261025.md) | 复用已核对版本，错误延续仍是核心机制 |

## 当前写作安排

整体投稿工作量粗估完成55%—65%。问题、理论主体和一轮冻结验证已具备，正文现在可以并行推进。此估计是项目工作量判断，主要剩余工作是动态更新机制的实证解释、证明与术语的最终复核、英文全文、图表整合及ACM格式整理。

本版先采用Markdown组织完整内容，避免在论证稳定前耗时拟合页数。参考文献沿用已有核对的标题／简称、版本和原文链接，并补入ALFWorld数据与Qwen3模型的原始论文。作者、完整出版元数据与ACM引用格式在英文排版阶段统一整理，当前复现身份以仓库协议和报告为依据。

四次预算采集和独立核对均已完成，预算关闭，原参数、执行器、评分器与受保护设计保持不变。当前先核对成功侧标定和后继状态外推，不启动更大的相同矩阵。主要比较未获支持的结论不由补充指标替换；下一候选须作为模型修订，在新的独立比较中检验。
