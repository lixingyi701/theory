# 实现状态

更新：2026-10-23。当前研究方向为反思式 Agent 恢复后的跨条件可靠性预测。

## 已完成

- 项目、CPU runner、ALFWorld 数据、GPU 模型服务和原始日志已统一在远程服务器运行。
- 默认 `reference-run` 执行固定作者 ReAct 函数、示例和完整历史，骨干为 Qwen3-8B BF16。
- 19 项离线检查通过；direct 恢复审计和官方 Gym 对照各通过 10 任务／40 检查。
- dev12 完成 12 个任务，成功 7 个、覆盖 5 类；其余 5 个耗尽 49 轮预算。共 406 次调用，未确认请求 0。134 unseen 尚未调用模型。
- 已接入固定官方 Reflexion，保留作者 actor 与反思更新函数。Qwen 模板和动作行解码配置在三次尝试内成功 8/12、覆盖六类，750 次请求全部保存，无截断。
- 首组位置事实五分支 pilot 已完成，共 500 次真实请求；新场景清单冻结了六类各四个任务，其 direct／官方环境审计均为 10 任务／40 检查通过。

开发结果见[结果报告](experiments/results-20261022.md)，方法配置见[作者参考基线说明](experiments/reference-alignment-20261022.md)。

## 下一实施单元

作者 Reflexion 已接入，包含跨尝试反思、短输出重试、事件日志与持久请求记录。首组同状态五分支 pilot 已完成：500 次真实请求，原始／sham／定点纠正轨迹一致，错误和丢弃分支产生部分轨迹差异，五个条件均未成功。完整结果见[Reflexion 实验记录](experiments/reflexion-results-20261022.md)。

raw 接口的十次尝试 dev12 已完成，最终成功 5/12。修正 Qwen 接口后通过接入检查，24 个独立场景的 pilot 已从预算断点继续；本次追加上限为 6000 次调用、三小时。接下来完成目标绑定的配对分叉，依据事件覆盖和任务间方差确定正式校准样本与冻结验证条件。

研究定义见[当前协议](research/agent_error_recovery_model_and_protocol.md)，阶段安排见[实验路线](EXPERIMENT_ROADMAP.md)。历史自建 Agent 和部署诊断见[历史索引](HISTORY.md)。
