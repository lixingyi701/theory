# Agent reliability: residual errors after recovery

研究目标：解释并预测长程 LLM Agent 在不完全恢复之后的错误复发与任务结果。当前仓库提供**第一实施单元**：ALFWorld 文本环境、显式任务内记忆、可续跑的执行器和 checkpoint/replay 审计。

**2026-10-19 状态：本机Linux runner、远程Conda theory / vLLM0.10.2、固定8B模型与数据已部署并核验；真实10任务/40恢复审计通过。公开ReAct适配的开发预检base6/12过能力门槛，memory/sham各5/12未过仪器化门槛；[结果与下一节点](docs/experiments/results-20261019.md)已记录，未启动因果矩阵。** 原27/48零成功pilot作为失败联调记录归档，不能用于论文复现或理论结论。

## 从哪里开始

1. 阅读 [Linux 执行手册](docs/LINUX_RUNBOOK.md)，按“证据与冻结方案→真实恢复审计→基础Agent能力→记忆/sham对照→因果实验”的顺序执行。
2. Linux + Codex 的接续入口是 [实施交接](docs/IMPLEMENTATION_STATUS.md)；仓库 [AGENTS.md](AGENTS.md) 规定了运行和记录要求。
3. 研究依据：[五篇论文精读](docs/research/agent_error_recovery_five_paper_review.md)、[最小模型与实验协议](docs/research/agent_error_recovery_model_and_protocol.md)。
4. 本机/远程版本、隔离环境、复建freeze与脱敏SSH模板见 [部署记录](docs/deployment/README.md)；预设门槛见 [基础协议](docs/experiments/baseline-20261019.md) / [v2适配](docs/experiments/baseline-20261019-v2.md)。
5. 预计2–4周完成实测的顺序见 [实测路线](docs/EXPERIMENT_ROADMAP.md)。这是工作计划，不是已完成实验。

## 本次实现

| 组件 | 入口 | 当前范围 |
|---|---|---|
| 任务清单 | `agent-rel manifest` | 官方数据过滤、六类分层、场景分组、固定hash |
| 恢复审计 | `agent-rel audit` | 默认10任务、每任务2个检查点×2次恢复；检查facts、输出及固定后缀 |
| 基础/仪器化预检 | `agent-rel react-run` | 公开两示例、完整历史；base/memory/sham，49 decisions，语法适配 |
| 历史Agent runner | `agent-rel run` | 旧clean/sham工程路径，保留审计，不作为默认baseline |
| 显式记忆 | `src/agent_reliability/memory.py` | 16条结构化事实、来源步、确定性淘汰 |
| 中断恢复 | `--resume` | 重放环境、恢复上下文/记忆，复用已保存模型响应 |
| 请求审计 | 每个episode的`calls/` | 保存请求/响应、种子、usage、截断和不确定请求 |
| 下载/服务配置 | `scripts/`、`configs/models.json` | 固定模型revision；默认只打印计划 |

本阶段使用自建的 ReAct 式记忆Agent，**不是原版ReAct或RIR的完整复现**。尚未实现错误注入、R0–R3四恢复臂、Dependency-Workflow环境、参数估计与长度外推；这些均为后续单元。

## 今天可以运行的离线检查

```bash
bash scripts/offline_checks.sh
```

只使用Python标准库与测试替身；不会下载权重、调用LLM或运行ALFWorld。

## 固定资源

- 主模型：[Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)，revision `b968826d9c46dd6066d109eabc6255188de91218`。
- 后续模型：[Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)，revision `40c069824f4251a91eefaf281ebe4c544efd3e18`。
- [ALFWorld官方源码](https://github.com/alfworld/alfworld)，commit `aaba6870f86c5be6a08a491f32a50b906227bc3e`；TextWorld `1.6.2`。
- [官方下载器](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/scripts/alfworld-download)中的文本数据归档链接已写入下载脚本；不下载视觉检测器。

依赖的首选部署是独立的 Python 3.10 runner 环境与 Python 3.11 vLLM 服务环境。实际已安装环境的Conda构建清单与pip版本freeze位于 `requirements/locks/`；没有每个pip wheel的hash，不宣称跨平台二进制完全可复现。

## English

This repository studies residual-error propagation after agent recovery. The first unit implements an instrumented ReAct-style agent with explicit memory, an ALFWorld text adapter, replay audits, bounded budgets and a durable model-call journal. Start with [the Linux runbook](docs/LINUX_RUNBOOK.md). See the deployment snapshots and preregistered engineering gates under docs/. The failed historical pilot, replay checks and development capability results are reported separately; none establishes the recovery theory.
