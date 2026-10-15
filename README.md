# Agent reliability: residual errors after recovery

研究目标：解释并预测长程 LLM Agent 在不完全恢复之后的错误复发与任务结果。当前仓库提供**第一实施单元**：ALFWorld 文本环境、显式任务内记忆、可续跑的执行器和 checkpoint/replay 审计。

**2026-10-15 状态：代码与离线检查已完成；尚未安装真实 ALFWorld 环境、下载模型/数据、启动模型服务或运行 benchmark。** 离线 fake backend/provider 只用于软件测试，没有研究结论。

## 从哪里开始

1. 阅读 [Linux 执行手册](docs/LINUX_RUNBOOK.md)，按“安装→数据→10任务恢复审计→模型服务→24任务 clean/sham”的顺序执行。
2. Linux + Codex 的接续入口是 [实施交接](docs/IMPLEMENTATION_STATUS.md)；仓库 [AGENTS.md](AGENTS.md) 规定了运行和记录要求。
3. 研究依据：[五篇论文精读](docs/research/agent_error_recovery_five_paper_review.md)、[最小模型与实验协议](docs/research/agent_error_recovery_model_and_protocol.md)。
4. 预计2–4周完成实测的顺序见 [实测路线](docs/EXPERIMENT_ROADMAP.md)。这是工作计划，不是已完成实验。

## 本次实现

| 组件 | 入口 | 当前范围 |
|---|---|---|
| 任务清单 | `agent-rel manifest` | 官方数据过滤、六类分层、场景分组、固定hash |
| 恢复审计 | `agent-rel audit` | 默认10任务、每任务2个检查点×2次恢复；检查facts、输出及固定后缀 |
| Agent runner | `agent-rel run` | clean/sham，80动作/164模型调用的独立预算 |
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

依赖的首选部署是独立的 Python 3.10 runner 环境与 Python 3.11 vLLM 服务环境。当前是直接版本固定方案；目标Linux机器完成安装后还要保存实际`pip freeze`，不能称为已验证的完整依赖锁。

## English

This repository studies residual-error propagation after agent recovery. The first unit implements an instrumented ReAct-style agent with explicit memory, an ALFWorld text adapter, replay audits, bounded budgets and a durable model-call journal. Start with [the Linux runbook](docs/LINUX_RUNBOOK.md). Offline contract tests are engineering checks only; no real-agent benchmark results are included.
