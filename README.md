# Agent reliability: residual errors after recovery

研究目标：解释并预测长程 LLM Agent 在不完全恢复之后的错误复发与任务结果。当前仓库提供**第一实施单元**：ALFWorld 文本环境、显式任务内记忆、可续跑的执行器和 checkpoint/replay 审计。

**2026-10-22：改为服务器完整运行，默认入口 `reference-run` 直接执行固定作者ReAct ALFWorld函数，保留Qwen3-8B BF16骨干。开发12任务成功7个、覆盖5类，通过基础能力门槛，见[结果](docs/experiments/results-20261022.md)。** [方法对齐依据](docs/experiments/reference-alignment-20261022.md)逐项说明模型/环境差异及独立记忆的研究边界；[部署记录](docs/deployment/README.md)说明远程runner与原生库重建。历史适配器成绩单列，未启动因果矩阵。

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
| 作者参考基线 | `agent-rel reference-run` | 原作者函数/示例/完整历史/49轮/原始completion解码，外围记录与恢复 |
| 历史Agent runner | `agent-rel run` / `react-run` | 自建clean/sham及chat适配；需显式允许legacy，不作为默认baseline |
| 显式记忆 | `src/agent_reliability/memory.py` | 16条结构化事实、来源步、确定性淘汰 |
| 中断恢复 | `--resume` | 重放环境、恢复上下文/记忆，复用已保存模型响应 |
| 请求审计 | 每个episode的`calls/` | 保存请求/响应、种子、usage、截断和不确定请求 |
| 下载/服务配置 | `scripts/`、`configs/models.json` | 固定模型revision；默认只打印计划 |

默认基线复用作者算法代码；8B骨干、现代ALFWorld版本和固定语法别名属于明确记录的替换，**不是原论文全条件成绩复现或RIR复现**。自建显式记忆仅保留在历史路径。尚未实现错误注入、R0–R3四恢复臂、Dependency-Workflow环境、参数估计与长度外推。

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

This repository studies residual-error propagation after agent recovery. The default server-local baseline executes the fixed upstream ReAct ALFWorld functions with a documented Qwen3-8B substitution, official Gym interface, replay audits and durable request journals. Start with [the Linux runbook](docs/LINUX_RUNBOOK.md). Historical custom agents remain for audit. Engineering checks and development results do not establish the recovery theory.
