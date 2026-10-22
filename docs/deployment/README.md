# 当前远程运行环境

实验统一在服务器运行。项目、数据、journals 与评估器位于 `/data02/<user>/theory/project`，模型位于同一项目根下的模型存储；工作站负责编辑、Git 和 SSH。

| 组件 | 配置 |
|---|---|
| CPU runner | 独立 Python 3.10.22 prefix，ALFWorld／TextWorld |
| GPU 服务 | Conda theory，Python 3.11.15，vLLM 0.10.2，Qwen3-8B BF16 |
| 当前 ReAct profile | upstream_react_raw_serial_v1，原始 completion，100 tokens，温度 0，newline stop |
| 通信 | 服务器 loopback；tmux 内运行 |

启动与审计命令见[执行手册](../LINUX_RUNBOOK.md)。版本 freeze 位于 `requirements/locks/`；机器和迁移凭据保留在本目录 JSON 中。原生 planner 已针对服务器 glibc 重建，新宿主需再次核验。

Reflexion 接入时另设 actor／reflector 调用配置，覆盖作者的重试与输出长度需求。历史部署过程见[归档](history-20261022.md)。
