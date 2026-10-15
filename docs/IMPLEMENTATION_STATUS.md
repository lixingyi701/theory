# 实施交接：第一单元

更新时间：2026-10-15。下一位执行者：用户的Linux服务器上的Codex。

## 已完成

- Python包与CLI：doctor、manifest、audit、run、summarize。
- ALFWorld直接TextWorld适配器，关闭domain randomization，请求facts供独立审计。
- 逐步显式记忆、确定性淘汰、最近8轮上下文。
- 环境动作与模型请求分别限额；格式重试计入模型调用。
- checkpoint包含环境前缀、上下文、记忆和已用动作数；逐步重放校验。
- 持久请求journal：保存响应可复用，不确定请求显式重发并保留成本。
- 10任务审计协议与48个clean/sham episode入口。
- 模型固定revision、文本数据下载计划、模型服务启动脚本。
- 两份前置研究文档上传至`docs/research/`。

## 本机实际检查

在Python 3.12标准库环境执行`bash scripts/offline_checks.sh`：10项离线契约测试通过，源码与脚本编译通过，模型/数据下载及模型启动均为plan-only。

覆盖：facts漂移与checkpoint篡改检测；10任务/40次restore gate的软件逻辑；原子记忆更新/淘汰；请求缓存/不确定请求/调用限额；clean/sham runner与oracle字段隔离；中断续跑；格式失败区分；动作与调用预算区分；manifest过滤/冻结/完整性；本地fake HTTP端点的请求字段与模型名检查。

测试中使用的fake世界与fake模型不是ALFWorld或Qwen，不提供任务成功率、恢复确定性或模型机制证据。

## 尚未执行

- 目标Linux依赖安装及完整freeze。
- 真实benchmark数据/模型权重下载。
- ALFWorld原生库、planner与replay实际联调。
- Qwen3/vLLM请求、显存/吞吐测量。
- 真实10任务恢复检查、24任务clean/sham。

## 协议落实时的明确收窄

1. 本单元sham是当前真值验证后的等内容事实替换，检查hook流程；不是语言改写实验。
2. 记忆暂只实现结构化事实，不实现独立的失败反思条目。
3. 当前提示是零样本ReAct式执行范式，未复现原ReAct的两条示例及旧模型。
4. replay审计证明记录facts/输出/固定后缀一致；不宣称完全隐藏状态等价。
5. 数据依赖直接pin，不是已在Linux验证的完整传递依赖锁。

上述收窄不支持对完整ReAct、RIR或记忆方法作性能结论。

## 明后天第一条接续指令

> 阅读AGENTS.md和Linux执行手册，检查服务器资源，安装两个独立环境，下载文本数据并冻结pilot24清单。先执行10任务恢复审计；通过后部署固定revision的Qwen3-8B，再执行24任务clean/sham。保留失败与运行指纹，汇总工程结果；暂不启动错误注入实验。

## 下一代码单元

在真实clean/sham通过后实现：可核验根事实注入、固定传播父轨迹、R0–R3分支、回滚后记忆有效性处理、目标错误/派生后果标注。再实现Dependency-Workflow与预测拟合。完整研究协议仍以`docs/research/agent_error_recovery_model_and_protocol.md`为依据。
