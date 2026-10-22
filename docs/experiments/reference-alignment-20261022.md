# 2026-10-22：改用作者代码的参考基线

## 修正的实验定位

之前的零样本Agent和`src/agent_reliability/react.py`都由本项目实现，后者虽复用了示例和历史设计，也不等于直接运行作者Agent。10月19日base6/12、memory5/12、sham5/12仅属于这些历史适配器，不能作为本次参考基线的成绩。停止沿该自建记忆Agent继续调优；旧记录保留，不续跑、不混分母。

本次保留最初的固定Qwen3-8B BF16骨干。**8B是语言模型规模；ReAct是Agent方法；ALFWorld是仿真环境。** 默认研究入口改为直接调用作者发布的ReAct ALFWorld执行函数。不把本项目额外设计的16条结构化记忆当成标准Agent的组成部分；原版ReAct只建立基础能力和上下文载体的对照。

## 证据与选择

- [ReAct论文](https://arxiv.org/html/2210.03629v3) 与 [作者代码](https://github.com/ysymyth/ReAct/tree/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9)：存在公开ALFWorld notebook、示例及环境配置，可固定代码后直接复用，故选作主参考。
- [作者README](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/README.md)区分PaLM主文与GPT-3公开代码；原公开模型为text-davinci-002，不能把我们的8B结果直接与其78.4%作等条件比较。
- [RIR §5.1](https://arxiv.org/html/2609.18304v1)使用Qwen3-14B/DeepSeek-V3与AgentScope。同日检查论文链接及公开搜索未定位其精确Agent源码/提示与完整配置；不能仅安装AgentScope或自建ReAct就称复现RIR。若后续做严格RIR对齐，需要取得其具体代码/配置并单列14B骨干，不把8B结果混入。

## 作者代码的使用方式

原始文件保存在`third_party/react/`，保留MIT许可、固定commit、逐文件hash。`reference.py`只加载白名单函数和作者原始的prompt/初始观察赋值语句；`alfworld_run`及`process_ob`不改写。运行记录保存两个函数的原始源码hash。

| 项目 | 本次处理 |
|---|---|
| Agent循环 | 原`alfworld_run`，49轮上限；不由我们另写策略循环 |
| thought | 按原代码送入环境，再在历史中使用`OK.`；撤回旧适配器“跳过环境step”的做法 |
| 示例/顺序 | 原`react_{type}_1`后接`_0`，使用原赋值语句 |
| 历史 | 原函数内部累积完整action/observation；无recent-8截断或额外memory updater |
| 解码 | 原温度0、top_p1、100 token、newline stop、频率/存在惩罚0 |
| 模型接口 | 换成固定Qwen3-8B raw completion；不注入chat模板、不剥除模型输出提示符、不修复策略文本 |
| 环境 | 官方`AlfredTWEnv.init_env`及batch接口，使用作者base_config；开发模式仅覆盖所选数据目录与seed |
| 数据差异 | 固定现代ALFWorld源码/数据；仅在环境边界把原示例`put X in/on Y`映射到该数据的`move X to Y`，不读取真值或admissible来选动作 |
| 观察/真值 | 策略只见官方公开观察；审计/真值置于独立evaluator，不能填入提示 |
| 日志/续跑 | 外围journal和环境观察器；重启从reset重放已缓存请求，核对committed checkpoint，不自动重发未确认POST |

因此，本次是**作者算法代码固定、8B替换骨干的参考实验**，具有可追溯的算法对照；不是原论文模型/环境全条件成绩复现，也不由单一参考Agent证明通用性。后续通用性需在第二个有公开实现的Agent/恢复方法上检验，不能靠自建更多变体宣称。

## 验证顺序与本次范围

1. 全量迁移文件校验；服务器重建runner，不复用Ubuntu26编译的二进制。
2. 19项离线契约/原函数回归；真实direct backend 10任务/40恢复；官方Gym再做相同10任务/40前缀与固定后缀的公开输出对照。事实层审计只由独立backend提供，不冒称已获取官方接口的所有隐藏状态。
3. 先在原冻结dev12清单的第1任务做边界暂停smoke，记录完整12分母，确认不依赖本机隧道/代理；可用后同配置resume完成12，不挑任务。
4. 运行前保留6/12且至少4类成功作为工程能力门槛；它是资源释放条件，不是论文理论。如果前6全失败，停止剩余任务并标明缺失。未过线先做模型/接口可追溯诊断，不改作者策略、不按结果删样本。
5. 本次不恢复旧memory/sham矩阵，不启动注入/R0–R3；134 unseen继续封存，正式实验协议另行冻结。下一步先固定具有原生记忆载体的公开Agent及外部干预接口，验证“无干预/等内容干预”不改变作者输入，再实施配对因果实验。

## 持久记忆假设需要另一项前置核验

原版ReAct没有与活动历史独立的持久M；不能把7/12解释为残余记忆理论已可检验，更不能把同一段历史人为分成C和M就宣布识别了两种载体。

已核对[官方Reflexion源码](https://github.com/noahshinn/reflexion/tree/218cf0ef1df84b05ce379dd4a8e47f17766733a0/alfworld_runs)，固定commit及四个文件hash见`repro/reference-20261022/reflexion-source-review.json`。`main.py`维护逐任务memory，失败后由`generate_reflections.py`写反思；`alfworld_trial.py`把最近三条记忆交给新尝试的`EnvironmentHistory`。这个原生载体比自建16条事实表更适合检验跨尝试保留/清理效应。

这是下一实施候选，尚未运行：作者actor带重复动作停止和短输出升温调用；反思有独立提示/API；作者函数把done当成功，必须先核对固定环境里是否可能done但未won。不能直接移植本轮ReAct成绩或绕过原生won核验。还需固定完整依赖/示例/工具函数、模型替换和总尝试预算，通过其自身基础能力及sham门槛。反思只在尝试边界更新，理论的时间粒度与可识别参数必须据此修订；不能照搬逐动作回写率β。若后续只用ReAct，则研究主张应收窄为上下文污染，放弃独立持久记忆机制的结论。
