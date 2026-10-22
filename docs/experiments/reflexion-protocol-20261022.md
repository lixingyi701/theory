# Reflexion 接入与开发实验

更新：2026-10-23。沿用 main 确定的研究问题，先完成原生反思的开发能力检查，再运行同状态分叉 pilot。正式校准和封存验证依赖这一阶段的事件覆盖与方差。

当前工程入口采用[接口规范化](interface-normalization-20261023.md)中的模型与工具契约。旧工具接口的 24 任务 pilot 已完成；语法重放解释了六条失败命令，随后在同采样下将 dev12 从 8/12 提高到 10/12。原目标绑定四例保留为预览，新的分叉从修复后的任务前缀选取。

## 方法与接口

上游为 noahshinn/reflexion，固定提交 `218cf0ef1df84b05ce379dd4a8e47f17766733a0`。原始源码与 MIT 许可保存在 `third_party/reflexion/`，receipt 保存逐文件 SHA256。适配器从这些文件编译作者的 actor、EnvironmentHistory、反思 query 和 update_memory，替换模型与环境 I/O。

模型替换为固定 Qwen3-8B BF16。actor 与 reflector 均采用 utils 的 256 输出 tokens；actor newline stop，短输出按作者代码最多六次、温度 0 至 1；reflector 温度 0、无 stop。保留完整历史、最近三条反思、重复动作结束和每次最多 49 决策。外层在成功后跳过后续尝试，失败后生成反思。

当前执行配置为 `qwen3_non_thinking_actionline`：把作者完整输入放入固定 tokenizer 的单 user 非思考模板，再通过 raw completion 传输；仅在 actor 输出带前导 `> ` 时移除一个行标记。journal 保存实际发送的输入与原始响应，validation 保存解码器与解码后 hash。作者算法函数未改写。模板文件为 `configs/qwen3_8b_non_thinking_wrapper.json`，已与固定 tokenizer 的六条实际查询逐字核对。

早期 raw 配置采用十次尝试，后续接口复核与新场景 pilot 采用三次尝试；所有配置在独立目录中记录。模板但无行标记解码的诊断也单独保存。不同配置的运行目录不交叉续跑。

与上游的运行差异包括：每个任务独立环境、固定现代 ALFWorld 与语法别名、显式请求种子、持久 journal，以及请求失败后停止并检查。上游函数以 done 返回 success，外围同时核对 native won；若二者不同，停止该执行并检查。原始算法函数保持原文。

## 初始批次与接续预算

初始 raw dev12 先完整执行一个任务检查反思与重试，再完成全部 12 个，每任务最多十次尝试，能力检查最多 4000 次 dispatch。此后的分叉与接口诊断纳入同一批次，合计上限为 5200 次请求、从首次启动计两小时。134 holdout 不用于开发。

GPU 0 原有 theory 服务空闲，切换至独立 Reflexion 服务，loopback 8001、单请求、eager、无 prefix cache、32768 上下文、显存比例 0.55。GPU 1–3 的训练维持原进程。

代码在独立研究 checkout 中运行，原执行代码基于 `5b58de8`，对应 main `f833a6d` 的同一套执行源；新增模块、driver、环境审计、模型和上游源身份在 run.json 中固定。

## 分叉 pilot 的进入条件

失败尝试能够生成非空反思，no-hook／等内容 sham 的调用输入一致；原生反思中存在有公共观察证据支持的事实命题。选择规则依据事实是否合格，不能依据后续成功率挑选。

分叉在下一尝试起点，任务、环境种子、初始观察、历史初始化、控制器、剩余尝试／决策预算相同；只改变新增反思的指定命题或保留策略。每个分支独立未来请求，同 payload 的等内容对照配对种子，缓存不跨分支复用。记录目标错误的存储、可见、行为采用与下一次写入，分别汇总自然结果和目标注入后的结果。

本轮 pilot 用于检验机制可观测性与正式采样设计。校准模型、状态定义与测试恢复条件在正式采样前冻结。

## 新场景 pilot 与接续

`repro/reflexion-20261022/pilot24.json` 固定六类各四个任务、24 个独立场景，排除 dev12、旧 pilot、holdout 和已识别的示例场景。完成 dev12 能力检查及边界 pilot 后，在通过新清单的 direct／官方 Gym 两项恢复审计的前提下运行新场景 pilot。每任务三次尝试，用于调查合格反思和事件覆盖。

初始批次在新场景第五个任务途中用尽 5200 次请求，所有响应均已保存。2026-10-23 用户要求继续后，`scripts/continue_reflexion_pilot.py` 记录追加最多 6000 次请求、三小时的接续合同。合同保存既有各阶段成本、原 pilot 身份 hash、源文件身份和截止时间；保留原 run.json，从既有 checkpoint 与响应重放。每个新增请求同时检查接续总额度和截止时间，已经保存的请求不重复派发。

## 目标绑定分叉

新一轮单列 `native_plan_goal_destination`。在固定 pilot 顺序中，检查首次尝试的公开任务及失败后原生反思，取前四个具有正确 put/place 目的地绑定的任务。用同一场景公开可见的另一容器替换一次目的地实体；若包含编号，替换编号也必须是该场景中的有效实例。其他文本保持逐字不变，保留反思中其他正确目的地提及，并记录这一信息冲突。

`scripts/prepare_reflexion_goal_targets.py` 不读取 trial-01 及以后的文件或任何分支结果。先预览并核对公开证据，再冻结目标 registry。目标绑定来自公开任务要求；前一轮对象位置事实来自公开观察，两类分别汇总。

`scripts/run_reflexion_pilot_forks.py` 从同一个首次失败边界运行五种条件，各有两次后续尝试。初始环境观察必须与父节点一致，模型、模板、作者函数、种子、控制器与剩余预算均继承父配置。原始、sham 与定点纠正采用相同输入但独立真实请求，汇总中比较完整 payload 序列与轨迹 hash。单个分叉批次最多 3500 次请求，且仍受追加合同的总额度约束。

## 服务器命令

从独立研究 checkout 根目录执行，使用已有 runner Python 和 `PYTHONPATH=src`：

```bash
python scripts/run_reflexion.py --output runs/reflexion-dev12-20261022
python scripts/run_reflexion_forks.py \
  --parent-run runs/reflexion-dev12-20261022 \
  --target repro/reflexion-20261022/target.json \
  --output runs/reflexion-forks-20261022
python scripts/run_reflexion_pilot.py --output runs/reflexion-pilot24-20261022
python scripts/summarize_reflexion.py \
  --parent-run runs/reflexion-dev12-20261022 \
  --forks runs/reflexion-forks-20261022 \
  --output runs/reflexion-progress-20261022.json
```

上述命令对应初始 raw 批次。当前接续使用：

```bash
python scripts/continue_reflexion_pilot.py --initialize --additional-calls 6000 --seconds 10800
# 同一接续合同再次恢复时去掉 --initialize。
python scripts/prepare_reflexion_goal_targets.py --output runs/reflexion-continuation-20261023/target-preview.json
# 核对公开任务和目标句后，使用新路径加 --freeze 固定目标，再运行：
python scripts/run_reflexion_pilot_forks.py \
  --parent-run runs/reflexion-normalized-prefix24-20261023 \
  --manifest repro/reflexion-20261023/pilot-prefix24-v2.json \
  --targets repro/reflexion-20261023/goal-targets.json --revision-kind model_profile
```

初始 raw 服务由 `serve_reflexion.py` 保存；当前 GPU 服务使用 `scripts/deployment/serve_reflexion_profiles.py --model-dir models/Qwen3-8B`，分别实现作者解码与 Qwen 推荐非思考采样。`run_reflexion_tool_stage.py` 检查模型／环境报告与整个接续额度，`continue_reflexion_normalization.py` 接续配置比较、工程冻结和新前缀采样。环境审计文件与数据位于同一服务器。原始请求留在 `runs/`，脱敏汇总进入 `repro/`。
