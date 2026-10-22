# Reflexion 接入与开发实验

本轮沿用 main 在 2026-10-22 确定的研究问题，首先完成原生反思的开发能力检查，再运行同状态分叉 pilot。正式校准和封存验证依赖这一阶段的事件覆盖与方差。

## 方法与接口

上游为 noahshinn/reflexion，固定提交 `218cf0ef1df84b05ce379dd4a8e47f17766733a0`。原始源码与 MIT 许可保存在 `third_party/reflexion/`，receipt 保存逐文件 SHA256。适配器从这些文件编译作者的 actor、EnvironmentHistory、反思 query 和 update_memory，替换模型与环境 I/O。

选择上游支持的 raw completion 路径，模型替换为固定 Qwen3-8B BF16。actor 与 reflector 均采用 utils 的 256 输出 tokens；actor newline stop，短输出按作者代码最多六次、温度 0 至 1；reflector 温度 0、无 stop。保留完整历史、最近三条反思、重复动作结束和每次最多 49 决策。外层采用作者脚本的 10 次尝试、成功后跳过后续尝试、失败后生成反思的规则。

与上游的运行差异包括：每个任务独立环境、固定现代 ALFWorld 与语法别名、显式请求种子、持久 journal，以及请求失败后停止并检查。上游函数以 done 返回 success，外围同时核对 native won；若二者不同，停止该执行并检查。原始算法函数保持原文。

## 本轮执行范围

仅 dev12，先完整执行一个任务检查反思与重试，再完成全部 12 个。每任务最多 10 次尝试，本轮能力检查最多 4000 次 dispatch、墙钟 7200 秒；失败请求计入成本，已有响应通过 journal 恢复。134 holdout 不用于开发。

GPU 0 原有 theory 服务空闲，切换至独立 Reflexion 服务，loopback 8001、单请求、eager、无 prefix cache、32768 上下文、显存比例 0.55。GPU 1–3 的训练维持原进程。

代码在独立研究 checkout 中运行，原执行代码基于 `5b58de8`，对应 main `f833a6d` 的同一套执行源；新增模块、driver、环境审计、模型和上游源身份在 run.json 中固定。

## 分叉 pilot 的进入条件

失败尝试能够生成非空反思，no-hook／等内容 sham 的调用输入一致；原生反思中存在有公共观察证据支持的事实命题。选择规则依据事实是否合格，不能依据后续成功率挑选。

分叉在下一尝试起点，任务、环境种子、初始观察、历史初始化、控制器、剩余尝试／决策预算相同；只改变新增反思的指定命题或保留策略。每个分支独立未来请求，同 payload 的等内容对照配对种子，缓存不跨分支复用。记录目标错误的存储、可见、行为采用与下一次写入，分别汇总自然结果和目标注入后的结果。

本轮 pilot 用于检验机制可观测性与正式采样设计。校准模型、状态定义与测试恢复条件在正式采样前冻结。

## 新场景 pilot 与接续

`repro/reflexion-20261022/pilot24.json` 固定六类各四个任务、24 个独立场景，排除 dev12、旧 pilot、holdout 和已识别的示例场景。完成 dev12 能力检查及边界 pilot 后，在通过新清单的 direct／官方 Gym 两项恢复审计的前提下运行新场景 pilot。每任务三次尝试，用于调查合格反思和事件覆盖。

整个接入与 pilot 批次合计最多 5200 次请求、从 dev12 首次启动计两小时。新场景 pilot 最多 2000 次请求、一小时，实际额度受批次剩余量约束。`scripts/continue_reflexion_campaign.py` 按实际结果接续；任务缺失、请求未确认或能力检查未过时保留状态并停止相应阶段。

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

模型服务使用 GPU 环境的 `scripts/deployment/serve_reflexion.py --model-dir models/Qwen3-8B`。环境审计文件与数据位于同一服务器，脚本默认检查本批次固定路径。原始请求留在 `runs/`，脱敏汇总进入 `repro/`。
