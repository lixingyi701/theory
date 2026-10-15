# Linux 执行手册：恢复审计与24任务 clean/sham

本手册用于明后天在Linux服务器上执行。今天只生成了代码与配置，下面真实环境和模型命令尚未执行。第一轮目标是确定实验系统是否可用，不是检验最终模型或争取SOTA。

## 0. 约定、资源与运行顺序

- 所有命令从仓库根目录执行；安装环境采用conda示例。
- CPU端runner：Python 3.10；模型服务：单独Python 3.11，vLLM 0.10.2。
- 8B BF16权重约16GB量级，另需KV cache和运行开销。优先使用24GB或更大显存卡；这不是已测得的最低显存要求。14B或较小显存机器可通过多卡TP处理，先记录实际配置。
- 需要Linux构建工具、Git、可访问GitHub/Hugging Face。TextWorld PDDL依赖的安装以目标机器实际日志为准。
- runner与server不要安装在同一个conda环境，尤其不要为了ALFWorld替换服务端torch。
- 顺序：**离线检查→安装/下载文本数据→冻结24任务→10任务40次恢复检查→下载模型/服务→48个clean/sham episode**。

建议在tmux中分别维护`runner`和`server`窗口。没有GPU时仍可先完成数据准备和无LLM恢复审计。

## 1. 拉取仓库与runner环境

```bash
git clone https://github.com/lixingyi701/theory.git
cd theory
conda create -n theory-runner python=3.10 -y
conda activate theory-runner
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/runner.txt
bash scripts/offline_checks.sh
agent-rel doctor --output runs/setup/runner-environment.json
python -m pip freeze > runs/setup/runner-freeze.txt
```

如果TextWorld/fast-downward构建失败，先保存报错并检查编译器、cmake和Python版本。Ubuntu系统包的典型准备命令是：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl
```

这一步需要服务器管理员权限；已有依赖时无需重复执行。当前版本组合尚未在目标Linux机器联调，首次安装日志是实施证据，不应直接标记“依赖锁验证通过”。

## 2. 下载文本数据

先看计划：

```bash
python scripts/download_alfworld.py --data-root data/alfworld
```

确认后执行：

```bash
python scripts/download_alfworld.py --data-root data/alfworld --execute
```

脚本只下载官方JSON和TextWorld-PDDL归档，不下载Mask R-CNN、BUTLER权重或视觉后端。保存归档大小和SHA256；这是本次下载的指纹，不是上游发布的独立校验和。归档下载中断后重新执行会重下该未完成文件，已经完成的归档会复用。

若使用官方下载器也可运行：

```bash
alfworld-download --data-dir data/alfworld
```

它还会下载视觉检测器，纯文本任务并不需要它。两种方式选择一种即可。

## 3. 冻结24个基础任务

```bash
agent-rel manifest \
  --data-root data/alfworld \
  --split train --per-type 4 --seed 20261015 \
  --output manifests/pilot24.json
```

清单选择六类各4个任务，优先要求24个不同floor_plan场景，记录game/traj hash。实例来自官方train split；不训练LLM。

实际数据加载数以manifest的`eligible_count`和过滤计数为准。若归档解压结构与预期不同导致找不到任务，检查`traj_data.json`、同目录`game.tw-pddl`及路径中的`train`层。不要手工编造任务清单或把缺失实例当作失败。

manifest路径存在会拒绝覆盖。修改选择规则时使用明确的新清单名并记录原因；论文正式数据划分不要沿用开发中随意挑选的集合。

## 4. 无LLM的10任务恢复审计

```bash
agent-rel audit \
  --data-root data/alfworld \
  --manifest manifests/pilot24.json \
  --tasks 10 \
  --output runs/audit/replay10.json
```

审计分层选10个任务；用evaluator-only planner生成参考动作，随后在**不开planner的执行后端**重放该参考轨迹，检查两种接口设置不改变记录的状态。每任务取两个不同的中间检查点，各恢复两次，目标共40次恢复检查，并执行最多2步固定后缀。

检查项包括逐步反馈、规范化facts、admissible commands、reward/won/terminated、moves。参考动作和全部审计信息均不进入LLM提示。检查点位置按该任务参考长度确定并写入报告；不是所有任务固定在第2/6步。

通过条件：报告`passed=true`，10任务成功完成审计，`restore_checks_passed=40`，级别为`facts_and_observations`。这仍是已记录符号状态与后缀的一致性检查，不能证明全部隐藏状态数学上相等。

如果planner失败、参考路径太短或重放不一致，报告保留失败原因，退出码为1。先修复具体原因，再保存新的audit文件；不得悄悄替换任务让“10/10”成立。未通过的报告不能启动clean/sham命令。

## 5. 模型环境与下载

另开窗口：

```bash
conda create -n theory-server python=3.11 -y
conda activate theory-server
python -m pip install --upgrade pip
python -m pip install -r requirements/server.txt
python scripts/download_model.py --model qwen3_8b
python scripts/download_model.py --model qwen3_8b --execute
```

资源已经固定：

| 模型 | 官方链接 | revision | 本地目录 |
|---|---|---|---|
| 主模型 | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `b968826d9c46dd6066d109eabc6255188de91218` | `models/Qwen3-8B` |
| 后续复核 | [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) | `40c069824f4251a91eefaf281ebe4c544efd3e18` | `models/Qwen3-14B` |

`snapshot_download`使用完整revision，并核对权重索引所列shard存在，保存下载receipt。当前不自动做多GB权重的第二遍独立checksum验证。14B可用相同命令加`--model qwen3_14b`，首轮不用下载。

如果HF访问需要本地配置token，使用服务器已有的登录/环境设置，不把token写进仓库。公开链接失败时先核验连通性与对应revision，不要自动换模型。

## 6. 启动本地模型服务

```bash
bash scripts/serve_model.sh qwen3_8b --plan
nvidia-smi
mkdir -p runs/server
python -m pip freeze > runs/server/server-freeze.txt
bash scripts/serve_model.sh qwen3_8b --execute 2>&1 | tee runs/server/qwen3-8b.log
```

服务绑定`127.0.0.1:8000`，模型名`qwen3-8b`；本地回环不向公网暴露。BF16、max-model-len=8192、max-num-seqs=4、GPU利用率0.85；这些是初始部署配置，需按实际显卡确认。多卡示例：

```bash
RELIABILITY_TP=2 bash scripts/serve_model.sh qwen3_8b --execute
```

若OOM，优先检查显存占用、并发和上下文配置。任何量化或精度更改都要作为新协议版本记录。`--generation-config vllm`避免隐式套用模型自带采样默认值；每次请求显式指定non-thinking和采样参数。

服务端无须reasoning parser，因为本阶段关闭thinking。`/v1/models`检查只能验证served name，不能独立证明加载了哪个revision；需要同时保留下载receipt、启动命令和server日志。

## 7. 24任务 clean/sham

切回runner窗口：

```bash
conda activate theory-runner
export RELIABILITY_API_BASE=http://127.0.0.1:8000/v1
export RELIABILITY_API_KEY=local-development
agent-rel run \
  --data-root data/alfworld \
  --manifest manifests/pilot24.json \
  --audit-report runs/audit/replay10.json \
  --config configs/clean_sham.json \
  --output runs/clean-sham24
```

当前配置：24任务×1 seed×2条件＝**48个episode**；每个episode最多80次环境动作、164次实际模型请求，模型输出格式最多重试一次且计费。最多请求上界48×164=7,872，实际由成功/终止与格式情况决定；不是预估API费用或实际调用量。

Agent每轮先更新结构化记忆，再生成一个行动；活动历史最近8轮，记忆16条。所有环境动作计预算，包括无效动作。模型请求、环境动作、tokens分别记录。回滚恢复world不返还runner预算。

首次真实联调时保留24任务清单，启动后观察首个episode日志，发现接口错误便停止修复，使用同一run的`--resume`接续。配置/提示已更改则用新run目录；禁止用旧run混合新协议。不能通过缩小清单绕过10任务审计门槛。

### sham具体是什么

本单元采用**已验证当前位置事实的等内容替换**：第3–12动作的记忆更新后，evaluator发现一条与当前facts一致的location事实时，复制并替换同一事实，内容、来源和顺序保持相同。无额外LLM调用，不给策略任何“这是sham”的标签。

这是检查干预hook、真值隔离与条件分支的工程sham。它不是语义改写，也不能作为“LLM对无害语言改写鲁棒”的实验。clean与sham使用相同的逻辑随机种子；在确定性服务下轨迹应相同。服务端仍可能有数值非确定性，不能要求逐token保证。

窗口内没有可验证位置事实时记录`sham_applied=false`，保留该任务在总体结果中，单报覆盖率。该设计在前置研究协议中的一般sham设想基础上作了可执行收窄；后续语言改写sham需要另建协议，不能混用。

## 8. 中断与恢复

正常中断后接续：

```bash
agent-rel run \
  --data-root data/alfworld \
  --manifest manifests/pilot24.json \
  --audit-report runs/audit/replay10.json \
  --config configs/clean_sham.json \
  --output runs/clean-sham24 --resume
```

已经完成的episode跳过；进行中的episode恢复最后一次已提交检查点，重放动作而不重新调用前缀LLM。未提交步若已有保存响应，就从缓存重建；committed checkpoint决定进度，单独存在的step文件不决定进度。

若日志出现`UncertainRequest`：请求已写入journal，但缺失响应，无法确定服务端是否已完成生成。先检查对应`calls/step-.../attempt-...request.json`及server日志。确实要重发时加：

```bash
# 与上面的完整run命令相同，追加以下参数：
# --resume --retry-uncertain
```

重发会记录一个新attempt并消耗预算。原来的不确定调用不会抹掉，token费用可能未知，报告中单列。不要删除journal让账目归零。

当前runner只支持本地可重放的ALFWorld；不保证对真实不可逆外部工具做上述恢复。相同run目录只允许一个进程执行。

## 9. 输出及检查

```bash
agent-rel summarize runs/clean-sham24
```

关键文件：

| 路径 | 含义 |
|---|---|
| `run.json` | 配置、代码hash、版本、manifest/audit/model身份、预计episode数 |
| `summary.json` | 条件分组的完成数、状态、成本；未完成不冒充完整分母 |
| `episodes/<id>/identity.json` | 任务、种子、条件及模型/配置身份 |
| `checkpoint.json` | 已提交的world replay前缀、活动上下文、显式记忆、预算进度 |
| `steps/*.public.json` | 策略可见信息、记忆变化、行动、调用/格式情况 |
| `steps/*.evaluator.json` | 真实facts、动作合法性、sham记录；不用于提示 |
| `calls/` | 每个逻辑调用的请求、原始响应、usage、延时、验证结果 |
| `result.json` | 单episode结果；基础设施异常时不伪造该文件 |

六种状态/异常要区分：success、environment_failure、agent_format_failure、timeout_env_budget、timeout_llm_budget，以及未完成的基础设施异常。上下文超过服务上限、连接失败等属于基础设施问题，先修接口。

本轮要汇报：审计通过情况；48个episode是否完成；24个clean中成功数；sham实际覆盖；格式/截断率；每任务调用和tokens；失败例子与是否具备后续注入位置。不能从这轮得出残余污染规律。

## 10. 本轮之后

clean至少能稳定完成一部分任务、没有接口问题之后，才实施根错误注入与R0–R3。若clean24成功少于8，先审查提示、动作格式、预算和环境接口；该阈值是开发决策，不是统计检验。初始Agent没有加入ReAct原论文的两条few-shot示例；若需要添加，从train数据构造并固定，另起协议/run。

参考：[HF下载文档](https://huggingface.co/docs/huggingface_hub/guides/download)、[vLLM 0.10.2服务文档](https://docs.vllm.ai/en/v0.10.2/serving/openai_compatible_server.html)、[TextWorld 1.6.2 start接口](https://github.com/microsoft/TextWorld/blob/1.6.2/textworld/helpers.py)。
