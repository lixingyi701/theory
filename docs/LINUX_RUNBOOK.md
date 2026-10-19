# Linux 执行手册：先验证基础能力，再比较仪器化

更新：2026-10-19。环境已部署；旧27/48全失败pilot归档，不能再作为默认起点或在改代码后续跑。资源、安装freeze、SSH/systemd与服务启动见 [部署记录](deployment/README.md)。

当前完成结果见 [门槛报告](experiments/results-20261019.md)。以下命令用于可追溯的协议执行；不要把原在线v2目录续跑到新profile上。当前8000为串行诊断profile，若重现在线v2需先按部署文档切回原profile并使用新目录。新增任务验证需先冻结新的记忆/服务身份检查。

## 1. 核对证据与协议

阅读AGENTS.md、[实施状态](IMPLEMENTATION_STATUS.md)、[基础方案v1](experiments/baseline-20261019.md)及[chat适配v2](experiments/baseline-20261019-v2.md)。这是Qwen适配的公开ReAct预检，不是严格论文复现。所有命令从仓库根目录执行，先激活runner环境。

```bash
conda activate theory-runner
bash scripts/offline_checks.sh
agent-rel doctor --output runs/setup/current-runner.json
python -m pip check
```

## 2. 数据与固定清单

```bash
python scripts/download_alfworld.py --data-root data/alfworld  # 先显示计划
python scripts/download_alfworld.py --data-root data/alfworld --execute
```

`repro/baseline-20261019/dev12.json` 是冻结的开发12任务，六类各2、12场景；完整134任务 `holdout134.json` 继续封存。不要基于结果重选任务。复算选择仅用于核对：

```bash
PYTHONPATH=src python scripts/prepare_baseline.py --data-root data/alfworld \
  --pilot repro/legacy-pilot24.json --output manifests/recomputed-baseline-20261019
```

公开示例缺少task/scene ID，不能证明开发集与示例场景完全独立；开发集不估计泛化成绩。

## 3. 真实恢复审计

```bash
agent-rel audit --data-root data/alfworld \
  --manifest repro/baseline-20261019/dev12.json --tasks 10 \
  --output runs/audit/react-dev12-new.json
```

必须10任务/40恢复全通过，且事实与输出重放一致。审计使用的planner仅供evaluator；不进入策略/记忆提示。源代码或backend版本改变时按CLI校验重做。不要覆盖旧失败审计。

## 4. 模型服务与基础 Agent

远程只在Conda theory中按部署手册启动服务。runner通过SSH的本地loopback访问：

```bash
export RELIABILITY_API_BASE=http://127.0.0.1:8000/v1
curl --fail "$RELIABILITY_API_BASE/models"
agent-rel react-run --data-root data/alfworld \
  --manifest repro/baseline-20261019/dev12.json \
  --audit-report runs/audit/react-dev12-new.json \
  --config configs/react_base.json --output runs/react-base-NEW
```

先检查base完整分母与各类型覆盖。6/12成功且至少4/6类型成功才进入下一步。前6任务全失败会提前停止，剩余任务标记缺失。任何不确定POST都先检查journal，不自动重试。需要显式重发时，额外dispatch计入成本，并保留旧尝试。

## 5. 记忆与sham对照

仅在上述能力门槛通过后运行：

```bash
agent-rel react-run --data-root data/alfworld \
  --manifest repro/baseline-20261019/dev12.json \
  --audit-report runs/audit/react-dev12-new.json \
  --config configs/react_instrumented.json --output runs/react-instrumented-NEW
```

同模型、同示例/历史/seed/动作预算，比较memory相对base逐任务损失、实际调用/token成本，以及sham是否触发、是否产生完全相同提示和轨迹。详见预先方案门槛。不要把equal-content sham写成语言改写或错误注入。

## 6. 续跑和研究分界

只有完全相同的source hashes/config/model/task/audit可以 `--resume`。run.json 是运行身份，checkpoint.json 是已提交进度；未提交step文件不代表完成。不要启动两个进程共用目录。

基础能力/仪器化门槛失败先诊断；不得直接扩大任务或启动错误注入。通过后还需锁定正式任务范围与留出协议，再实现配对注入和R0–R3、短任务拟合与长度外推。软件测试、真实replay、LLM能力预检和理论实验必须分别报告。

历史 `agent-rel run` / `configs/clean_sham.json` 仍保留供原版本审查，不作为本轮有效baseline。原run目录和v1适配smoke均不续跑。
