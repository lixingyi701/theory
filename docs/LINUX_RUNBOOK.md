# Linux 执行手册：服务器上的作者参考基线

更新：2026-10-22。项目、CPU runner、ALFWorld数据、GPU模型与原始日志均在服务器。工作站用于编辑、版本控制和SSH。先读[实施状态](IMPLEMENTATION_STATUS.md)、[方法依据与差异](experiments/reference-alignment-20261022.md)、[部署记录](deployment/README.md)。历史自建Agent及失败目录不续跑。

## 1. 进入服务器与独立环境

```bash
ssh -F ~/.ssh/theory.conf theory-gpu
export THEORY_ROOT=/data02/<user>/theory
source "$THEORY_ROOT/project/scripts/deployment/activate_remote_runner.sh"
python -m pip check
bash scripts/offline_checks.sh
agent-rel doctor --output runs/setup/runner-NEW.json
```

`theory-runner`为服务器Python3.10隔离prefix，`theory`为原有Python3.11/vLLM0.10.2环境；不要混装，也不安装到base。原生planner需针对服务器glibc编译，不能直接搬用新Ubuntu的二进制。任务记录采用新目录，保持失败证据。

## 2. 数据、清单与作者源码

已有数据从迁移校验凭据核对，不重复下载。首次部署可使用默认plan-only的`scripts/download_alfworld.py`。开发集固定为`repro/baseline-20261019/dev12.json`，六类各2，12场景；134 unseen继续封存，不按结果重选任务。公开示例没有场景ID，开发成绩不用于泛化估计。

`third_party/react/receipt.json`固定作者commit和全部原始文件hash。默认入口运行原notebook的`alfworld_run` / `process_ob`函数和示例拼接语句，使用原始completion协议。禁止为提高分数改写作者策略、修复模型动作文本或把evaluator真值放进提示。

## 3. 两层真实环境审计

```bash
agent-rel audit --data-root data/alfworld \
  --manifest repro/baseline-20261019/dev12.json --tasks 10 \
  --output runs/audit/direct-NEW.json
python scripts/check_official_environment.py --data-root data/alfworld \
  --manifest repro/baseline-20261019/dev12.json \
  --audit-report runs/audit/direct-NEW.json \
  --output runs/audit/official-gym-NEW.json
```

第一层检查direct backend的facts/观察恢复；第二层比较官方Gym接口的公开观察、reward、done、won与已审计backend，10任务/40前缀及固定后缀全通过。第二层不声称获取了官方Gym全部隐藏事实。原生库hash、源码、manifest不匹配会阻止新模型运行。

## 4. 服务器本地模型与预检

在服务器另一个tmux会话中，使用既有`theory`环境启动模型：

```bash
export CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
"$HOME/miniconda3/envs/theory/bin/python" scripts/deployment/serve_reference.py \
  --model-dir "$THEORY_ROOT/models/Qwen3-8B"
```

服务只监听服务器`127.0.0.1:8000`；同一端口已有本项目服务时先核对身份，不重复启动。profile必须为`upstream_react_raw_serial_v1`；8B revision固定，BF16/32768上下文、单请求/eager/无prefix cache。其他用户/GPU上的进程不属于本项目。

```bash
curl --fail http://127.0.0.1:8000/v1/models
THEORY_ROOT="$THEORY_ROOT" bash scripts/deployment/run_reference.sh \
  --data-root data/alfworld --manifest repro/baseline-20261019/dev12.json \
  --audit-report runs/audit/direct-NEW.json \
  --environment-report runs/audit/official-gym-NEW.json \
  --output runs/author-react-8b-NEW --stop-after-episodes 1
```

在tmux中执行，所有路径/API均服务器本地，运行脚本取消代理；无需工作站8000转发。第1任务边界检查后，用完全相同参数去掉`--stop-after-episodes 1`、加`--resume`完成原12分母。不要修改source/config/model后续跑原目录。

预定工程门槛是至少6/12成功且至少4类成功；前6任务全失败则停止，未完成项报告为missing。这个门槛只验证基础可用性，不能证明理论或启动因果矩阵。

## 5. 成本、断线与下一节点

```bash
agent-rel summarize runs/author-react-8b-NEW
```

`run.json`固定身份，`checkpoint.json`表示已提交进度，`calls/`记录全部dispatch、响应、usage与未确认请求。SSH断开不影响远程tmux；未知POST不会自动重发。只有确认需要额外dispatch时使用`--retry-uncertain`，并计入成本。

作者ReAct基线通过后，先固定官方Reflexion等具有原生持久记忆的方法及其外部干预接口，通过该方法的基础能力和无干预/等内容sham一致性，再冻结配对错误注入、恢复和留出协议。ReAct本身没有独立持久记忆M；旧16条显式记忆不自动带入。错误注入、R0–R3、参数估计与长度外推尚未实现；134 unseen不调用模型。

历史`run` / `react-run`需显式`--allow-legacy-agent`，仅供审查旧实现；10月19日6/12、5/12、5/12的结果见[历史报告](experiments/results-20261019.md)，不得归入新作者参考基线。
