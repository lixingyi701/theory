# Linux 工作站 + 远程 NVIDIA 部署记录

核验日期：2026-10-19。真实路径、内网地址、SSH身份文件和账户缓存保留在本机操作记录；下列路径用占位符表示，便于公开复现。

## 当前机器与环境

| 项目 | 本机 runner | 远程模型服务 |
|---|---|---|
| OS | Ubuntu 26.04.1 LTS / kernel 7.0.0-38 | Ubuntu 20.04.6 / kernel 5.4.0-216 |
| CPU / RAM | Core Ultra 9 285H，16逻辑核，约30 GiB | 约125 GiB RAM |
| GPU | Intel 集显，无 NVIDIA | 4 × A800 80GB PCIe；本任务只用 GPU0 |
| 隔离环境 | 项目内 `theory-runner`，Python 3.10.22 | Conda **theory**，Python 3.11.15 |
| 核心依赖 | ALFWorld 0.5.0 / TextWorld 1.6.2 / numpy 1.26.4 | vLLM 0.10.2 / torch 2.8.0 + CUDA12.8 / transformers 4.57.6 |
| 编译工具 | GCC12 / CMake3.31.8 / fast-downward-textworld20.6.4 | NVIDIA driver570.133.20，glibc2.31 |

实际版本见 [本机快照](local-snapshot-20261019.json) 和 [远程快照](server-snapshot-20261019.json)。两端 `pip check` 通过。远程原有 vLLM0.21.0 环境未用于本实验，也未被修改。2026-10-19重新比对原始部署前的 base Conda history SHA256 和 base pip freeze，两者均未变化；这个检查不代表机器全部系统文件都未改变。

模型固定为 Qwen/Qwen3-8B，revision `b968826d9c46dd6066d109eabc6255188de91218`，BF16五个分片共16,381,516,776 bytes。存放 `/data02/<user>/theory/models/Qwen3-8B`；核验时 data02 尚余约229 GiB。下载因网络限制走官方 Qwen ModelScope 固定文件revision，所有文件再匹配固定HF revision的LFS SHA256或Git blob身份；[公开下载凭据](model-receipt-20261018.json)只含文件身份，无账户数据。

在线v2 ReAct 预检服务为 loopback `127.0.0.1:8000`，32768 context，BF16，TP1，最多4 sequences，显存利用率0.85，prefix caching。原失败 pilot 服务 context8192；本轮扩展在模型原生40960位置上限内。在线启动模板是 `scripts/deployment/serve_react.sh`。在线确定性门槛未过后，当前切换为 `scripts/deployment/serve_serial.py` 的同步离线候选profile：仍在同一环境/GPU0/BF16/32768 context，单请求、eager、无prefix cache；36请求有限预检一致。该切换和V1实际prefill行为见 [诊断记录](../experiments/serial-serving-20261019.md)。没有同时占用其他卡。

## 复建环境

从仓库根目录执行。显式Conda清单包含linux-64具体构建及MD5（服务端也保留上游URL）；pip清单固定全部已记录版本，但未提供每个wheel的hash，属于已部署环境的版本freeze，不是跨平台/跨时间的完整二进制可复现性证明。

```bash
conda create -n theory-runner --file requirements/locks/runner-linux-64.conda.txt
conda activate theory-runner
python -m pip install -r requirements/locks/runner-pip.txt
python -m pip check
bash scripts/offline_checks.sh

# 在远程执行；不能安装进base，也不能混用runner依赖。
conda create -n theory --file requirements/locks/server-linux-64.conda.txt
conda activate theory
python -m pip install -r requirements/locks/server-pip.txt
python -m pip check
```

runner清单包含GCC12构建工具；本机最初是通过独立micromamba prefix创建，同样不依赖系统Python。实际部署经历过TLS下载失败与显式归档安装，安装日志保留在本地；已验证freeze不等于保证另一网络首次安装必成功。依赖来源和版本有变化时，另存新快照与审计，不覆盖旧记录。

```bash
# 模型在远程下载并验证后，在tmux里启动：
conda activate theory
export THEORY_MODEL_DIR=/data02/<user>/theory/models/Qwen3-8B
bash scripts/deployment/serve_react.sh  # 重现在线v2预检服务
# 当前串行诊断profile（两者不能同时占8000）：
# CUDA_VISIBLE_DEVICES=0 python scripts/deployment/serve_serial.py --model-dir "$THEORY_MODEL_DIR"
```

## SSH 长连接与恢复

`ssh_config.example` 是脱敏模板，需在本机替换服务器/用户/密钥路径。host key 先通过可信渠道核实并加入 known_hosts；之后开启严格校验。key权限600，SSH目录700。不要上传私钥、公钥授权内容、密码或Codex登录缓存。

```bash
install -m 600 scripts/deployment/ssh_config.example ~/.ssh/theory.conf
# 编辑 ~/.ssh/theory.conf 后：
install -Dm644 scripts/deployment/theory-ssh.service ~/.config/systemd/user/theory-ssh.service
systemctl --user daemon-reload
systemctl --user enable --now theory-ssh.service
systemctl --user status theory-ssh.service
curl --fail http://127.0.0.1:8000/v1/models
```

任务专用的实际服务已开启：SSH keepalive15秒 ×3，断线后5秒重启，曾测试强制断开后的重连。校园内网路由受本机VPN TUN影响时，通过实际物理网卡的 BindInterface 连接。工作站睡眠/关机/断网期间隧道不可用，网络恢复后自动重建；远程tmux服务独立存活。runner journal 不会自动重发断网时未确认的POST。

远程 Codex CLI 0.163.0 已安装并验证 ChatGPT 登录与一次只读READY请求；该验证发生在2026-10-18，本轮未额外调用Codex模型。远程CLI依赖可选反向loopback代理17897→本机代理7897，因此不是脱离工作站的独立联网方案。登录凭据不进入仓库。

本仓库当前runner使用Linux `fcntl` 与TextWorld/PDDL原生依赖；原生Windows没有验证。Windows可作为工作站，通过WSL2/Linux runner连接同一远程服务。不能据此声称原生Windows直接可跑。

## 数据与证据

文本数据约1.8 GB，不含视觉检测器。两个官方归档SHA256：

- JSON：`25171f16e20ad7b048c47275c45b0babf3aa1cbab29cec97387922350a9844bc`
- PDDL：`5df77ea759f2211a4106082839ddbbb790f1ba4e7d097ed732cf453f72aa36cf`

`repro/` 提供任务清单与hash；`docs/experiments/` 提供预定方案和脱敏汇总。原始请求/完整模型响应、oracle审计轨迹、权重与数据保存在本地/远程 `runs/`、`models/`、`data/`，不提交Git。
