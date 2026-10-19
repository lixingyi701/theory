# 推理确定性预检与串行服务配置（新profile试运行前记录）

在线v2对照已经观察到相同payload hash（包含相同提示/seed/解码）返回不同文本，首次明确案例为开发任务 ec35b3ffe3eb225f 的 decision-0012-policy。因此在线sham不满足原定逐轨迹一致性门槛，不能把其差异解释为hook效应。

[vLLM0.10.2官方复现指南](https://docs.vllm.ai/en/v0.10.2/usage/reproducibility.html)明确指出在线serve不支持确定性复现；离线V1路径需关闭multiprocessing以固定调度条件。这支持先验证服务层，而不是根据任务输赢继续调提示。

新增候选profile `offline_serial_eager_nocache_v1`：在同一Conda theory、同一8B BF16权重、同一GPU0与32768 context下，串行HTTP适配同步 `LLM.generate`；单次一个prompt、max_num_seqs1、全局seed0、保留请求seed、无prefix cache、无chunked prefill、eager、关闭cascade attention。无输出响应缓存，不把重放结果伪装为新生成。HTTP只监听loopback。这个profile是实现选择，确定性需实测，不能从配置名直接推断。

先选12个在线memory任务中的策略/记忆请求，含已观察到分歧的payload；不按成功或失败筛选。在同一进程内以正序、逆序、正序各生成一次（36个实际请求），要求每个payload的完整文本与finish_reason三次一致。输入均为已允许模型看到的公开反馈/Agent状态；oracle字段不参与。另记成本，不能混入任务成功率。

若预检失败，不扩大任务矩阵。若通过，则在新目录完整重跑base12；仍使用原能力门槛。base通过后再跑memory/sham24，仍使用原仪器化与sham门槛。Agent提示、任务、解码、预算不改变；配置增加serving_profile身份字段，分别存为react_*_serial.json。在线v2成绩保留，不与新profile混成一个分母。

## 在线对照完成后的执行收窄（36次新profile预检调用之前）

在线v2完整结果为base6/12、memory5/12、sham5/12，memory相对base丢失2个成功、获得1个；sham只触发4/12。两项既定门槛均失败。因此原条件式的“确定性预检通过后立即重跑36个任务episode”不执行：当前只完成36请求的服务诊断与固定轨迹的真实hook重放。之后先设计影子记忆/事实覆盖诊断，再冻结新的能力与仪器化验证；不根据输赢反复重跑同一矩阵。

运行时核对：vLLM V1虽接收 `enable_chunked_prefill=False`，实际engine仍报告 `chunked_prefill_enabled=True`；不能把请求的开关写成已生效。候选配置的max_num_batched_tokens=32768，单请求输入加输出也限制在32768内；本次按实际V1行为检验，不宣称已关闭chunked prefill。eager、prefix cache关闭和单请求串行已由初始化日志确认。

## 实测结果

12个固定payload ×3种顺序，共36个实际请求，36份已保存响应、0未确认；12/12 payload 的完整输出文本及finish_reason均三次一致。[脱敏结果](serial-repro-36-20261019.json)保留输入/输出hash与实际成本，不上传完整输入或模型回复。它仅覆盖同一进程、同一GPU/版本的有限请求集，不证明跨硬件、跨重启或所有后续请求都确定。

当前远程loopback8000由这个已预检的串行profile提供，在线v2服务已停止。没有同时占用第二张卡，也没有升级vLLM、torch或base环境。记忆门槛失败后没有追加任务矩阵。
