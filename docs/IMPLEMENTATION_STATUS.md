# 实施状态（2026-10-19）

## 已核验

- 本机独立Python3.10 runner与远程Conda **theory** / Python3.11服务环境；vLLM0.10.2。远程base环境的history和pip包表与部署前一致。实际版本、freeze、SSH模板及复建命令见 [部署记录](deployment/README.md)。
- 官方文本数据与固定revision的Qwen3-8B BF16已下载校验。工作站通过自动重连SSH连接远程loopback模型服务。
- 真实ALFWorld恢复审计已通过两次：历史pilot修复后10任务/40次；新dev12清单也是10任务/40次，均为facts_and_observations。
- 修复sham对真实 `inreceptacle` 谓词的识别；离线真实frame检查发现原先漏掉的9个候选，不回写成原pilot的干预次数。
- 新增公开ReAct示例、完整历史、think/环境动作分离、固定语法转换和chat提示符解析；保留旧执行路径供审计。新路径仍使用持久journal、checkpoint、重放与不确定请求显式处理。
- 2026-10-19离线16项检查通过。fake backend/provider只测试代码，不提供模型能力结论。
- 新base v2开发预检完成：12/12任务，成功6，成功覆盖5/6类；4个decision预算耗尽、2个输出截断。达到预先6/12且至少4类的工程门槛。memory/sham各完成12任务、成功均5；instrumentation与sham覆盖门槛失败，见[完整结果](experiments/results-20261019.md)。

## 历史失败必须保留

旧pilot只完成27/48（clean14、sham13），成功均0，余下21未运行。该自建零样本/短历史/记忆Agent不是论文ReAct或RIR。语法接口还存在 `put ... in/on ...` 未转换为固定环境 `move ... to ...` 的缺陷；sham触发0。不得续跑旧目录或据此检验恢复理论。

公开ReAct chat适配v1另有提示符解析错误，1个完成episode、1个部分episode后停止；54次dispatch/53响应/1未确认。v2另建目录，仅添加一个行首 `>` 的解析，不改模型、提示、任务或门槛。每次失败均独立计数，完整journal仅留本地。

- 串行离线vLLM候选profile完成12固定payload ×3顺序的36请求预检，输出全部一致；固定轨迹的真实hook重放12/12通过、hook触发4/12。有限确定性预检不等于跨硬件保证。

## 下一门槛与尚未实现

参见 [预先基础方案](experiments/baseline-20261019.md)、[v2适配说明](experiments/baseline-20261019-v2.md)。必须先核对同骨干下memory能力和sham触发/轨迹一致性。134任务官方unseen集合全部留出，本轮没有调用模型评测它。

错误注入、污染传播父轨迹、R0–R3恢复分支、回滚后记忆有效性、Dependency-Workflow生成器、参数估计与留出长度外推仍未实现。已有软件、replay和开发能力检查不能被写成这些模块的完成证据。
