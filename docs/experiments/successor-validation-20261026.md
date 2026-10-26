# 七维候选冻结验证：执行记录

日期：2026-10-26。按[固定设计](successor-validation-design-20261026.md)执行100任务、25场景、550条轨迹，最多四次尝试。三项离线检查、200次官方环境重置身份核对及两副本服务身份核对均已完成；当前发布执行源码与协议冻结，随后启动副本请求核对和正式采集。

六维／七维是成功概率预测函数的输入维数。真实Agent继续使用冻结Qwen3-8B及完整Reflexion；七维是新一轮预测候选，六维保留作补充参照。正式实验检验冻结参数的新任务预算预测，接口核对属于启动准备。

主要比较保持四次预算的“固定起点七维Brier减完整更新七维Brier”，按25场景等权和预定t95%区间评价。C/N预测分别为89.84%和65.82%；补充比较包括六维完整更新以及保留、纠正、删除起点。

使用GPU2／3各一个原串行推理副本，生成配置与校准阶段一致。所有请求包括四次actor／reflector副本核对都计入125000次调用、24小时的新预算，原134任务留出不使用。模型加载不产生任务结果。

24小时总时限从本批模型加载准备开始计入，新契约单列准备耗时与实际采集耗时。原来约11.2小时为按历史吞吐量估算的采集时间，后续以实际进度更新。

专用源码：[协议与重置](../../scripts/prepare_successor_protocol.py)、[外层执行器](../../scripts/run_successor_validation.py)、[冻结评分器](../../scripts/evaluate_successor_validation.py)、[自动结束与资源释放](../../scripts/finish_successor_validation.py)。原运行、旧评分器和原冻结参数保持不变。

启动工件：[协议](../../repro/reflexion-20261025/successor-validation-protocol-20261026.json)、[200次重置记录](../../repro/reflexion-20261025/successor-reset-checks-20261026.json)、[离线检查](../../repro/reflexion-20261025/successor-execution-checks-20261026.json)。采集结束后自动运行接口审计、原生环境成功与记忆链核对、冻结评分，并关闭预算、释放本项目的两个副本。
