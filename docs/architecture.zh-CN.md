# Nav-Eval 插件架构

简体中文 | [English](architecture.md)

## 单一执行链路

```text
experiment + permanent local installation
  -> manifest discovery
  -> resolve / preflight
  -> resident environment slots + isolated/shared model workers
  -> dynamic episode queue -> shared episode kernel
  -> committed attempts
  -> offline evaluation
```

`nav_eval.plugins.Registry` 扫描 `extensions/**/manifest.json` 和实验显式提供的
`plugin_dirs`；方法 bundle 可为共用实现的紧密变体增加 `<variant>.manifest.json`。
发现阶段不导入模型或仿真依赖；worker 启动后才加载 factory。

| 位置 | 责任 |
|---|---|
| `nav_eval/sdk` | 公共类型、方法生命周期、worker facade、有界合批、二进制 framing |
| `nav_eval/plugins` | manifest 发现、入口加载、bundle 身份摘要 |
| `nav_eval/planning` | 传感器、动作、binding、证据和资源配置解析 |
| `nav_eval/execution` | launcher、总资源 preflight、常驻副本池、动态调度、run/resume |
| `nav_eval/storage` | attempt 原子提交、文件锁、摘要和派生 JSONL |
| `nav_eval/rollout/generic.py` | 唯一 episode 动作循环与错误归因 |
| `nav_eval/evaluation` | 独立离线评分和聚合 |
| `extensions/<kind>/<id>` | 具体插件的 manifest、factory 和实现 |

## 插件边界

插件类型为 simulator、benchmark、method、metric、controller。Benchmark 通过
`bindings[simulator_id]` 声明真实 task binding；仅注册 simulator 不会自动使所有
benchmark 可用。

Simulator backend 管理 SDK 生命周期、原生动作和几何。Benchmark binding 管理数据、
任务配置、公开观测、动作翻译、终止和证据。同一 environment worker 内可使用仿真器
私有对象，但它们不得进入 method RPC。

Method bundle 包含完整算法状态，包括 prompt、预处理、历史、地图、KV cache 和动作
队列。统一调用链为：

```text
WorkerService -> MethodBoundaryAdapter -> ServiceRuntime -> method service
```

`MethodBoundaryAdapter` 位于 `nav_eval/sdk/method.py`，只提供公共契约，不是方法注册表。
每次 reset 生成独立 generation；过期回复、错 episode、缺失 transition、私有传感器
和未声明动作都会被拒绝。

共享模式由 `sdk/batching.py` 为每个 session 复用这条边界链路，把不同 session 的
act 送入有界队列，再调用具体方法的 `act_batch`。方法必须显式声明能力；reset、
transition、close 与模型计算互斥。SDK 不持有方法历史或实现 prompt，没有方法名称分支。

## 协议与真实性

实验必须声明 `track: native | standardized`。native 使用方法观测默认值；standardized
使用 benchmark 默认值。显式 observation 覆盖会进入比较键。

计划会冻结 benchmark、simulator、binding、任务设置、观测、控制器、指标、seed 和
episode 选择。绝对路径保存在永久本机配置 `configs/local.json`；执行结果记录源码、插件、模型和资产摘要。

离散任务使用 `control_tick`，不会把动作序号伪装为物理秒。测地距离不可用时保存
null 与原因，依赖该证据的指标返回 unavailable。GT 和几何证据只进入评估产物，
不会发给方法。

每个插件 manifest 的 `validation.level` 记录其已验证的层级：入口可导入、配置可解析、
真实 episode 可运行、全量 split 认证。

## 部署、存储与恢复

方法和环境使用独立解释器或固定 digest 的容器。资源映射分别指定 Python、GPU、环境
变量和只读路径；prepare 会实际加载模型或场景。

`execution/pool.py` 管理副本生命周期；`runner.py` 只向每个空闲副本投递一个 episode，
并在控制线程统一提交结果。`parallelism` 默认 1，增加副本不改变方法内部推理顺序。
副本先并发初始化并核对相同的运行时与资产，再执行队列。不同 episode 的任务耗时可
重叠。默认每个 episode 占用一对隔离进程；共享模式每个环境仍隔离，多个 session
共用一份模型，方法内的可变状态按 session 保存，调度器按共享依赖协调故障重启。

每个 episode attempt 先写临时文件、fsync，再原子替换。策略错误不重试；基础设施
错误只按冻结配置重试。resume 会核对计划、源码、插件、权重、资产和 episode 列表，
并只处理未完成项。

JSON/base64 是参考传输；binary 使用版本化 header 和无损 tensor buffer。两者走同一
worker 契约，不改变模型预处理语义。

接入细节见 [integration.zh-CN.md](integration.zh-CN.md)，运行资源与恢复见
[deployment.zh-CN.md](deployment.zh-CN.md)。
