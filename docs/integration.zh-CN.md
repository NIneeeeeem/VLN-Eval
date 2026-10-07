# 插件接入指南

简体中文 | [English](integration.md)

所有插件从内置 `extensions/**/manifest.json` 或实验配置的 `plugin_dirs` 自动发现；
方法 bundle 还可声明 `extensions/**/*.manifest.json` 形式的紧密变体。没有第二套中央
注册表，接入新插件不需要修改 CLI、runner 或任何中央列表。

```bash
python -B -m nav_eval plugins list
python -B -m nav_eval plugins inspect navida --kind method
python -B -m nav_eval plugins check navida --kind method
```

manifest 必须包含 `schema_version: nav-eval-plugin/1`、`kind`、`id` 和字符串 `version`。
同类 ID 重复会报错，外部插件不能静默覆盖内置插件。

各类型插件的职责边界：

| 类型 | 位置 | 职责 |
|---|---|---|
| method | `extensions/methods/<id>/` | 完整算法状态：prompt、预处理、历史、地图、动作队列 |
| simulator | `extensions/simulators/<id>/` | SDK 生命周期、原生动作和几何 |
| benchmark | `extensions/benchmarks/<task>/<id>/` | 数据、任务配置、公开观测、动作翻译、终止和证据 |
| metric | `extensions/metrics/<id>/` | 只读取已提交证据的离线评分 |
| controller | `extensions/controllers/<id>/` 或外部 `plugin_dirs` | 无状态动作转换 |

Benchmark 通过 `bindings[simulator_id]` 声明与某个仿真器的真实 task binding；仅注册
simulator 不会自动使所有 benchmark 可用。同一 environment worker 内可使用仿真器私有
对象，但它们不得进入 method RPC。

## 添加方法

创建 `extensions/methods/<id>/`，至少包含：

```text
extensions/methods/<id>/
├── manifest.json   # ID、入口、传感器、动作、资源要求
├── adapter.py      # create(config) factory
└── service.py      # 模型生命周期与方法实现
```

manifest 示例：

```json
{
  "schema_version": "nav-eval-plugin/1",
  "kind": "method",
  "id": "my_method",
  "version": "1",
  "entrypoint": "extensions.methods.my_method.adapter:create",
  "settings": [],
  "defaults": {},
  "capabilities": {
    "requires_sensors": ["rgb"],
    "emits_actions": ["primitive", "stop"]
  },
  "requires": {
    "gpu": true,
    "paths": ["checkpoint"]
  },
  "validation": {"level": "unverified"}
}
```

`adapter.py` 暴露 `create(config)`，返回实现 `call(operation, payload)` 的 service。
模型依赖必须延迟到 factory 或 `prepare()` 导入，保证 manifest discovery 不加载 Torch、
Transformers 或仿真器。

方法推理实现放入 bundle 的 `runtime/` 包，使用完整包名或相对导入。
内置方法不接受 `repo_path`，不修改 `sys.path`；用户只需提供模型资产和依赖环境。

同源方法可以在 `extensions/methods/shared/<family>/` 共享纯工具函数，模型类、
注册逻辑、对话模板和 episode 状态仍保留在各自 bundle 中。原 runtime 导入入口
可以直接重导出共享函数。用 `"code_dependencies": ["../shared/image_io", "../shared/vila"]`
声明目录；每项必须是同级 `shared/` 下唯一且存在的直接子目录，拒绝绝对路径、
路径逃逸和符号链接。发现阶段只读元数据，不导入工具模块。声明目录及可选的
`shared/__init__.py` 参与使用方的 bundle 摘要，包含来源记录和许可证说明。
外部 Docker bundle 会一并复制声明的共享目录。未声明的同级工具目录不影响
该插件摘要；运行恢复仍校验项目的全局源码身份。

`code_dependencies` 声明指纹和文件传输，Python 模块解析沿用插件原有的导入约定。
内置工具使用 `extensions.methods.shared.*` 完整包名；外部文件入口使用已安装的
完整包名，或基于 `__file__` 显式加载文件，例如
`runpy.run_path(str(Path(__file__).parent.parent / "shared/family/helper.py"))`。
Docker 复制外部 bundle 时保留这一相对布局。只声明目录不会使未限定包名的
`shared.*` 自动可导入。

可选的文件系统设置（如 `vision_tower`、`base_vlm`）只放入 `resource_paths`，不要在
`resource_settings` 中重复声明；必需路径放入 `requires.paths`。`resource_settings`
保留给旧的非路径部署设置。宿主机路径在 worker 切换目录
前固定为绝对路径，Docker 路径保留容器语义。声明的方法路径也参与资源指纹计算，
替换辅助模型会使旧的 resume 资源锁失效。
上游 loader 隐式读取的其他文件用 `resource_companions` 声明，例如
`"resource_companions": {"checkpoint": ["../config.yaml", "../dataset_statistics.json"]}`。
路径相对文件资源的父目录（目录资源则相对该目录本身）；这些文件必须存在，
且不受扩展名过滤地参与指纹。目录指纹也包含 YAML。未声明的外部依赖不会自动
被锁定，接入新方法时需要完整保留并声明。

仿真器 manifest 通过 `requires.modules` 声明依赖探针，benchmark binding 可以用
同名字段补充任务注册模块；smoke 不再按插件 ID 硬编码依赖。有主线程事件循环的
仿真器声明 `requires.main_thread: true`，通过独立 worker 运行；内联 smoke 会明确
拒绝这类后端。GPU 后端应声明 `requires.gpu`。

可选动作使用 binding 的
`"action_requirements": {"camera_tilt": {"allow_tilt": true}}` 声明。
规划阶段按 benchmark settings 固定实际启用的动作集合，用于兼容性检查和运行时
校验。需要抬头/低头的方法必须显式启用 `benchmark_settings.allow_tilt`。

注册外部运行时时带上发现目录，并在实验的 `plugin_dirs` 中保留该目录；`configure`
只记录本机解释器和资产，而 `plugin_dirs` 使 bundle 在 plan 和 worker 阶段可移植：

```bash
python -B -m nav_eval configure --plugin-dir /opt/vendor/nav-plugin --method vendor-method \
  --method-path auxiliary_model=/models/vendor-aux
python -B -m nav_eval configure --plugin-dir /opt/vendor/nav-plugin --simulator vendor-sim
```

外部插件也可通过 `benchmarks smoke --plugin-dir <directory>` 和
`matrix --plugin-dir <directory> --track native` 检查。冻结计划只包含该实验选中的
method 和 simulator，不包含无关的本机运行时库存。

方法 service 操作：

- `describe`：声明 role、传感器、动作和 transition 能力。
- `reset`：接收公开 context，初始化 episode 状态。
- `act`：接收公开 observation，返回 episode、sequence 和非空动作列表。
- `observe_transition`：仅在声明支持时接收每个实际执行动作的反馈。
- `close_episode`：清理 episode 状态；已加载权重可以保留。

通用 worker 负责 readiness、独占会话、generation、公开输入和动作校验。具体方法不要
复制 RPC server，也不要在 `nav_eval/sdk` 或 CLI 中增加方法名称分支。

默认 `parallelism` 通过复制隔离进程工作，不要求方法实现 batch API。每份模型跨
episode 常驻，`reset/close_episode` 必须完整清理历史、缓存和动作队列。按此契约
接入的新方法自动适配副本并行，无需修改 runner。

可选共享模型模式要求 manifest 的 `capabilities.batching` 声明
`"independent_greedy"`，并提供 `act_batch(payloads) -> list[reply]`。它接收
不同 session 的公开 observation，必须按输入顺序返回同样数量的标准 `act` 回复，
每个回复保留自己的 episode_id 和 observation_sequence。单样本 `act` 仍须可用。

只有模型参数可以共享；历史、动作队列、地图等可变状态必须按 session 隔离，不能让
某个 reset 清空另一条轨迹。批量调用由 SDK 串行执行，与 reset/transition/close
互斥；SDK 复用原有 WorkerService 边界进行逐会话校验，不允许同一 session 并发请求。
大于 1 的 batch 要求 service 实时声明 `independent_greedy`，不支持依赖全局随机数
序列的采样策略。batch=1 时 service 可声明 `independent_sessions`，但必须将随机
状态也按 session 隔离。两种能力均不自动保证浮点逐 token 等价。提交适配时需提供
不同历史长度、pending action、session 重用、单条错误和
真实闭环串行/批量对照。参考实现见 `extensions/methods/navida/service.py`。

若只有特定设置可合批，使用 manifest 的 `capabilities.batching_requires` 声明
要求，例如 `{"decoding": "greedy"}`。plan 对大于 1 的 batch 检查合并 defaults 后的方法
设置，worker prepare 也检查实时能力。共享模式的 max_batch_size 默认 1；
能力声明不代表较大 batch 已取得数值等价证明。

只有共享同一实现和依赖的紧密变体才应共用 bundle。此时保留主 `manifest.json`，并为
变体增加 `<variant>.manifest.json`；factory 从 `config["plugin"]["id"]` 选择变体。
不要为仅仅相似的方法建立新的中央分派表。

## 添加仿真器

创建 `extensions/simulators/<id>/`，包含 manifest 和 backend 实现：

```text
extensions/simulators/<id>/
├── manifest.json
└── backend.py
```

manifest 示例（见 `extensions/simulators/habitat030/manifest.json`）：

```json
{
  "schema_version": "nav-eval-plugin/1",
  "kind": "simulator",
  "id": "my_sim",
  "version": "1",
  "entrypoint": "extensions.simulators.my_sim.backend:MyBackend",
  "defaults": {"version": "1.2.3"},
  "capabilities": {
    "offers_sensors": ["rgb", "depth"]
  },
  "requires": {
    "gpu": true,
    "paths": ["data_root"]
  },
  "validation": {"level": "unverified"}
}
```

要点：

- `entrypoint` 指向 Backend 类（不是 factory 函数），需要实现
  `initialize(task_config) / reset(episode, seed) / step(native_action) / close()`。
- `offers_sensors` 声明该仿真器能提供的传感器；benchmark binding 的传感器需求
  会与此对齐做兼容检查。
- `requires.paths` 中声明的键（如 `data_root`）由资源映射提供具体路径。
- 仿真 SDK 依赖同样必须延迟导入；manifest discovery 阶段不加载 Habitat/Isaac。
- 上游有主线程需求的仿真器（如 Isaac/InternUtopia）通过 service 的 `run_main()`
  处理，不需要 CLI 特例。

参考实现：`extensions/simulators/habitat/backend.py`（同一 backend 服务 017/024/030
三个 manifest，用 defaults.version 区分）、`extensions/simulators/isaacsim500/backend.py`
（单个 Kit App，通过 reset 更换场景）。

新仿真器只有在对应 benchmark 增加了 binding 后才能参与评测；见下一节。

## 添加 Benchmark

创建 `extensions/benchmarks/<task>/<id>/`，包含 manifest、binding 实现和数据注册：

```text
extensions/benchmarks/<task>/<id>/
├── manifest.json
├── dataset.py           # 数据集注册（如需要）
└── bindings/
    └── <simulator_id>.py
```

按[benchmark 扩展指南](benchmark-extensions.md)选择 task 类型。
benchmark ID 不包含目录层级；公共实现放入 `shared/`。

manifest 示例（节选自 `extensions/benchmarks/vln/r2r_ce/manifest.json`）：

```json
{
  "schema_version": "nav-eval-plugin/1",
  "kind": "benchmark",
  "id": "my_benchmark",
  "version": "1",
  "default_simulator": "habitat030",
  "splits": ["my_val_unseen"],
  "defaults": {"benchmark_id": "my_val_unseen", "success_distance_m": 3},
  "settings": ["benchmark_id", "success_distance_m", "simulator_version"],
  "observation": {"width": 640, "height": 480, "hfov": 90},
  "bindings": {
    "habitat030": {
      "entrypoint": "extensions.benchmarks.vln.my_benchmark.bindings.habitat:create",
      "accepts_actions": ["primitive", "stop"],
      "capture_schema": "habitat-r2r-evidence/1",
      "provides_evidence": [
        "trajectory.goal_distances_m",
        "trajectory.positions_xyz_m",
        "reference.geodesic_start_to_goal_m",
        "reference.success_radius_m"
      ],
      "clock": "discrete_control_ticks",
      "action_semantics": {"forward_m": 0.25, "turn_rad": 0.2617993877991494},
      "settings": {"simulator_version": "0.3.0"}
    }
  },
  "metrics": ["r2r_ce_standard"],
  "validation": {"level": "unverified"}
}
```

binding 字段说明：

- `entrypoint`：指向 `create(config)` factory，返回 benchmark service。
- `accepts_actions`：该 binding 实际能执行的动作类型；方法动作与之不兼容时必须
  显式选择控制器，runner 不提供隐式动作转换。
- `capture_schema` / `provides_evidence`：声明采集的证据 schema 和字段；指标按
  `requires_evidence` 匹配，缺失字段返回 unavailable 而不是退化。
- `clock`：`discrete_control_ticks` 或物理秒；离散任务不得把动作次数冒充物理秒。
- `action_semantics`：动作幅度语义（米、弧度），进入比较键。

benchmark service 提供 `describe / episodes / reset / step / finish / close_episode`。
`finish` 返回执行 record 与 evidence，不在采集阶段计算指标。`asset_files()` 必须列出
实际使用的数据、场景和 NavMesh；`runtime_identity()` 可记录 SDK build。

参考实现：Habitat binding 位于 `extensions/benchmarks/vln/r2r_ce/bindings`，RxR 的数据集
注册位于 `extensions/benchmarks/vln/rxr_ce/dataset.py`，Isaac binding 位于
`extensions/benchmarks/vln/vlnverse/bindings`。

## 添加指标

metric manifest 声明 `requires_evidence` 与版本化 `metric_set`：

```json
{
  "schema_version": "nav-eval-plugin/1",
  "kind": "metric",
  "id": "my_standard",
  "version": "1",
  "requires_evidence": [
    "trajectory.goal_distances_m",
    "reference.geodesic_start_to_goal_m"
  ],
  "metric_set": {
    "schema_version": "nav-eval-metric-set/0.1",
    "id": "my_standard",
    "evidence_schema": "habitat-r2r-evidence/1",
    "metrics": [{"entrypoint": "nav_eval.evaluation.metrics:MyMetric"}]
  }
}
```

指标只读取已提交证据；字段缺失时返回 unavailable。新增指标可基于已存证据重评已有
轨迹。测地距离不可用时保存 null 与原因，依赖该证据的指标返回 unavailable，而不是
退化成欧氏分数。

参考实现：`extensions/metrics/r2r_ce_standard`、`extensions/metrics/vlnverse_standard`。

## 添加控制器

Controller manifest 显式声明输入/输出动作。方法动作与 binding 不兼容时必须选择
控制器；runner 不提供隐式动作转换。

控制器应是无状态的纯动作转换函数；并行副本可在控制进程中并发调用它，不能通过
模块全局变量存放 episode 历史。

## 外部 bundle

实验配置可提供绝对路径：

```json
{"plugin_dirs": ["third_party/plugins"]}
```

相对 `.py` 入口不得逃逸 bundle。身份摘要覆盖 bundle 内 `.py` 和 `.json`；模型、数据和
其他大资源必须分别通过资源映射与 `asset_files()` 锁定。

## 接入验收清单

`plugins check` 只覆盖第一项；完整验收至少包括：

1. manifest 与入口导入（`plugins check`）。
2. 公开契约：describe 声明与 manifest 一致，传感器/动作/证据字段齐全。
3. 固定输入下的上游行为对照：接入后行为与上游原始实现一致。
4. 真实 episode 闭环：隔离 worker 完成 reset/act/step/finish 全流程。
5. 离线 scorer 对照：证据字段能被指标集正确消费。

论文级复现还需完整 split 的认证运行——把该层级记录在 manifest 的
`validation.level` 中。
