# Nav-Eval

Linux 上的 VLN（视觉语言导航）方法统一评测平台。方法、仿真器、benchmark 和指标全部以
manifest 插件接入，控制面只负责配置解析、依赖隔离、执行、恢复和离线评分——不包含任何
具体模型或仿真器实现。

🌐 简体中文 | [English](README.md)

📚 [文档](docs/README.md) | 🧭 [方法](#支持的方法) | 🌍 [仿真器与 Benchmark](#仿真器与-benchmark) | ⚡ [快速开始](#快速开始)

---

## 为什么是 Nav-Eval？

VLN 评测的痛点是耦合：每篇论文自带一套脚本、一个固定版本的仿真器、一套不可比的计算
口径，换一台机器结果就不可复现。Nav-Eval 把「评测流程」从「被评对象」中剥离出来：

- **可复现** —— 计划冻结 benchmark、simulator、binding、观测、指标和 seed；源码、
  插件、权重和资产以摘要锁定；episode 原子提交，可从断点恢复且不重复推理。
- **隔离** —— 方法与仿真器运行在各自的 Python 环境和进程中；方法只收到 manifest
  声明的公开观测，shape、dtype、单位和动作在推理边界校验，GT 证据不会泄漏给模型。
- **诚实分级** —— 「入口可导入」「episode 能跑通」「论文级复现」是不同验证级别，
  结果中显式携带 diagnostic/unverified 标记，不互相冒充。

## 快速开始

控制面只依赖标准库（Python >= 3.9），模型和仿真器使用各自的上游环境：

```bash
git clone https://github.com/NIneeeeeem/VLN-Eval.git && cd VLN-Eval

# 1. 查看已注册插件
python -B -m nav_eval plugins list

# 2. 从模板建立本机资源映射（解释器、GPU、权重、数据路径）
cp configs/resources/example.json configs/resources/my-host.json

# 3. 校验计划（不启动模型和仿真器）
python -B -m nav_eval plan \
  --config configs/experiments/navida-r2r.json \
  --resources configs/resources/my-host.json

# 4. 采集 + 离线评分
python -B -m nav_eval run \
  --config configs/experiments/navida-r2r.json \
  --resources configs/resources/my-host.json
python -B -m nav_eval evaluate --run runs/<run-id>
```

`plan` 能走通说明插件发现和资源规划正确；真实运行还需要资源文件指向有效的权重、
数据和上游环境。

## Bash 推理与评测

所有模型/数据集组合共用两个入口：

```bash
# 先填写解释器、GPU、权重/上游源码和数据路径。
cp configs/resources/example.json configs/resources/my-host.json
# 仅解析计划，不加载模型或仿真器；不代表资产/运行环境已验证。
MODE=plan bash scripts/inference.sh navida r2r configs/resources/my-host.json

# 按需选择其中一条，资源文件须匹配所选模型和仿真器：
bash scripts/inference.sh streamvln r2r configs/resources/my-host.json
bash scripts/inference.sh streamvln vlnverse configs/resources/my-host.json
bash scripts/inference.sh navida r2r configs/resources/my-host.json
bash scripts/inference.sh navida vlnverse configs/resources/my-host.json

# 以下两条仅在外部 InternVLA-N1 插件接入后可用：
CONFIG=/absolute/path/internvla-r2r.json bash scripts/inference.sh internvla-n1 r2r configs/resources/my-host.json
CONFIG=/absolute/path/internvla-vlnverse.json bash scripts/inference.sh internvla-n1 vlnverse configs/resources/my-host.json

# 使用运行输出 output_dir 指向的实际目录，必须包含 <run-id>。
bash scripts/eval.sh runs/navida-r2r/<run-id>
python -B -m nav_eval resume --run runs/navida-r2r/<run-id>
```

| 模型 | R2R-CE val_unseen | VLNVerse fine val_unseen |
|---|---|---|
| `streamvln` | `habitat024`，native track | `isaacsim500`，standardized track；真实运行未验证 |
| `navida` | `habitat030`，native track | `isaacsim500`，standardized track |
| `internvla-n1` | 需要外部模型插件 | 需要外部插件及兼容 Isaac 的观测/动作 |

`inference.sh 模型 数据集 资源文件 [输出父目录]` 默认读取
`configs/experiments/模型-数据集-full.json`。这些配置不限制 episode，保留 `diagnostic`
标记，不表示论文级复现；原有少量样本配置保持不变。仓库**尚未实现 InternVLA-N1**，
对应的两份配置是接入模板，不是已支持的运行配置。未注册插件时脚本明确退出；外部配置
须通过 `plugin_dirs` 注册 ID 为 `internvla-n1` 的模型，声明兼容的传感器/动作，必要时
提供 controller，详见[插件接入指南](docs/integration.md)。

使用规则：

- `PYTHON=/path/to/python` 指定控制面解释器，默认 `python`；模型/仿真器解释器、GPU
  和资产由资源 JSON 指定。`example.json` 是 NaVIDA/Habitat 模板，不能原样用于所有
  组合；需修改两个角色，或在自己的资源映射中按模型/仿真器 ID 配置 `runtimes`。
  Isaac 还需要 `challenge_repo`。
- `MODE=plan` 只检查配置；`MODE=run`（默认）执行推理采集，**结束后自动评分**。
  默认输出为 `runs/模型-数据集/<run-id>`，第四个参数仅改变父目录，不固定生成的 run ID。
- 用 `CONFIG=/path/to/experiment.json` 选择自定义配置，例如 `episode_limit: 1` 冒烟测试、
  指定 episodes、并行运行，或将 `benchmark_settings.benchmark_id` 改为
  `vlnverse_coarse_val_unseen`。配置的模型/benchmark 必须与命令匹配；R2R 内部 ID 为 `r2r_ce`。
- bash 路径参数和 `CONFIG` 相对调用时目录解析；**JSON 内部路径**建议使用绝对路径
  （相对路径从仓库根目录解析）。
- `eval.sh 运行目录` 只对冻结轨迹离线重评，不再次推理、不分配 GPU；默认采用运行时冻结的
  指标集。可传 `--metric-set FILE`、`--plugin-dir DIR`。结果写入
  `运行目录/evaluations/<evaluation-id>/` 下的 `summary.json`、`episodes.jsonl`，
  不改采集产物。中断恢复使用 `nav_eval resume`，不要重新调用 `inference.sh`。

脚本职责：

| 脚本 | 用途 |
|---|---|
| `inference.sh`、`eval.sh` | 日常推理采集和离线评分入口 |
| `compare_inference.py` | 对照完整运行的轨迹、资源锁和耗时，拒绝无效加速结论 |
| `check_preprocessing_rollout.py` | 生成预处理参考插件（`preprocessing_reference.py`）并对照其与真实方法的 rollout |

早期开发阶段的专项审计脚本不参与 inference/eval 主流程，不随仓库发布。

## 安装

```bash
# 控制面：标准库即可，也可以 editable 安装
pip install -e .

# 方法与仿真器：不做统一安装
```

每个真实方法/仿真器保留各自的上游依赖环境（Torch、Transformers、Habitat、Isaac 等），
在资源映射中为每个角色指定解释器、GPU 和路径；控制面不会替你安装它们。
部署细节（含 Docker）见[部署指南](docs/deployment.zh-CN.md)。
方法权重、数据集与上游环境的获取来源统一见 [data/README.md](data/README.md)。

## 使用示例

> 更多示例见 [`configs/experiments/`](configs/experiments) 与 [`configs/resources/`](configs/resources)。

### 串行评测

```bash
python -B -m nav_eval run \
  --config configs/experiments/navida-r2r.json \
  --resources configs/resources/my-host.json
```

### 并行推理（副本数据并行）

实验配置设置 `parallelism`，资源文件用 `replicas` 为每个副本显式分配 GPU：

```bash
python -B -m nav_eval run \
  --config configs/experiments/navida-r2r-parallel.json \
  --resources configs/resources/my-host.json
```

所有副本共享一个动态 episode 队列；每个副本内部仍严格顺序执行，不要求方法改成线程
安全。多卡模板见 [parallel.example.json](configs/resources/parallel.example.json)。

### 共享模型与合批

已适配方法（当前仅 NaVIDA）可以让多个环境共用一份权重，重叠环境执行与推理：

```bash
python -B -m nav_eval run \
  --config configs/experiments/navida-r2r-shared.json \
  --resources configs/resources/my-host.json
```

此示例保留上游解码，单次生成一个样本，并按 session 保存 CPU/CUDA 随机状态。
更大的张量 batch 需要显式 greedy 设置和独立轨迹一致性验证，默认不启用。

### 恢复与重评

```bash
python -B -m nav_eval resume --run runs/<run-id>      # episode 级恢复
python -B -m nav_eval evaluate --run runs/<run-id>    # 离线重评已有轨迹
```

更多参数：`python -B -m nav_eval --help`。

## 支持的方法

| 方法 ID | 实现 | 运行资源 | 当前状态 |
|---|---|---|---|
| `navida` | `extensions/methods/navida` | checkpoint，GPU | 两条 R2R 轨迹完成 JPEG 缓存等价对照；全集复现 unverified |
| `awarevln` | `extensions/methods/awarevln` | checkpoint、上游源码、GPU | 两条 R2R 轨迹完成预处理等价对照；全集复现 unverified |
| `streamvln` | `extensions/methods/streamvln` | checkpoint、上游源码、GPU | R2R/Habitat 0.2.4 已实跑；两条任务通过上游轨迹对照，全集复现仍 unverified |
| `navila` | `extensions/methods/navila` | checkpoint、上游源码、GPU | 两条 R2R 轨迹完成预处理等价对照；论文复现 unverified |
| `navid` | `extensions/methods/navid` | checkpoint、上游源码、EVA 视觉塔、GPU | 已修正 30° 转向并完成两条 R2R 轨迹；论文复现 unverified |
| `uni_navid` | `extensions/methods/navid`（变体 manifest） | 同 NaVid | 官方在线缓存、30° 转向、HFOV 120；两条 R2R 轨迹完成，论文复现 unverified |

未绑定实现、合成策略和验证夹具不列为支持的方法。`plugins check` 只证明 manifest 与
Python 入口可加载，不证明权重、仿真资产或论文复现结果可用。

## 仿真器与 Benchmark

### 仿真器

| 仿真器 ID | 实现 | 上游版本 | 提供传感器 |
|---|---|---|---|
| `habitat017` / `habitat024` / `habitat030` | `extensions/simulators/habitat` | Habitat-Lab 0.1.7 / 0.2.4 / 0.3.0 | rgb、depth |
| `isaacsim500` | `extensions/simulators/isaacsim500` | Isaac Sim 5.0 | rgb |

### Benchmark

| Benchmark | 数据 | 可用 binding | 指标集 |
|---|---|---|---|
| `r2r_ce` | R2R val_seen / val_unseen | `habitat017` / `habitat024` / `habitat030` | `r2r_ce_standard`（SR、SPL、NE、OSR） |
| `rxr_ce` | RxR val_unseen | `habitat024` / `habitat030` | `r2r_ce_standard` |
| `vlnverse` | VLNVerse fine/coarse val_unseen | `isaacsim500` | `vlnverse_standard` |

Benchmark 通过 `bindings[simulator_id]` 声明与某个仿真器的真实 task binding；仅注册
仿真器不会自动使 benchmark 可用。

## 项目组织

```text
Nav-Eval/
├── nav_eval/            # 控制面（纯标准库）
│   ├── plugins/         #   manifest 发现、入口加载、bundle 身份摘要
│   ├── planning/        #   传感器/动作/binding/证据/资源配置解析
│   ├── execution/       #   launcher、preflight、副本池、动态调度、run/resume
│   ├── rollout/         #   唯一 episode 动作循环
│   ├── evaluation/      #   独立离线评分与聚合
│   ├── storage/         #   attempt 原子提交、文件锁、派生 JSONL
│   └── sdk/             #   跨方法共享契约与 worker 边界（不含具体方法）
├── extensions/          # 插件 bundle，按类型分目录
│   ├── methods/         #   navida、navid、navila、awarevln、streamvln
│   ├── simulators/      #   habitat（017/024/030）、isaacsim500
│   ├── benchmarks/      #   r2r_ce、rxr_ce、vlnverse（binding 在各自 bindings/）
│   └── metrics/         #   r2r_ce_standard、vlnverse_standard
├── configs/
│   ├── experiments/     # 实验配置：benchmark + simulator + method + track + seed
│   └── resources/       # 机器资源映射：解释器、GPU、权重、数据路径
├── scripts/             # inference.sh、eval.sh 及对照工具
├── docs/                # 项目文档
└── runs/                # 运行产物：attempts、轨迹、证据、评分
```

实验配置示例：

```json
{
  "schema_version": "nav-eval-experiment/1",
  "benchmark": "r2r_ce", "simulator": "habitat030", "method": "navida",
  "track": "native", "launcher": "python", "transport": "binary",
  "benchmark_settings": {"benchmark_id": "r2r_val_unseen"},
  "episode_limit": 1, "seed": 0, "claim": "diagnostic"
}
```

## 添加插件

所有插件从 `extensions/**/manifest.json` 或实验配置的 `plugin_dirs` 自动发现，
没有第二套中央注册表：

| 类型 | 位置 | 指南 |
|---|---|---|
| 方法（model） | `extensions/methods/<id>/` | [添加方法](docs/integration.zh-CN.md#添加方法) |
| 仿真器 | `extensions/simulators/<id>/` | [添加仿真器](docs/integration.zh-CN.md#添加仿真器) |
| Benchmark | `extensions/benchmarks/<id>/` | [添加 Benchmark](docs/integration.zh-CN.md#添加-benchmark) |
| Benchmark-simulator binding | benchmark bundle 的 `bindings/` | 同上 |
| 指标 | `extensions/metrics/<id>/` | [添加指标](docs/integration.zh-CN.md#添加指标) |
| 控制器 | `extensions/controllers/<id>/` 或外部 `plugin_dirs` | [添加控制器](docs/integration.zh-CN.md#添加控制器) |

方法 bundle 最小结构：

```text
extensions/methods/<id>/
├── manifest.json   # ID、入口、传感器、动作、资源要求
├── adapter.py      # create(config) factory
└── service.py      # 模型生命周期与方法实现
```

`nav_eval.plugins.Registry` 自动发现 manifest；不需要修改 CLI、runner 或任何中央列表。
完整指南（含 manifest 字段、service 契约和验收清单）见
[插件接入](docs/integration.zh-CN.md)。

## 常见问题

**Nav-Eval 支持哪些方法？**
内置 6 个方法 ID（见上表），覆盖 NaVid/NaVILA/NaVIDA/AwareVLN/StreamVLN 等上游实现。
任何满足 service 契约的方法都可以通过 bundle 或外部 `plugin_dirs` 接入。

**支持哪些仿真器和 benchmark？**
Habitat-Lab（0.1.7/0.2.4/0.3.0）与 Isaac Sim 5.0；R2R-CE、RxR-CE、VLNVerse。
新增仿真器/benchmark 见[插件接入指南](docs/integration.zh-CN.md)。

**`plugins check` 通过说明什么？**
只说明 manifest 和 Python 入口可加载，不证明权重、仿真资产或论文复现结果可用。
接入验收清单见[插件接入指南](docs/integration.zh-CN.md)。

**换机器怎么跑？**
从 `configs/resources/example.json` 新建本机资源映射，填入解释器、GPU、权重和数据
路径；缺少资源、binding、传感器或动作兼容性会在 plan/preflight 阶段明确失败，不会
自动改用合成实现。

**如何并行/多 GPU？**
实验设置 `parallelism`，资源文件用 `replicas` 为每个副本显式分配 GPU；模型和渲染
可以使用不同卡。资源预算、恢复和吞吐口径见[部署指南](docs/deployment.zh-CN.md#并行推理)。

**评测中断了怎么办？**
`resume` 从 episode 边界恢复，核对计划、源码、插件、权重和资产一致后只处理未完成
项；已完成 attempt 不会重复推理。`evaluate` 可随时离线重评。

## 文档

| 文档 | 内容 |
|---|---|
| [插件接入](docs/integration.zh-CN.md) | 添加方法/仿真器/benchmark/指标/控制器的完整指南 |
| [架构](docs/architecture.zh-CN.md) | 执行链路、插件边界、协议与真实性 |
| [部署](docs/deployment.zh-CN.md) | 资源映射、并行推理、共享模型、Docker、恢复与吞吐 |
| [协议](docs/protocol.zh-CN.md) | worker 会话、公开观测、动作与失败语义、传输 |

Documentation is also available in [English](README.md#documentation).

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 发布。

## 致谢

插件化评测框架的设计参考了 [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)。
