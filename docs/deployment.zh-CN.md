# Linux 部署与推理

[简体中文](deployment.zh-CN.md) | [English](deployment.md)

Nav-Eval 把「跑什么」（实验配置，可移植，入 Git）与「在哪跑」（永久本机配置，机器
专属，git 忽略）分开。每个方法保留其上游依赖环境；框架把模型与仿真器作为独立进程
启动并接线。

本页覆盖完整路径：验证控制面 → 搭环境 → 下载资产 → 一次登记本机配置 → 运行、扩展与
容器化。资产下载链接见[资产指南](../data/README.zh-CN.md)；各 benchmark 配置见
[Benchmark 指南](benchmarks.zh-CN.md)。

首次下载完成后，用 `python -m nav_eval configure --method <模型> --checkpoint <路径>`
登记权重；用 `--simulator <插件> --data-root data --environment-python <解释器>`
登记仿真器。配置永久保存于 `configs/local.json`，`plan/run` 自动读取。
环境可由 `bash scripts/setup_environment.sh <方法或仿真器>` 安装到仓库的 `envs/`。
常规评测在 Bash 中选择模型与 benchmark：

```bash
METHOD=streamvln BENCHMARK=r2r_ce GPU=0 bash scripts/eval.sh
GPU=0 bash scripts/eval_suite.sh
# 多 GPU 的逐次选择；本机配置另需足够的显存预算。
METHOD=streamvln BENCHMARK=r2r_ce GPUS=0,1 bash scripts/eval.sh
```

`scripts/eval.sh <run目录>` 保留离线评分接口。

## 总览

一次典型评测涉及：

| 组成 | 位置 | 示例 |
|---|---|---|
| 实验配置 | `configs/experiments/*.json` | `navida-r2r.json` |
| 永久本机配置 | `configs/local.json`（git 忽略） | 首次下载后用 `configure` 登记 |
| 方法环境 | 共享 Conda 前缀 | `envs/nav_streamvln` 或 `envs/nav_vlm` |
| 仿真器环境 | `envs/nav_habitat030/` 等 | 每个 Habitat 版本一个 |
| 数据集与场景 | `data/`（git 忽略） | `data/datasets/r2r/`、`data/scene_datasets/mp3d/` |
| 权重 | `checkpoints/<method>/`（git 忽略） | `checkpoints/streamvln/` |
| 运行输出 | `runs/<output>/<run-id>/` | `episodes.jsonl`、评分报告 |

前置条件：Linux、一块 NVIDIA GPU、Conda，以及所选 benchmark 的数据集/场景
（[资产指南](../data/README.zh-CN.md)）。

## 控制面验证

在仓库根目录执行，无需 GPU 与仿真器：

```bash
python -B -m nav_eval plugins list
python -B -m nav_eval doctor
python -B -m nav_eval plan \
  --config configs/experiments/navida-r2r.json \
  --gpu 0
```

`plugins list` 与 `doctor` 检查插件发现、解释器与工具链。`plan` 解析完整实验——
benchmark、仿真器、绑定、传感器、动作与资源设置——在启动前报告不兼容项。模型与
仿真器只在 `run` 的 preflight/prepare 阶段才加载。

## 首次运行指南（NaVIDA on R2R-CE）

NaVIDA 是环境最简单的方法：直接经 Transformers 加载，方法环境就是纯 `pip`。其他
方法流程相同，换成各自的环境即可（见[方法环境](#方法环境)）。NaVIDA 权重暂未公开
（[arXiv 2601.18188](https://arxiv.org/abs/2601.18188)）；公开前可换成任何已有权重
的模型走本流程，或使用完全不需要权重的
[benchmark smoke 配方](../configs/benchmarks/README.md)。

**1. Habitat 0.3.0 仿真器环境：**

```bash
# 在克隆的 Nav-Eval 仓库根目录执行。
export NAV_EVAL_ROOT="$PWD"
mkdir -p "$NAV_EVAL_ROOT/envs"
bash scripts/setup_environment.sh habitat030
```

**2. 方法环境：**

```bash
export CUDA_HOME=/usr/local/cuda-12.8
bash scripts/setup_environment.sh vlm
MODEL_PY="$NAV_EVAL_ROOT/envs/nav_vlm/bin/python"
```

**3. 资产**放在仓库根目录下（相对路径相对调用 `nav_eval` 的目录解析）：

```text
data/
├── datasets/r2r/val_unseen/val_unseen.json.gz
└── scene_datasets/mp3d/<scene>/<scene>.glb
checkpoints/navida/        # NaVIDA checkpoint（config + tokenizer + 权重分片）
```

**4. 永久本机配置**——保存为 `configs/local.json`（git 忽略；也可用
绝对路径）：

```json
{
  "method": {
    "python": "envs/nav_vlm/bin/python",
    "gpu": 0,
    "min_free_memory_mib": 12000,
    "timeout_s": 900,
    "settings": {"checkpoint": "checkpoints/navida"}
  },
  "environment": {
    "python": "envs/nav_habitat030/bin/python",
    "gpu": 0,
    "min_free_memory_mib": 4000,
    "timeout_s": 900,
    "settings": {"data_root": "data"}
  }
}
```

本机配置只保存一次；在 Bash 中通过 `GPU` / `GPUS` 逐次选择物理卡。

**5. 运行。** 自带的 `configs/experiments/navida-r2r.json` 选 1 个 episode；删掉
`episode_limit` 即全量 1,839 episode：

```bash
python -B -m nav_eval plan --config configs/experiments/navida-r2r.json \
  --gpu 0
python -B -m nav_eval run --config configs/experiments/navida-r2r.json \
  --gpu 0 --output runs/navida-r2r
python -B -m nav_eval evaluate --run runs/navida-r2r/<run-id>
```

当 `runs/navida-r2r/<run-id>/` 出现 `episodes.jsonl` 和带 SR/SPL 的评分时，环境就绪。
各角色日志在 `workers/*/method.log` / `environment.log`；失败汇总在 `failure.json`。

## 正式运行

四条命令覆盖完整生命周期：

```bash
python -m nav_eval plan    --config <experiment.json> --gpu 0
python -m nav_eval run     --config <experiment.json> --gpu 0 --output <dir>
python -m nav_eval resume  --run runs/<output>/<run-id>   # 续跑未完成 episode
python -m nav_eval evaluate --run runs/<output>/<run-id>   # 对已有证据重新评分，不推理
```

永久本机配置说明：

- 从 [example.json](../configs/resources/example.json) 复制修改。`method` /
  `environment` 设置两个默认角色；`runtimes` 按插件 ID 配置其他角色。每个角色可填
  `python`、`gpu`、`pythonpath`、`library_paths`、`cwd`、`env`、`timeout_s`、
  `min_free_memory_mib`，以及 `checkpoint` / `vision_tower` / `data_root` 等 `settings`。
- `plan` / `run` 支持 `--gpu <卡号>`：指定本次运行两个角色使用的物理卡，覆盖
  文件中的值。文件保留静态信息（解释器、权重、数据路径），显卡逐次指定；
  `scripts/` 包装脚本通过 `GPU` 环境变量透传。
- 资源 settings 只接受 manifest 的 `requires.paths` / `resource_settings` 键。
  相机、动作、成功阈值等实验参数属于实验配置，在那里进入对比键。
- 模型与渲染进程可以放在不同 GPU；物理卡经 `CUDA_VISIBLE_DEVICES` 映射为进程内
  设备 0。
- VLNVerse 登记 Isaac 解释器与 `data_root`；用
  `python -B scripts/setup_benchmark.py vlnverse` 安装仓库内任务运行时——见
  [Benchmark 指南：VLN-VERSE](benchmarks.md#vln-verse-vlnverse)。

执行行为：

- Preflight 在启动前检查路径、解释器、GPU 与磁盘。模型加载与环境初始化并发执行；
  所有 replica 通过 prepare 与身份校验后 episode 才开始。
- 模型跨 episode 常驻；`reset`/`close_episode` 按上游重置规则清理历史与缓存。
- Launcher 清理整个进程组，包括解释器脚本启动的子进程。Ctrl-C 保留已提交的
  attempt；未提交的任务从 episode 边界恢复。
- 更换权重、解码、相机参数或仿真器后请新建运行——`resume` 只延续同一实验。

## StreamVLN on R2R

`configs/experiments/streamvln-r2r.json` 运行 33-episode 子集（11 个 val_unseen
场景各取 3 条路线）；删除 `episodes` 即全量 1,839 episode。该方法环境可同时充当
habitat024 仿真器环境（配方见 [README 快速开始](../README.zh-CN.md)与下文
[方法环境](#方法环境)）。

```bash
python -B -m nav_eval run \
  --config configs/experiments/streamvln-r2r.json \
  --gpu 0
```

默认 `preprocess_mode: lazy`、`tokenizer_mode: reuse`：保留全部原始 RGB 观测，只
预处理选中的当前/历史帧，上游 prompt 构造使用专用 tokenizer 副本。模型精度、生成
设置与随机采样遵循上游。`eager` / `upstream` 用于精确复现方法自身的预处理行为。

`min_free_memory_mib` 按模型+仿真器的保守预算填写——短 smoke 会低估长轨迹的显存
需求。StreamVLN 支持独立 replica；不支持 shared 会话与批式生成。只有实验、资产、
方法身份与指标设置完全一致的两个 run 才可比较。

## 默认解码与批组织

每个方法只有一个冻结的默认解码，记录在 `runtime_identity()` 中；解码不得在基线间
变化，等价性比较不得混用解码设置。

各方法默认解码：

| 方法 | 默认解码 |
|---|---|
| `navid` / `uni_navid` | 上游 checkpoint 采样——生成完整委托给上游 agent（`extensions/methods/navid/service.py`） |
| `navida` | `decoding: upstream`（`use_model_defaults=True`；应用 checkpoint 采样覆盖）。可选 `decoding: greedy`，批大小 >1 时必需 |
| `streamvln` | 贪心（`do_sample=false`、`num_beams=1`） |
| `navila` | 贪心（`do_sample=false`、`temperature=0.0`） |
| `awarevln` | 贪心（`do_sample=false`、`temperature=0.0`） |

批组织：

- 默认 `inference.mode: replicas`——每个 replica 拥有常驻模型，每个 worker 恰持有
  一个会话，episode 内 observe/reason/act 严格顺序执行，生成恒为单例。上游全局
  历史（NaVid）与缓存（StreamVLN）无需改成线程安全。
- 显式 `inference.mode: shared`——一个模型服务多个环境会话。`max_batch_size` 默认 1
  且保持单例数值行为；>1 需要方法声明 `independent_greedy` 能力及 `batching_requires`
  设置（目前仅 NaVIDA，且要求 `decoding: greedy`）。见
  [共享模型与批处理](#共享模型与批处理)。

## 并行推理

在实验配置加 `"parallelism": 2`，在资源映射加两条分配（也可直接编辑
[parallel.example.json](../configs/resources/parallel.example.json)）：

```json
{
  "replicas": [
    {"method": {"gpu": 0}, "environment": {"gpu": 0}},
    {"method": {"gpu": 1}, "environment": {"gpu": 1}}
  ]
}
```

规则：

- `replicas` 条目与现有 `runtimes` / `method` / `environment` 配置合并——解释器、
  权重与 settings 沿用；每条只固定部署字段（GPU、显存、超时）。
- `replicas` 数量必须等于 `parallelism`；每个需要 GPU 的角色都要写明设备。一个
  replica 的方法与环境可在不同卡上；多个 replica 可共享一卡。
- Replica 不能覆盖 checkpoint 与方法/benchmark 设置。

示例：

```bash
python -B -m nav_eval run \
  --config configs/experiments/navida-r2r-parallel.json \
  --gpus 0,1
```

`navida-r2r-parallel.json` 固定单 GPU、双 replica、4 个诊断 episode。所有 replica
共享一个 episode 队列，完成当前任务即取下一个，长轨迹不会让其他 replica 空等。
其他方法用自己的基础资源映射套同样的 `parallelism`/`replicas` 配置。

GPU 角色需要正的 `min_free_memory_mib`——该 worker 的保守显存预算。Preflight 按物理
GPU UUID 汇总预算（指向同一张卡的数字索引与 UUID 会合并），检查总量不超过可用
VRAM。请为最长历史与其他任务留出余量；同卡多 replica 的加速比取决于模型大小与
GPU 负载。

运行细节：

- 各 replica 日志在 `workers/000/`、`workers/001/`、…；`replicas.lock.json` 锁定
  各 replica 的运行时与资产。
- 执行前所有 replica 对身份与 episode 列表达成一致。重启只影响遇到基础设施错误或
  清理失败的 worker 对；身份会重新校验。结果、评分与 resume 都在同一 run 目录内。
- `local` launcher 保持单 replica，以隔离 RNG 与仿真线程。

## 共享模型与批处理

适配过的方法可用一个常驻模型服务多个独立环境（已接入方法中 NaVIDA 声明了该能力）。
在实验配置中加入：

```json
{
  "parallelism": 2,
  "inference": {"mode": "shared", "max_batch_size": 1, "max_wait_ms": 0}
}
```

```bash
python -B -m nav_eval run \
  --config configs/experiments/navida-r2r-shared.json \
  --gpus 0,1
```

- 所有 `replicas[].method` 合并出的部署资源必须一致；模型只加载一次，显存预算只计
  一次，环境预算累加。
- `parallelism` 封顶环境/会话数；`max_batch_size` 封顶每批合并请求数（1 …
  parallelism，默认 1）；`max_wait_ms` 限制首个请求就绪后等待更多请求的上限
  （默认 5 ms——是排队界限，不是 RPC 延迟上限）。
- 只合并**不同** episode 的当前请求；episode 内循环仍等待执行与新观测。历史、动作
  队列、生成与观测序列按会话隔离。NaVIDA 多输入用左 padding 并保留上游 JPEG/帧
  采样/动作解析；KV 与响应绝不跨 episode 复用。
- 模型日志在 `workers/shared-method/method.log`；环境日志仍在 `workers/000/`、…。
  真实模型批数请读生成计数器——合并的队列请求可能包含待执行动作。
- 基础设施或清理失败会先排空在途 attempt、提交，再整体重启模型与环境并重新校验
  身份。策略错误只终止受影响的 episode。
- NaVIDA 默认 `decoding: upstream`，按会话保存/恢复 RNG。批大小 >1 需
  `method_settings: {"decoding": "greedy"}`——贪心在验证中改变了真实轨迹，请作为
  独立基线。量化、张量并行与 vLLM/SGLang 服务在路线图中，本版本未包含。

## 仿真与资产

Isaac/InternUtopia 使用单个 Kit App，通过 reset 切换场景。绑定默认允许
`max_control_steps=500` 次控制决策与 `max_task_steps=25000` 上游任务步；两者都是
benchmark 设置并进入对比键。`navida-vlnverse-two-scenes.json` 是两场景部署诊断
（每场景 20 次决策）。Isaac rc/build 身份记录在 lock 文件中；模型初始化、Kit 启动、
场景 reset 与推理时间分开统计。

Habitat 平台要点：

- ActiveVLN/OneVLA 的 RxR 入门配置共享从零安装的 Habitat 0.2.4 环境。
  仅安装 0.2.4 与 0.3.0；专用任务也使用这两个 SDK。
- R2R/RxR 的两个 Habitat 版本均提供实测 `pose` 传感器（xyz + 四元数，habitat 世界系），GA-VLN
  需要。
- r2r/rxr 的两个版本绑定均支持可选 `camera_tilt` 动作
  （`benchmark_settings.allow_tilt`），供 InternVLA-N1 的地面视角探测——每次 tilt
  消耗 500 步预算中的一个控制 tick。

首次运行会计算权重与所选资产的内容摘要。run 目录包含解析后的配置、各角色最小
配置、环境/模型/评分锁、episode attempt、轨迹、私有证据、计时与独立评分。

## 方法环境

按下面的命令从空的 `envs/nav_*` 前缀安装，无需已有模型环境。核心只需
`nav_streamvln`（含 Habitat 0.2.4）、`nav_vlm` 和 `nav_habitat030` 三个环境。

内置模型代码在 `extensions/methods/<method>/runtime/` 中。方法环境只需安装第三方
Python 依赖并准备权重与 HF 缓存，无需方法 checkout、editable 方法安装或源码
`pythonpath` / `cwd`。下面的依赖文件记录所用的安装版本。

共享安装使用 `bash scripts/setup_environment.sh streamvln habitat024` 安装
StreamVLN / GA-VLN 加 Habitat，`bash scripts/setup_environment.sh vlm` 安装其余
内置模型，分别放在 `envs/nav_streamvln`（Python 3.9、Transformers 4.45.1）和
`envs/nav_vlm`（Python 3.10、Transformers 4.57.0）。下表各模型依赖文件是旧的独立配方，
不要叠加到共享依赖组。Python 版本差异和已隔离命名空间的 VLM fork 本身不是拆环境
的理由。

| 方法 | 安装 | 备注 |
|---|---|---|
| NaVIDA | 全新安装——[上文命令](#首次运行指南navida-on-r2r-ce) | 纯 Transformers；无需上游仓库 |
| StreamVLN | `configs/environments/streamvln-inference.txt` | 模型代码与提示辅助函数已内置；Habitat 只供环境 worker 使用 |
| NaVid / Uni-NaVid | `configs/environments/navid-inference.txt` | `vision_tower` 指向 EVA 权重，或把 `eva_vit_g.pth` 放入 checkpoint 目录；CLIP 处理器配置已内置 |
| NaVILA / AwareVLN | `configs/environments/vila-inference.txt` | Python ≥3.10；两个 VLM fork 各有独立命名空间 |
| ActiveVLN | 任意支持 Qwen2.5-VL 的环境；权重在 `checkpoints/activevln/{rl,sft}_{r2r,rxr}` | 进程内 Transformers 服务；采样（t=0.2，top_p=0.8）按 episode 平台播种 |
| JanusVLN | `configs/environments/janusvln-inference.txt` | VGGT 已内置，KV 缓存逐 episode 清理；贪心 24 token |
| InternVLA-N1 | `configs/environments/internvla-n1-inference.txt` | 地面探测使用 `camera_tilt`（`benchmark_settings.allow_tilt`，habitat030）；NavDP flow head 平台播种 |
| OneVLA | `configs/environments/onevla-inference.txt` | `checkpoint` 为 `run/checkpoints/*.pt`；需 `run/config.yaml` 与 `run/dataset_statistics.json`；需要时配置资源 `base_vlm`；默认注意力为 `sdpa` |
| GA-VLN | `configs/environments/gavln-inference.txt` | 需 `pose`（habitat024/030）、`vision_tower`（SigLIP）与 `vggt_path`；8 步 KV 窗口重置；仅 R2R |

InternVLA-N1 的辅助深度权重放入配置的 checkpoint 目录：异步 NextDiT 使用
`depth_anything_v2_metric_hypersim_vits.pth`，异步 NavDP 使用
`depth_anything_v2_vits.pth`。模型代码与深度 resize 不再依赖 Habitat。
第三方 Python 依赖和仿真 SDK 分别安装，CUDA wheel 需匹配对应解释器。

StreamVLN 环境：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
bash scripts/setup_environment.sh streamvln habitat024
```

来源版本与许可说明在各 runtime 的
`provenance.json` 和 `notices/` 中，根目录 Apache 许可不覆盖这些上游条款。
StreamVLN 声明 CC BY-NC-SA 4.0；记录的 JanusVLN / GA-VLN 版本未发现顶层许可授予。
导出环境锁以便复现（`build/` 已被 git 忽略）：

```bash
mkdir -p "$NAV_EVAL_ROOT/build/env-locks"
for role in nav_vlm nav_habitat030; do
  conda list -p "$NAV_EVAL_ROOT/envs/$role" --explicit \
    > "$NAV_EVAL_ROOT/build/env-locks/$role-conda.txt"
  "$NAV_EVAL_ROOT/envs/$role/bin/python" -m pip freeze \
    > "$NAV_EVAL_ROOT/build/env-locks/$role-pip.txt"
done
```

## Docker

本仓库不附带预构建镜像——用下面的配方从你的可用环境构建。推荐布局是单张合并镜像，
方法与 Habitat 环境并排放置；宿主机跑控制面并从该镜像启动 worker 容器。Launcher
实现：`nav_eval/execution/launchers.py`。

规划指南：一个原生方法需要**两个运行时环境**（方法 + Habitat 0.3.0），打包为一张
合并镜像或两张。四会话 `replicas` 即 4 个方法 + 4 个仿真容器（四份模型拷贝）；
NaVIDA `shared` 为 1 + 4。共享安装使用 `streamvln` 和 `vlm` 两个模型依赖组，
按需增加 SDK 环境。StreamVLN 加 Habitat 0.2.4 可共用一个解释器，模型与仿真器
仍由各自的 worker 进程加载。

### 宿主机准备

按 [NVIDIA 安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
安装 NVIDIA 驱动、Docker Engine 与 NVIDIA Container Toolkit；调用用户需要 daemon
权限。初始运行时配置（管理员）：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
nvidia-smi && docker info
CUDA_IMAGE='your-cuda-image@sha256:REPLACE_WITH_REAL_DIGEST'
docker run --rm --gpus device=0 "$CUDA_IMAGE" nvidia-smi
```

不要在镜像内安装宿主内核驱动；`nvidia-smi` 正常不代表 headless 渲染可用——用真实
episode 验证。

### 构建合并镜像

在 git 忽略的构建目录中对可用环境做快照：

```bash
mkdir -p "$NAV_EVAL_ROOT/build/docker-streamvln" && cd "$NAV_EVAL_ROOT/build/docker-streamvln"
METHOD_ENV="$NAV_EVAL_ROOT/envs/nav_vlm"
SIM_ENV="$NAV_EVAL_ROOT/envs/nav_habitat030"
conda list -p "$METHOD_ENV" --explicit > conda-explicit.txt
"$METHOD_ENV/bin/python" -m pip freeze > pip-freeze.txt
conda pack -p "$METHOD_ENV" -o model.tar.gz
conda pack -p "$SIM_ENV" -o habitat030.tar.gz
```

两个归档旁放 Dockerfile（基础镜像按 digest 固定，含 Habitat 所需系统库，包括
EGL/OpenGL）：

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY model.tar.gz habitat030.tar.gz /tmp/
RUN mkdir -p /opt/envs/nav_vlm /opt/envs/nav_habitat030 \
    && tar -xzf /tmp/model.tar.gz -C /opt/envs/nav_vlm \
    && /opt/envs/nav_vlm/bin/python /opt/envs/nav_vlm/bin/conda-unpack \
    && tar -xzf /tmp/habitat030.tar.gz -C /opt/envs/nav_habitat030 \
    && /opt/envs/nav_habitat030/bin/python /opt/envs/nav_habitat030/bin/conda-unpack \
    && rm /tmp/model.tar.gz /tmp/habitat030.tar.gz
ENV PATH=/opt/envs/nav_vlm/bin:$PATH
ENV PYTHONUNBUFFERED=1
```

```bash
BASE_IMAGE='your-compatible-base@sha256:REPLACE_WITH_REAL_DIGEST'
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t nav-eval-streamvln-habitat030:repro .
```

Launcher 通过 **RepoDigest**（`repo@sha256:<64 hex>`）引用镜像，而非 tag 或镜像 ID。
发布到你的 registry 并记录 digest
（[Docker 发布](https://docs.docker.com/reference/cli/cli/image/push/)）。

### 资源映射与启动

实验配置设 `launcher: docker`；各角色把 `python` 换成 `image` + `container_python`，
保留 settings/预算，并加入挂载与环境变量：

```json
{
  "method": {
    "image": "registry.example/nav-eval-streamvln-habitat030@sha256:REPLACE_WITH_64_HEX",
    "container_python": "/opt/envs/nav_vlm/bin/python",
    "min_free_memory_mib": 24000,
    "timeout_s": 1800,
    "shm_size": "8g",
    "mounts": [
      {"source": "/srv/nav-eval/assets/checkpoints/streamvln", "target": "/srv/nav-eval/assets/checkpoints/streamvln", "read_only": true},
      {"source": "/srv/nav-eval/cache", "target": "/srv/nav-eval/cache", "read_only": false}
    ],
    "env": {
      "HF_HOME": "/srv/nav-eval/cache/huggingface",
      "XDG_CACHE_HOME": "/srv/nav-eval/cache/xdg",
      "HF_HUB_OFFLINE": "1",
      "TRANSFORMERS_OFFLINE": "1",
      "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
      "PYTHONPATH": "/opt/nav-eval"
    },
    "settings": {
      "checkpoint": "/srv/nav-eval/assets/checkpoints/streamvln"
    }
  },
  "environment": {
    "image": "registry.example/nav-eval-streamvln-habitat030@sha256:REPLACE_WITH_64_HEX",
    "container_python": "/opt/envs/nav_habitat030/bin/python",
    "min_free_memory_mib": 4000,
    "timeout_s": 1800,
    "shm_size": "8g",
    "mounts": [
      {"source": "/srv/nav-eval/assets/data", "target": "/srv/nav-eval/assets/data", "read_only": true},
      {"source": "/srv/nav-eval/cache", "target": "/srv/nav-eval/cache", "read_only": false}
    ],
    "env": {
      "XDG_CACHE_HOME": "/srv/nav-eval/cache/xdg",
      "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics"
    },
    "settings": {"data_root": "/srv/nav-eval/assets/data"}
  },
  "replicas": [
    {"method": {"gpu": 0}, "environment": {"gpu": 0}},
    {"method": {"gpu": 1}, "environment": {"gpu": 1}},
    {"method": {"gpu": 2}, "environment": {"gpu": 2}},
    {"method": {"gpu": 3}, "environment": {"gpu": 3}}
  ]
}
```

Launcher 行为：

- 宿主 preflight 与哈希在容器启动前读取资产路径：资产在**两侧使用相同绝对路径**
  挂载，包括外部符号链接目标。
- 代码快照只读挂载在 `/opt/nav-eval`（运行时包 + 选定外部插件）。启用离线标志前先
  准备好 vision/tokenizer 缓存。
- Worker 以宿主 UID:GID 运行；以该用户创建可写缓存目录。`container_python` 必须能
  直接启动——镜像 ENTRYPOINT 会被替换。
- 容器 cwd 固定为 `/opt/nav-eval`；资源 `cwd` 只影响宿主启动的进程。
  `pythonpath`/`library_paths` 不翻译为容器路径——用 `env.PYTHONPATH` /
  `env.LD_LIBRARY_PATH`，并在 PYTHONPATH 中保留 `/opt/nav-eval`。
- 每个 worker 选择一块物理 GPU 并使用本地设备编号。Headless Habitat 需要 `graphics`
  驱动能力（[NVIDIA driver capabilities](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)）；
  `shm_size` 不增加 GPU 显存。
- Worker HTTP 端口动态发布在宿主 loopback 上——不开放公共 8000 端口、不挂宿主网络、
  不挂 Docker socket。这是单机 launcher；Isaac、外部 VLM 服务与多机执行需要单独配方。

先在镜像内验证一个 CUDA 操作，再跑真实 episode：

```bash
METHOD_IMAGE='registry.example/nav-eval-streamvln-habitat030@sha256:REPLACE_WITH_REAL_DIGEST'
docker run --rm --gpus device=0 --entrypoint /opt/envs/nav_vlm/bin/python \
  "$METHOD_IMAGE" -c 'import torch, transformers; print(torch.ones(1, device="cuda").cpu())'
python -B -m nav_eval plan \
  --config configs/experiments/my-streamvln-docker.json \
  --gpu 0
python -B -m nav_eval run \
  --config configs/experiments/my-streamvln-docker.json \
  --gpu 0 \
  --output runs/docker-streamvln-r2r
```

### 可选：单个外层容器

在合并镜像内以 `launcher: "python"` 运行控制面：框架启动隔离的 Python worker 而不是
额外容器（不挂 Docker socket）。四个 replica 仍是四个模型 + 四个仿真进程，显存开销
相同。设 `method.python=/opt/envs/nav_vlm/bin/python`、
`environment.python=/opt/envs/nav_habitat030/bin/python`，保留 replicas/预算/资产，去掉
image/container_python/mounts 字段：

```bash
docker run --rm --init --gpus all --shm-size 8g \
  --user "$(id -u):$(id -g)" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e HF_HOME=/srv/nav-eval/cache/huggingface -e XDG_CACHE_HOME=/srv/nav-eval/cache/xdg \
  -v "$NAV_EVAL_ROOT:/work/nav-eval:ro" \
  -v "$NAV_EVAL_ROOT/runs:/work/nav-eval/runs:rw" \
  -v "/srv/nav-eval/assets:/srv/nav-eval/assets:ro" -v "/srv/nav-eval/cache:/srv/nav-eval/cache:rw" \
  -w /work/nav-eval --entrypoint /opt/envs/nav_vlm/bin/python \
  nav-eval-streamvln-habitat030@sha256:REPLACE_WITH_REAL_DIGEST \
  -B -m nav_eval run \
  --config configs/experiments/my-container-python.json \
  --gpu 0 \
  --output runs/container-streamvln-r2r
```

资源里的 GPU ID 必须与容器内可见设备一致。外层容器能看到两个角色的资产，部署隔离
弱于按角色分开挂载。

## 复现记录

可引用的 run 请在 run 目录旁保留：实验 JSON、脱敏后的主机资源映射、仓库/上游
commit 与本地补丁、checkpoint 版本、资产/vision/tokenizer/数据哈希、依赖清单、镜像
digest、解释器路径、GPU/驱动信息、运行锁、轨迹、失败与评分。合并镜像对两个角色只有
一个 digest；digest 不冻结挂载的权重或缓存——那些由资产哈希覆盖。更换模型、解码、
几何参数或仿真器时新建运行；`resume` 只延续同一实验。
