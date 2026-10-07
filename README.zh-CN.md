<p align="center">
  <img src="docs/assets/nav-eval-banner.svg" alt="Nav-Eval — 具身导航，统一评测" width="100%">
</p>

<h1 align="center">Nav-Eval</h1>

<p align="center">
  <strong>一套框架，评测所有具身导航方法。</strong><br>
  组合兼容的方法与 benchmark，配置、执行、评分与轨迹管理共用同一条流水线。
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-2563eb?style=flat-square" alt="许可证：Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.9%2B-3776ab?style=flat-square&amp;logo=python&amp;logoColor=white" alt="核心框架：Python 3.9 及以上"></a>
  <a href="#支持的方法"><img src="https://img.shields.io/badge/Method%20adapters-11-137a62?style=flat-square" alt="11 个方法适配器"></a>
  <a href="#支持的-benchmark"><img src="https://img.shields.io/badge/Benchmark%20plugins-10-137a62?style=flat-square" alt="10 个 benchmark 插件"></a>
</p>

<p align="center">
  <a href="#快速上手">🚀 快速上手</a> &nbsp;·&nbsp;
  <a href="#支持的-benchmark">🌍 Benchmark</a> &nbsp;·&nbsp;
  <a href="#支持的方法">🤖 方法</a> &nbsp;·&nbsp;
  <a href="#目录与文档">📚 文档</a> &nbsp;·&nbsp;
  <a href="docs/integration.zh-CN.md">🧩 接入插件</a>
</p>

<p align="center"><a href="README.md">English</a> &nbsp;|&nbsp; 简体中文</p>

---

<a id="为什么需要-nav-eval"></a>

## 🧭 为什么需要 Nav-Eval

在新 benchmark 上评测导航模型，通常要手工对齐相机协议、动作空间、episode 循环和指标代码，跨论文的对比结果因此散落在互不兼容的运行目录里。Nav-Eval 把这些工作收进一套共享的插件框架。

| 🧩 组合插件 | 📦 内置模型代码 | 🔁 续跑与重评分 |
|---|---|---|
| 方法实现**观测 → 动作**，benchmark 提供 episode、传感器与指标；目标类型、传感器与动作匹配即可组合。 | 推理源码随仓库发布，无需 checkout 上游仓库、editable 安装或设置 `PYTHONPATH`。 | 每次运行都记录冻结的解码默认值、资产摘要与下载回执；保存的轨迹可**断点恢复、离线重评分**。 |

<p align="center">
  <img src="docs/assets/evaluation-flow.svg" alt="组合方法与 benchmark 插件 → 校验并冻结运行计划 → 在独立 worker 中运行模型与仿真器 → 提交轨迹并离线评分" width="100%">
</p>

推理实现位于 `extensions/methods/<method>/runtime/`。插件边界、worker 隔离与评测证据的设计见[架构指南](docs/architecture.zh-CN.md)。SCAND 直接对录制轨迹离线评分，无需仿真器。

<a id="支持的-benchmark"></a>

## 🌍 支持的 benchmark

语言、物体、图像、多目标与社交导航，横跨 **Habitat**、**Isaac Sim** 与真实世界录制轨迹。

| Benchmark | 任务 | 后端 | 指南 |
|---|---|---|---|
| R2R-CE（`r2r_ce`） | 语言引导 VLN | Habitat 0.2.4 / 0.3.0 | [Continuous VLN](docs/benchmarks.md#continuous-vln-r2r_ce-rxr_ce) |
| RxR-CE（`rxr_ce`） | 语言引导 VLN，长指令 | Habitat 0.2.4 / 0.3.0 | [Continuous VLN](docs/benchmarks.md#continuous-vln-r2r_ce-rxr_ce) |
| ObjectNav（`objectnav`） | 物体目标（MP3D v1 / HM3D v2） | Habitat 0.2.4 / 0.3.0 | [ObjectNav](docs/benchmarks.md#objectnav-objectnav) |
| InstanceImageNav（`instance_imagenav`） | 实例图像目标 | Habitat 0.2.4 / 0.3.0 | [InstanceImageNav](docs/benchmarks.md#instanceimagenav-instance_imagenav) |
| HM3D-OVON（`hm3d_ovon`） | 开放词表物体目标 | Habitat 0.2.4 / 0.3.0 | [配方](configs/benchmarks/README.md) |
| GOAT-Bench（`goat_bench`） | 多目标（图像 + 类别 + 语言） | Habitat 0.2.4 / 0.3.0 | [配方](configs/benchmarks/README.md) |
| MultiON MP3D（`multion`） | 多物体导航 | Habitat 0.2.4 / 0.3.0 | [配方](configs/benchmarks/README.md) |
| MultiON HSSD（`multion`） | 多物体导航 | Habitat 0.2.4 / 0.3.0 | [配方](configs/benchmarks/README.md) |
| HSSD SocialNav（`hssd_socialnav`） | 含人形机器人的社交导航 | Habitat 0.3.0 | [SocialNav](docs/benchmarks.md#social-navigation-hssd_socialnav) |
| VLNVerse（`vlnverse`） | 网页场景 VLN | Isaac Sim 5.0 | [VLN-VERSE](docs/benchmarks.md#vln-verse-vlnverse) |
| SCAND（`scand`） | 真实世界轨迹离线评分 | 无 | [SCAND](docs/benchmarks.md#scand-scand-offline-recorded-data-evaluation) |

所有 Habitat benchmark 同时提供 0.2.4 与 0.3.0 两套绑定；SocialNav 依赖 Habitat 3 的人形与 rearrangement API，仅支持 0.3.0。各版本对应的 JSON 配置见 [benchmark 配方](configs/benchmarks/README.md)；OVON 与 GOAT 还需用 `scripts/setup_benchmark.py` 安装任务源码快照。

<a id="支持的方法"></a>

## 🤖 支持的方法

内置 **11 个方法适配器**，下表列出各方法的观测约定与原生图像几何。

| 方法 | 目标输入 | 策略观测 | 原生图像 / HFOV | 模型环境 | 上游 |
|---|---|---|---|---|---|
| StreamVLN | 语言 | RGB | 640×480 / 79° | `nav_streamvln` | [StreamVLN](https://github.com/InternRobotics/StreamVLN) |
| GA-VLN | 语言 | RGB、深度、位姿 | 640×480 / 79° | `nav_streamvln` | [GA-VLN](https://github.com/jahhaoyang/GA-VLN) |
| NaVid | 语言、类别 | RGB | 640×480 / 90° | `nav_vlm` | [NaVid](https://github.com/jzhzhang/NaVid-VLN-CE) |
| Uni-NaVid | 语言、类别 | RGB | 640×480 / 120° | `nav_vlm` | [NaVid](https://github.com/jzhzhang/NaVid-VLN-CE) |
| NaVILA | 语言 | RGB | 512×512 / 90° | `nav_vlm` | [NaVILA](https://github.com/AnjieCheng/NaVILA) |
| AwareVLN | 语言 | RGB | 512×512 / 90° | `nav_vlm` | [AwareVLN](https://github.com/GWxuan/AwareVLN) |
| NaVIDA | 语言 | RGB | 640×480 / 90° | `nav_vlm` | [论文](https://arxiv.org/abs/2601.18188) |
| ActiveVLN | 语言 | RGB | 640×480 / 90° | `nav_vlm` | — |
| JanusVLN | 语言 | RGB | 640×480 / 79° | `nav_vlm` | [JanusVLN](https://github.com/MIV-XJTU/JanusVLN) |
| InternVLA-N1 | 语言 | RGB、深度；相机俯仰探测 | 640×480 / 79° | `nav_vlm` | [InternNav](https://github.com/InternRobotics/InternNav) |
| OneVLA | 语言 | RGB | 640×480 / 90° | `nav_vlm` | [OneVLA](https://github.com/linglingxiansen/OneVLA) |

GA-VLN 仅支持 Habitat VLN 绑定，**不支持 Isaac Sim / VLNVerse**——后者的策略接口不提供位姿。InternVLA-N1 需要 `benchmark_settings.allow_tilt: true`。其余方法只要图像几何与动作语义匹配，即可运行在 VLN 绑定上。

<details>
<summary><strong>🔬 仿真器信号、动作几何与协议约定</strong></summary>

| VLN 仿真器绑定 | 策略可见信号 | 动作与图像几何 |
|---|---|---|
| Habitat 0.2.4 / 0.3.0 | RGB `uint8`；沿光轴深度 `float32`，单位米；可选位姿 `[x,y,z,qw,qx,qy,qz]` | 前进 0.25 m、转向 15° 或 30°、STOP；可选 ±15° 相机俯仰；图像尺寸与 HFOV 可配置 |
| Isaac Sim 5.0 / VLNVerse H1 | RGB `uint8`；可选沿光轴深度 `float32`，单位米（由上游归一化深度还原）；不向策略提供位姿 | 物理 H1 控制器，名义前进 0.25 m / 转向 15°；逻辑 30° 转向执行两次控制；STOP、可选 ±15° 俯仰；固定 90° HFOV |

两套绑定共用的接口约定：

- **位姿是真值，不是里程计。** Habitat 位姿取自仿真器的 Y-up 世界坐标，四元数分量无量纲；目标坐标、参考路径、测地距离与 Isaac GPS 只进评测证据，不发给策略。
- **观测不带图像历史。** 每帧附带 episode/sequence 标识与 `control_tick`，`sim_time_s` 为 null，历史图像由各方法自行维护。
- **导航动作与相机探测分开计数。** 所有控制都受有限 watchdog 约束，smoke 步数上限保持不变。
- **相机初始俯仰可配置（仅 VLNVerse）。** 默认保留 USD 相机向下 30° 的安装姿态；`camera_pitch_deg: 0` 显式选择水平初始视角，并计入实验身份。
- **`robot_flash: true` 为实验特性。** 使用上游离散 flash 控制器并保持碰撞检查开启；相机探测只渲染、不推进物理，结果与物理 H1 运行分开报告。

</details>

权重来源与各方法的 checkpoint 目录见[资产指南 — 模型权重](data/README.md#3-model-weights)与[部署 — 方法环境](docs/deployment.md#method-environments)。方法的目标输入需与 benchmark 的任务类型匹配；已内置的方法 × benchmark 组合见 [`configs/experiments/`](configs/experiments)。

<a id="快速上手"></a>

## 🚀 快速上手

**安装 → 下载 → 登记 → 评测。** 用 StreamVLN 或 GA-VLN 跑 R2R/RxR，从下文的 `streamvln habitat024` 环境配置开始即可。

### 1. 安装环境

系统要求：Linux、Conda、C++ 编译器、CUDA 12.8 Toolkit、NVIDIA Ampere 及以上 GPU。在仓库根目录执行：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
bash scripts/setup_environment.sh streamvln habitat024   # envs/nav_streamvln
bash scripts/setup_environment.sh vlm                    # envs/nav_vlm
bash scripts/setup_environment.sh habitat030             # envs/nav_habitat030
```

| 环境 | 覆盖范围 | Python |
|---|---|---|
| `envs/nav_streamvln` | StreamVLN、GA-VLN；所有方法共用的 Habitat 0.2.4 仿真 worker | 3.9 |
| `envs/nav_vlm` | NaVid、Uni-NaVid、NaVILA、AwareVLN、NaVIDA、ActiveVLN、JanusVLN、InternVLA-N1、OneVLA | 3.10 |
| `envs/nav_habitat030` | 所有方法共用的 Habitat 0.3.0 仿真 worker（含 SocialNav） | 3.9 |
| `envs/nav_isaac` | VLNVerse 仿真 worker（可选） | 3.11 |

模型与仿真器运行在各自独立的环境中，共享 `data/`；方法 × benchmark 组合需满足传感器与动作能力。只跑 StreamVLN 或 GA-VLN 的 R2R/RxR，执行第一条命令即可。依赖配方见[部署 — 方法环境](docs/deployment.md#method-environments)。

### 2. 下载数据与权重

按[资产指南](data/README.md)操作：episode 数据集（§1）、需授权的 MP3D/HM3D/HSSD 场景（§2）、模型权重（§3）。以 StreamVLN 在 R2R-CE `val_unseen` 上运行为例：

```bash
mkdir -p checkpoints
python -B -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln", local_dir="checkpoints/streamvln")'
```

最终目录：

```text
checkpoints/streamvln/config.json
data/datasets/r2r/val_unseen/val_unseen.json.gz
data/scene_datasets/mp3d/<scan>/<scan>.glb
data/scene_datasets/mp3d/<scan>/<scan>.navmesh
```

### 3. 登记一次

```bash
METHOD=streamvln SIMULATOR=habitat024 bash scripts/register_assets.sh
python -B -m nav_eval configure --simulator habitat030 \
  --environment-python envs/nav_habitat030/bin/python --data-root data
```

登记结果写入 `configs/local.json`（已被 Git 忽略），后续运行自动读取。额外权重路径（GA-VLN 的 `vision_tower`、`vggt_path`，NaVid 的 `vision_tower`）用 `bash scripts/register_assets.sh --method-path KEY=PATH` 登记。详见[资产指南 — 登记](data/README.md#4-register-once-after-downloading)。

### 4. 运行推理与评测

```bash
# 在仿真器中运行模型推理，随后评分
METHOD=streamvln BENCHMARK=r2r_ce GPU=0 bash scripts/eval.sh

# 单方法快捷脚本，效果相同
GPU=0 bash scripts/methods/streamvln.sh

# 批量运行多组（组合在脚本内编辑）
GPU=0 bash scripts/eval_suite.sh
```

`MODE=plan` 只解析组合，不启动任何 worker。默认 R2R 运行 33 条诊断 episode；`CONFIG=configs/experiments/streamvln-r2r-full.json` 运行全量 1,839 条。尚无全量配置的方法需显式指定 `CONFIG` 或设置 `SMOKE_OK=1`。中断的运行可继续，已保存的轨迹可重新评分：

```bash
python -B -m nav_eval resume --run 'runs/streamvln-r2r/<run-id>'
python -B -m nav_eval evaluate --run 'runs/streamvln-r2r/<run-id>'
```

多 GPU 并行、共享模型批处理与 Docker：[真实运行](docs/deployment.md#real-runs) · [并行推理](docs/deployment.md#parallel-inference) · [Docker](docs/deployment.md#docker)。

### 专用 benchmark

- **OVON / GOAT / MultiON**：`python -B scripts/setup_benchmark.py hm3d_ovon`（`goat_bench`、`multion_hssd` 同理）安装任务源码快照；数据下载与两套版本的 JSON 见[配方](configs/benchmarks/README.md#environments-and-configurations)。
- **VLNVerse**：`bash scripts/setup_environment.sh isaac` 创建 `envs/nav_isaac`（Isaac Sim 5.0 + InternUtopia）。机器人本体数据需授权——先在浏览器向 [Embodiments](https://huggingface.co/datasets/InternRobotics/Embodiments) 申请访问，然后：

```bash
python -B scripts/setup_benchmark.py vlnverse
envs/nav_streamvln/bin/hf auth login
envs/nav_streamvln/bin/python -B scripts/download_vlnverse_smoke.py
METHOD=streamvln BENCHMARK=vlnverse GPU=0 \
  CONFIG=configs/experiments/streamvln-vlnverse-smoke.json bash scripts/eval.sh
```

`HF_ENDPOINT=https://hf-mirror.com` 为公开数据切换下载端点。完整指南见 [VLN-VERSE](docs/benchmarks.md#vln-verse-vlnverse)。

H1 本体数据说明：

- `hf auth login` 须使用申请过访问的同一账号的 read token。
- 本地测试复用已下载的 `data/Embodiments/vln-pe/h1/`；[资产来源与验证](docs/vlnverse-validation.md) 单独记录，与在线下载回执分开保存。
- 获得授权后，执行 `hf download InternRobotics/Embodiments --repo-type dataset --include 'vln-pe/h1/**' --local-dir data/downloads/embodiments-authorized` 下载到独立目录，现有文件保持不动。

VLNVerse 原生配置覆盖 StreamVLN、NaVIDA、AwareVLN、JanusVLN、NaVid、Uni-NaVid 与 InternVLA-N1。30° 转向在一次策略决策内执行两次 15° 控制；InternVLA 显式启用深度与相机俯仰。完整预算诊断、可复现配置与协议差异见[实测结果与边界](docs/vlnverse-validation.md)。

- **SCAND**：对录制轨迹离线评分（ADE/FDE），无需仿真器——[SCAND 指南](docs/benchmarks.md#scand-scand-offline-recorded-data-evaluation)。

<a id="目录与文档"></a>

## 📚 目录与文档

| 指南 | 从这里开始 |
|---|---|
| 🏗️ [架构](docs/architecture.zh-CN.md) | 理解插件、worker 隔离与执行流水线 |
| 🌍 [Benchmark](docs/benchmarks.zh-CN.md) | 选择任务，确认仿真器要求 |
| ⚙️ [部署与 Docker](docs/deployment.zh-CN.md) | 配置环境、并行推理与容器 |
| 📦 [资产](data/README.zh-CN.md) | 下载数据集、场景与权重，登记本地路径 |
| 🧩 [插件接入](docs/integration.zh-CN.md) | 新增方法、benchmark、仿真器或指标 |

<details>
<summary><strong>🗂️ 仓库目录</strong></summary>

```text
nav_eval/       统一评测运行框架
extensions/     方法、benchmark、仿真器、指标与控制器插件
configs/        实验与资源配置
scripts/        安装、登记与评测入口
data/           数据集与场景
checkpoints/    模型权重
envs/nav_*/     隔离的运行环境
runs/           轨迹、证据与评分
```

</details>

## 📄 许可证

Nav-Eval 原创代码采用 [Apache 2.0](LICENSE) 许可。内置推理源码保留上游条款，见各 runtime 的 `provenance.json` 与 `notices/`；数据与权重遵循各自上游许可。
