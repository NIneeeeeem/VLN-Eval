# Benchmark 目录与配置

[English](benchmarks.md) | 简体中文

本页列出框架内置的全部 benchmark 插件及其运行方式：episode 与场景数据从哪里来、
放到 `data/` 下的哪个位置、宿主机资源映射需要什么、用哪个实验配置运行。免模型的
环境检查见 [smoke 配方](../configs/benchmarks/README.md)；数据集与场景下载来源见
[资产指南](../data/README.zh-CN.md)。

查询完整目录：`python -m nav_eval benchmarks list`；查询接入条件：
`python -m nav_eval benchmarks inspect GOAT-Bench`。专用任务的运行时、目标传感器与
限制见[扩展任务配置](benchmark-extensions.md)。

所有内置 benchmark 均为连续具身评测（指标在 Habitat 或 Isaac 中随运动产生）；离散
视点图导航不在范围内。每个插件遵循同一契约：冻结的 episode 选择、逐步证据采集、
只读取已提交证据的离线指标、进入对比键的 `benchmark_settings`。接入只需添加数据
与配置——CLI 与 runner 无需改动。

| Benchmark | 任务族 | 仿真器绑定 | Episode 来源 | 场景 |
|---|---|---|---|---|
| `r2r_ce` | 连续 VLN | habitat024/030 | VLN-CE episodes | MP3D |
| `rxr_ce` | 多语言连续 VLN | habitat024/030 | RxR-CE episodes | MP3D |
| `vlnverse` | VLN（Isaac） | isaacsim500 | challenge 仓库 | challenge 资产 |
| `objectnav` | ObjectNav（MP3D/HM3D） | habitat024/030 | habitat CDN | MP3D / HM3D |
| `instance_imagenav` | 实例图像目标导航 | habitat030 | Habitat InstanceImageNav v2 | HM3D-Sem |
| `hm3d_ovon` (`ovon`) | 开放词汇 ObjectNav | habitat024 / habitat030 | OVON 任务适配 | HM3D-Sem |
| `goat_bench` (`goat`) | 多模态终身导航 | habitat024 / habitat030 | GOAT 任务适配 | HM3D-Sem |
| `multion` | 顺序多物体导航 | habitat024 / habitat030 | MP3D / HSSD 可移植任务 | MP3D / HSSD |
| `scand` | 离线真机 SocialNav | 无 | SCAND 录制 | 真机数据 |
| `hssd_socialnav` | 社交导航（habitat 3.0） | habitat030 | HuggingFace `ai-habitat` | HSSD (hab3) |

### 仿真器绑定

| 绑定 | 运行时要求 | Benchmark |
|---|---|---|
| `habitat030` | 从零安装 `nav_habitat030`，标准 Habitat 0.3.0 | 全部 Habitat 任务；SocialNav 仅支持此版本 |
| `habitat024` | 从零安装 `nav_streamvln`，标准 Habitat 0.2.4 | R2R、RxR、ObjectNav、InstanceImageNav、OVON、GOAT、MultiON |
| `isaacsim500` | Isaac Sim 5.0.0 pip 环境 `envs/nav_isaac` + 任务源码快照（见 [VLN-VERSE](#vln-versevlnverse)） | `vlnverse` |

绑定的 `simulator_version` 必须与其运行时 `python` 可导入的包版本一致。
`plugins check` / `plan` 校验 manifest 与设置；正式运行前请按部署指南做各环境的
导入检查。

## 连续 VLN（`r2r_ce`、`rxr_ce`）

VLN-CE 形式的指令跟随导航：连续 MP3D 场景中的离散平台原语（前进 0.25 m、15°/30°
转向、RGB-D 640×480）。同一绑定服务全部 split；`benchmark_settings.benchmark_id`
选择数据：

| `benchmark_id` | `<data-root>/datasets/` 下的 episode 文件 | 冻结选择 |
|---|---|---|
| `r2r_val_unseen` | `r2r/val_unseen/val_unseen.json.gz` | 1,839 episode / 11 场景 |
| `r2r_val_seen` | `r2r/val_seen/val_seen.json.gz` | 778 episode / 53 场景 |
| `rxr_val_unseen` | `rxr/val_unseen/val_unseen_guide.json.gz` | 11,006 episode / 11 场景（en/hi/te） |

RxR-CE 用 `"benchmark": "rxr_ce"`，R2R-CE 用 `"benchmark": "r2r_ce"`；两者都有
habitat024/030 绑定。`*_gt.json.gz` 伴随文件可选——评分读取绑定现场采集的
证据。Episode 与场景的来源、规模、许可见[资产指南](../data/README.zh-CN.md)；
同一份 91 场景 MP3D 也覆盖 ObjectNav MP3D。

1. 不加载仿真器验证冻结选择（同时检查本机是否缺少被引用的场景）：

   ```bash
   python -B -c "
   from pathlib import Path
   from extensions.benchmarks.vln.r2r_ce.dataset import select_episodes
   root = Path('data').resolve()
   print(len(select_episodes(root / 'datasets/r2r/val_unseen/val_unseen.json.gz', root)))
   "
   ```

   `missing scenes` 错误会列出缺失的 MP3D 场景；把它们复制到
   `data/scene_datasets/mp3d/` 下，或在实验里显式选择 `episodes` 子集。

2. 登记从零安装的 `nav_streamvln` 为 `habitat024`，`nav_habitat030` 为
   `habitat030`。两者均覆盖三个 split；使用仓库安装脚本即可。

3. 运行现有实验（如 `configs/experiments/navid-r2r.json`、`streamvln-r2r.json`），
   或复制一份改 `benchmark_settings.benchmark_id` 换 split。RxR 还可用
   `benchmark_settings` 内的 `"languages": ["en-IN", "en-US"]` 选择英语。

## ObjectNav（`objectnav`）

遵循 habitat 官方 ObjectNav 协议的物体目标导航
（[ObjectNav Revisited](https://arxiv.org/abs/2006.13171)）。目标是通过
`EpisodeContext.goal`（`kind: object_category`）传递的**物体类别**；纯语言方法自行
把类别渲染进 prompt（NaVid 即如此，见下）。证据记录 habitat 自身的
`DistanceToGoal` 度量（视点测地距离——与上游 success/SPL 使用的量相同）；指标集
`objectnav_standard` 计算 SR / SPL / Soft-SPL / NE / Oracle-SR，公式与 habitat 逐
字节一致（Soft-SPL 已与 habitat 自带度量在真实 episode 上交叉验证：1e-9 内一致）。

协议默认：`success_distance_m: 0.1`（上游任务默认，可在 `benchmark_settings`
覆盖）、500 控制步、前进 0.25 m、15° 或 30° 转向、RGB-D 640×480。

### Split A —— ObjectNav MP3D v1（无需新场景）

11 个 val 场景（`2azQ1b91cZZ`、`8194nk5LbLH`、`EU6Fwq7SyZv`、`QUCTc6BB5sX`、
`TbHJrupSAjP`、`X7HyMhZNoso`、`Z6MFQCViBuw`、`oLBMNvg9in8`、`pLe4wQe7qrG`、
`x8F5xyUWy9e`、`zsNo4HB9uLZ`）是任何 R2R 环境已有 MP3D 场景的子集——本机能跑
`r2r_ce` 就无需再下载场景。

1. 下载官方 episode 归档（单 zip，train+val+val_mini；路径段是 `m3d` 不是
   `mp3d`）——配方见[资产指南](../data/README.zh-CN.md#1-episode-数据集)。

2. 不加载仿真器验证选择（episode id 按 habitat 装载顺序定位——主文件，然后按场景
   排序的 `content/`）：

   ```bash
   python -B -c "
   from pathlib import Path
   from extensions.benchmarks.objectnav.objectnav.dataset import select_episodes
   root = Path('data').resolve()
   print(len(select_episodes(root / 'datasets/objectnav/mp3d/v1/val/val.json.gz', root)))
   "
   ```

3. 资源映射：无需新增——带 `settings.data_root` 的 `habitat030`（或 `habitat024`）
   运行时已覆盖该 benchmark。

4. 运行（NaVid 把类别目标渲染进自己的固定 prompt，属于方法 bundle 身份的一部分）：

   ```bash
   python -B -m nav_eval plan \
     --config configs/experiments/navid-objectnav.json
   python -B -m nav_eval run \
     --config configs/experiments/navid-objectnav.json \
     --output runs/objectnav-smoke
   python -B -m nav_eval evaluate --run runs/objectnav-smoke/<run-id>
   ```

   全量 2,195-episode split 请删除实验配置中的 `"episodes"`/`"episode_limit"`。
   免模型的绑定 smoke：

   ```bash
   python -m nav_eval benchmarks smoke --config configs/benchmarks/objectnav.json \
     --output runs/benchmark-smoke/objectnav.json
   ```

### Split B —— ObjectNav HM3D v2（challenge 版）

1. Episodes（约 248 MiB；布局 `objectnav_hm3d_v2/{train,val,val_mini}`）——
   [habitat CDN](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip)，
   解压到 `<data-root>/datasets/objectnav/hm3d/v2/val/val.json.gz` 存在。

2. HM3D 场景受 Matterport 许可约束——token 与下载器配方见
   [资产指南：HM3D](../data/README.zh-CN.md#2-场景)。全量 val 场景用
   `hm3d_val_v0.2`；场景放在 `data/scene_datasets/hm3d/` 下。

3. 把实验改成 `"benchmark_settings": {"benchmark_id": "objectnav_hm3d_val"}`
   后按上法运行。

### 未接入（暂缓）

- **Gibson ObjectNav**：没有官方 episode 下载（旧 habitat CDN 路径返回 403，任何
  habitat-lab 版本都未发布过 episodes）；社区镜像使用非官方 `gibson/v1.1` 布局，
  Gibson 场景还需 StanfordVL 协议。需要时把社区 episodes 放到
  `data/datasets/objectnav/gibson/v1.1/` 并重新加入 family 映射。
- **HSSD / ProcTHOR ObjectNav**：episodes 官方托管在 Dropbox
  （`objectnav_hssd-hab_v0.2.3.zip`），场景在 HuggingFace `hssd/hssd-hab`；获取并
  核对布局后，新增一条 `BENCHMARK_SPLITS` 即可接入。

## InstanceImageNav（`instance_imagenav`）

Habitat 0.3 绑定面向上游
`benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml` 任务及其两个
split：`instance_imagenav_hm3d_val` 与 `instance_imagenav_hm3d_val_mini`。
1000 步预算、0.25 m 前进、30 度转向、相机 tilt。公开目标是 `image_reference`，
由渲染的 `instance_imagegoal` 传感器支撑。目标图像保留原生 HxWx3 几何与 HFOV；
目标相机位姿、物体身份/类别与世界坐标是评测侧私有信息。

上游 episodes 放在
`<data-root>/datasets/instance_imagenav/hm3d/v2/<split>/<split>.json.gz`，伴随
`content/*.json.gz`；许可的 HM3D 场景放在 `<data-root>/scene_datasets/` 下。公开
episode ID 带场景限定（`scene-relative-path::upstream-id`），因为上游 ID 是场景内
局部的。非默认的 `content_scenes_path` 元数据会被拒绝，Habitat 装载行为不会静默
改变。

图像能力方法必须同时声明 `accepts_goals: ["image_reference"]` 与包含
`instance_imagegoal` 的 `requires_sensors`；plan 解析会在启动前拒绝缺失声明。
内置语言方法没有这些声明，因此不会作为 InstanceImageNav 策略提供——请按
[接入指南](integration.zh-CN.md)接入图像目标方法。

## 社交导航（`hssd_socialnav`）

Habitat 3.0 的社交导航任务（[论文](https://arxiv.org/abs/2310.13724)，任务类型
`RearrangePddlSocialNavTask-v0`）：Spot 机器人找到并接近在 HSSD 住宅中游走的仿真
人形。绑定把平台原语 1:1 转换为 `agent_0_base_velocity` 控制步驱动机器人（速度 =
原语位移 × `ctrl_freq`，按 habitat `BaseVelAction` 精确裁剪）；人形由上游脚本
`OracleNavRandCoordAction` 步进，除 RGB-D 外对方法不可见。

两层指标，各司其职：

- `socialnav_standard`（离线、只读证据）：基于机器人→人形 `DistToGoal` 采样位置的
  SR / SPL / Soft-SPL / NE / Oracle-SR——与本仓库其他 benchmark 直接可比。
- 证据 parity 层额外记录 habitat 官方的 `nav_seek_success`、`nav_to_pos_success`、
  `rot_dist_to_goal`、碰撞计数与完整 `social_nav_stats` 字典。habitat 3.0 论文的
  Finding Success / SPS / Following Rate / Collision Rate 请从该层读取。

协议默认：`success_distance_m: 1.5`（上游 `nav_to_pos_succ` 默认）、
`max_task_steps: 750`、前进 0.25 m、15° 转向。

### 配置（全部资产来自 HuggingFace `ai-habitat` 数据集）

用 `habitat030` 环境的 habitat-sim python 运行下载器（`--username/--password`
为 HF 凭据；以下 UID 均免费但需在 HF 页面确认许可）：

```bash
python -m habitat_sim.utils.datasets_download --username <hf-user> --password <hf-token> \
  --uids hab3-episodes --data-path <data-root>/          # val/social_rearrange.json.gz（约 1.2k episode）
python -m habitat_sim.utils.datasets_download --username <hf-user> --password <hf-token> \
  --uids hab3_bench_assets --data-path <data-root>/       # hab3-hssd 场景 + benchmark 配置
python -m habitat_sim.utils.datasets_download --username <hf-user> --password <hf-token> \
  --uids ycb --data-path <data-root>/                     # YCB 物体配置
python -m habitat_sim.utils.datasets_download --username <hf-user> --password <hf-token> \
  --uids hab_spot_arm --data-path <data-root>/            # 机器人 URDF/mesh
python -m habitat_sim.utils.datasets_download --username <hf-user> --password <hf-token> \
  --uids habitat_humanoids --data-path <data-root>/       # female_2 人形（agent_1）
```

然后核对绑定期望的布局（一切相对 `data_root` 解析）：

```text
<data-root>/
├── datasets/hssd/rearrange/val/social_rearrange.json.gz
├── objects/{ycb,amazon_berkeley,google_object_dataset}/configs/
├── robots/hab_spot_arm/urdf/hab_spot_arm.urdf
├── humanoids/humanoid_data/female_2/female_2.urdf
└── scene_datasets/<hab3-hssd scenes>
```

（`hab3-episodes` 与 `hab3_bench_assets` UID 会把树符号链接/克隆进 `data/`；把
各部分移动或链接到上述布局。）

运行：

```bash
python -B -m nav_eval run \
  --config configs/experiments/navid-socialnav.json
```

## VLN-VERSE（`vlnverse`）

NVIDIA Isaac Sim 渲染的 33 个合成 kujiale 场景中的指令跟随导航（VLN-VERSE EMR
challenge 数据）：`vlnverse_fine_val_unseen`（825 episode）与
`vlnverse_coarse_val_unseen`（835 episode），由 `isaacsim500` 绑定服务。指标集
`vlnverse_standard` 在采集的度量空间位置上计算 SR / SPL / NE / OSR。

首次执行 `python -B scripts/setup_benchmark.py vlnverse` 准备仓库内的任务运行时，
无需 challenge checkout。所有资产统一放在永久登记的 `data_root` 下：

- Episodes：`<data-root>/datasets/vlnverse/{fine,coarse}_val_unseen.json.gz`
  （从 challenge 仓库重新导出）。
- 场景：`<data-root>/scene_data/vlnverse/<scan>/`（kujiale 场景数据）。
- 机器人 USD：`<data-root>/Embodiments/`（vln-pe 机器人）。

`configs/local.json` 中永久登记的 `isaacsim500` 运行时需提供：

| 字段 | 值 |
|---|---|
| `python` | `envs/nav_isaac/bin/python`（Python 3.11 / Isaac Sim 5.0.0） |
| `settings` | `data_root` |
| `cwd` | 省略；绑定使用仓库内任务运行时 |
| `env` | `ISAAC_PATH=<pip-installed isaacsim directory>`、`OMNI_KIT_ACCEPT_EULA=YES`、`PYTHONNOUSERSITE=1`；由 `scripts/register_isaac.py` 登记 |
| `pythonpath` / `library_paths` | 省略；新环境包含自身的 Python 和 CUDA 依赖 |
| `timeout_s` / `min_free_memory_mib` | 从宽：Kit 启动慢（参考机上 1500 s / 12 GiB） |

从 `configs/experiments/navida-vlnverse.json` 起步（`benchmark: vlnverse`、
`simulator: isaacsim500`），用 `benchmark_settings.benchmark_id` 选 split。该
benchmark 不需要 MP3D 资产。

固定 12 条路线、3 个场景的诊断配置位于 `configs/experiments/`：
`streamvln-vlnverse-diagnostic.json` 和 `internvla-n1-vlnverse-diagnostic.json`。
两者显式设置水平初始视角 `camera_pitch_deg: 0` 与 USD 实际的 90° HFOV；
省略俯仰参数则保留 USD 的 −30° 安装姿态。默认使用物理 H1 控制器；实验选项
`robot_flash: true` 使用上游带碰撞检查的离散 flash 控制器，属于不同协议。

R2R/RxR 与 VLNVerse 绑定默认允许 500 次导航动作。相机探测只消耗
`max_control_steps`：开启 `allow_tilt` 时默认 5000，否则等于
`max_navigation_steps`。显式控制上限（包括 smoke 限制）优先。VLNVerse 还保留
上游物理步数上限 `max_task_steps`。记录保留原有 `steps`/`control_tick`，另加
`navigation_steps`、`control_steps` 和 `environment_end_reason`。
详见[实测结果与协议边界](vlnverse-validation.md)。

## SCAND（`scand`）：离线录制数据评测

SCAND 在本框架中没有仿真器绑定。其评测器比较对齐的预测轨迹与参考 JSONL 轨迹。
每行恰有 `episode_id`、`timestamp_s`、`frame_id`、`position_xy_m`；参考与预测的
episode ID、时间戳、坐标系必须一致。输出等权 episode 的 ADE/FDE。

```bash
python -B -m nav_eval benchmarks evaluate-offline scand \
  --reference data/scand/reference.jsonl --predictions runs/scand/predictions.jsonl \
  --output runs/scand/diagnostic.json

# 在 ROS1 环境中导出一个显式选择的 nav_msgs/Odometry 话题。
python -B -m nav_eval benchmarks export-scand \
  --bag recording.bag --topic /odom --episode-id trial-001 \
  --output runs/scand/trial-001.jsonl
```

导出器使用每条消息的 odometry header 时间戳，且不覆盖已有输出文件。ADE/FDE 只度量
模仿误差——不是闭环成功率或社交合规指标。完整的录制 bag 流程见
[smoke 配方](../configs/benchmarks/README.md#scand-offline-check-without-a-simulator)。

## 路线图（已调研，未接入）

| Benchmark | 族 | 状态 | 权威来源 |
|---|---|---|---|
| ImageNav HM3D v3 | 图像目标导航 | 现有适配器面向 InstanceImageNav HM3D v2；v3 需装载器/数据对齐与图像能力方法 | `…/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip`（habitat CDN） |
| REVERIE-CE / CVDN-CE / SOON-CE | VLN 变体 | 无官方统一下载；转换版（REVE-CE，RAL 2022）散见于论文仓库——需先冻结来源 | REVE-CE 论文仓库 |
| Arena-RoNav（Arena 5.0） | Isaac 社交导航 | 可沿用 `vlnverse` 的 isaac 绑定模式；仓库（`Arena-Rosnav/arena-rosnav`）需先审计出稳定评测 API | [5.arena-rosnav.org](https://5.arena-rosnav.org/) |
| ObjectNav HSSD / ProcTHOR | 物体导航 | episodes 在 Dropbox；待获取与核对布局 | habitat-lab `DATASETS.md` |

## 许可与来源

| 数据 | 许可/获取 |
|---|---|
| ObjectNav MP3D v1 episodes | MP3D Terms of Use + CC BY-NC-SA 3.0 US（衍生自 Matterport3D） |
| ObjectNav HM3D v2 episodes | 随 HM3D 分发；场景受 Matterport 协议约束 |
| hab3 episodes / 资产 / Spot / 人形 | HF `ai-habitat/*`（episodes CC-BY-NC-4.0；人形动作另受 SMPL body license；Spot 自带许可） |
| HSSD 场景 | HF `hssd/hssd-hab`，CC BY-NC 4.0 |

下载落在不入库的 `data/` 目录；资源映射把主机路径隔离在仓库之外。`asset_files()`
记录每次运行实际使用的 episode 文件、场景、navmesh 与 URDF；身份摘要覆盖插件
代码——跨主机比较时请单独锁定数据来源。
