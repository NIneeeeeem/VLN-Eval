# 资产获取：数据集、场景与权重

[简体中文](README.zh-CN.md) | [English](README.md)

Nav-Eval 只包含代码——数据集、场景与权重均从官方来源下载到本地（git 忽略）目录。
首次下载后只需在 `configs/local.json` 登记一次。本页列出全部下载来源与确切的目标目录布局。

从零配置统一使用仓库内的 `envs/nav_streamvln`、`envs/nav_vlm`、`envs/nav_habitat030`，见[安装入口](../README.zh-CN.md)。公开验证数据可直接下载：

```bash
python -B scripts/download_assets.py objectnav instance_imagenav hm3d_ovon \
  multion_mp3d multion_objects multion_hssd
python -B scripts/download_hssd_scene.py --scene 102816036
python -B scripts/download_assets.py goat_bench
python -B scripts/download_goat_cache.py
```

归档与 URL、SHA256、解压文件清单保存在 `data/downloads/`。脚本支持断点续传、只解压验证 split，并保留已有文件。GOAT 的 Drive 下载和目标缓存独立于 HM3D 场景；完整路径见[专用配方](../configs/benchmarks/README.md)。下载失败会返回非零，不代表资产已准备完毕。Hugging Face 官方站不可达时可在下载命令前加 `HF_ENDPOINT=https://hf-mirror.com`，记录会注明实际 endpoint。

专用 benchmark（OVON、GOAT-Bench、MultiON、HSSD SocialNav 资产）与免模型 smoke
检查有自己的配方，见
[configs/benchmarks/README.md](../configs/benchmarks/README.md)。

## 目录约定

所有资产放在这些 git 忽略目录下（或由 `nav_eval configure` 登记的任意本地路径）。
相对路径以仓库根目录解析：

```text
data/
├── datasets/r2r/{val_seen,val_unseen}/     # R2R-CE splits
├── datasets/rxr/val_unseen/                # RxR-CE splits
├── datasets/objectnav/{mp3d,hm3d}/         # ObjectNav splits
└── scene_datasets/{mp3d,hm3d}/             # 场景 mesh + navmesh
checkpoints/<method>/                           # 方法 checkpoint
```

方法推理源码已包含在本仓库。准备 Python 依赖环境，并通过 `nav_eval configure` 只登记一次解释器。

## 1. Episode 数据集

| 目标路径 | 内容 | 来源 |
|---|---|---|
| `datasets/r2r/val_unseen/val_unseen.json.gz` | R2R-CE val_unseen——1,839 episode / 11 场景 | [R2R_VLNCE_v1-3](https://drive.google.com/file/d/1T9SjqZWyR2PCLSXYkFckfDeIs6Un0Rjm/view)（[nav-habitat](https://github.com/jkrishnavs/nav-habitat) 布局） |
| `datasets/r2r/val_seen/val_seen.json.gz` | R2R-CE val_seen——778 episode / 53 场景 | 同一归档 |
| `datasets/rxr/val_unseen/val_unseen_guide.json.gz` | RxR-CE val_unseen guide——11,006 episode / 11 场景（en/hi/te） | [RxR_VLNCE_v0](https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view)（[rxr-ce](https://github.com/rvlab/rxr-ce)） |
| `datasets/objectnav/mp3d/v1/val/val.json.gz` | ObjectNav MP3D v1 val——2,195 episode | [habitat CDN](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip) |
| `datasets/objectnav/hm3d/v2/val/val.json.gz` | ObjectNav HM3D v2 val | [habitat CDN](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip) |
| `datasets/instance_imagenav/hm3d/v2/<split>/<split>.json.gz` | InstanceImageNav HM3D v2（`val` / `val_mini`）+ `content/*.json.gz` | [habitat CDN](https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v2/instance_imagenav_hm3d_v2.zip) |
| `datasets/ovon/hm3d/v1/val_seen/val_seen.json.gz` | OVON episode 与 content 分片 | [官方归档](https://huggingface.co/datasets/nyokoyama/hm3d_ovon/resolve/main/hm3d.tar.gz) |
| `datasets/goat_bench/hm3d/v1/val_seen/val_seen.json.gz` | GOAT episode | [官方 Drive 归档](https://drive.google.com/file/d/1N0UbpXK3v7oTphC4LoDqlNeMHbrwkbPe/view) |
| `datasets/multinav/3_ON/val/val.json.gz` | 原始 MP3D MultiON | [官方归档](https://aspis.cmpt.sfu.ca/projects/multion/multinav.zip) |
| `datasets/minival/minival.json.gz` | HSSD MultiON 与 content 分片 | [官方 minival](https://aspis.cmpt.sfu.ca/projects/langmon/minival) |
| `objects/multion/` | MP3D MultiON 目标圆柱 | [官方物体](https://aspis.cmpt.sfu.ca/projects/multion/objects.zip) |
| `goat-assets/goal_cache/` | GOAT 物体、语言、图像特征 | [官方缓存](https://huggingface.co/datasets/axel81/goat-bench) |

先解压到临时目录，确认顶层结构后再把 split 目录复制到目标布局（避免多出一层
归档名目录）。`*_gt.json.gz` 伴随文件可选——评分读取绑定现场采集的证据，不用 GT
文件。

ObjectNav MP3D 下载并解压验证集：

```bash
python -B scripts/download_assets.py objectnav
```

不加载仿真器即可校验冻结的 episode 选择——`missing scenes` 错误会列出还缺的
MP3D 场景：

```bash
python -B -c "
from pathlib import Path
from extensions.benchmarks.vln.r2r_ce.dataset import select_episodes
root = Path('data').resolve()
print(len(select_episodes(root / 'datasets/r2r/val_unseen/val_unseen.json.gz', root)))
"
```

## 2. 场景

### MP3D（R2R-CE、RxR-CE、ObjectNav MP3D）

MP3D mesh 受 Matterport 许可约束：

1. 在 <https://niessner.github.io/Matterport/> 接受条款，从
   [habitat DATASETS.md](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md)
   获取下载脚本。
2. 把场景下载到 `data/scene_datasets/mp3d/<scan>/<scan>.glb`（连同 `.navmesh`
   navmesh；语义任务还需 `.ply`/`.scn`）。

R2R val_unseen 用 11 个场景；全量 91 场景（约 21 GB）同时覆盖 val_seen、RxR 与
ObjectNav MP3D——只存一份，跨 benchmark 共享。VLN-CE 的
[数据说明](https://github.com/jacobkrantz/VLN-CE#data)描述了预期布局。

### HM3D（ObjectNav HM3D、InstanceImageNav、OVON、GOAT）

HM3D 需要签署 Matterport 研究协议：
<https://matterport.com/habitat-matterport-3d-research-dataset>。在
<https://my.matterport.com/settings/account/devtools> 生成 API token，然后在
habitat 环境中用 habitat-sim 下载器（token id = 用户名，secret = 密码）：

```bash
envs/nav_habitat030/bin/python -B -m habitat_sim.utils.datasets_download \
  --username "$MATTERPORT_TOKEN_ID" --password "$MATTERPORT_TOKEN_SECRET" \
  --uids hm3d_val_v0.2 --data-path data/
```

`hm3d_minival_v0.2` 取 minival 子集；`hm3d_train_v0.2` 补训练场景。HM3D 语义任务
需要版本匹配的语义标注，不只是视觉 mesh。token 只保存在本地——绝不写进 JSON 或
提交入库。

### HSSD（SocialNav、MultiON HSSD）

HSSD 场景在 HuggingFace（[hssd/hssd-hab](https://huggingface.co/datasets/hssd/hssd-hab)，
CC BY-NC 4.0）；HSSD SocialNav 资产包用 habitat-sim 下载器获取——见
[HSSD SocialNav 配置](../docs/benchmarks.md#social-navigation-hssd_socialnav)。

MultiON 的最小完整场景用 `scripts/download_hssd_scene.py` 获取：下载真实场景配置及引用的 stage、物体、碰撞网格与语义词表到 `data/scene_datasets/fphab/`。默认 `102816036` 对应 minival 中的 5 条 episode；示例 YAML 只选择该场景。其他场景逐个用 `--scene` 获取，全部备齐后删除 YAML 的 `content_scenes` 限制。此下载不包含 SocialNav 的 humanoid/episode 资产。

## 3. 方法权重

| 方法 | 基座模型 | 参考实现（已内置） | 权重 |
|---|---|---|---|
| `streamvln` | LLaVA-Video (Qwen1.5) | [InternRobotics/StreamVLN](https://github.com/InternRobotics/StreamVLN) | [mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln](https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln) |
| `navid` | 7B 视频 VLM | [jzhzhang/NaVid-VLN-CE](https://github.com/jzhzhang/NaVid-VLN-CE) | [Jzzhang/NaVid](https://huggingface.co/Jzzhang/NaVid) |
| `uni_navid` | 7B 视频 VLM | [jzhzhang/Uni-NaVid](https://github.com/jzhzhang/Uni-NaVid) | [Jzzhang/Uni-NaVid](https://huggingface.co/Jzzhang/Uni-NaVid) 的 `uninavid-7b-full-224-video-fps-1-grid-2` 子目录 |
| `navila` | 开源 VLM | [AnjieCheng/NaVILA](https://github.com/AnjieCheng/NaVILA) | 上游 README 链接的 HF collection |
| `awarevln` | InternVL2-8B | [GWxuan/AwareVLN](https://github.com/GWxuan/AwareVLN) | 上游 README 中的 checkpoint 链接 |
| `navida` | Qwen2.5-VL-3B | 经 Transformers 加载，无需仓库 | 暂未公开（arXiv 2601.18188） |
| `activevln` | Qwen2.5-VL | 内置 Transformers 策略 | `checkpoints/activevln/{rl,sft}_{r2r,rxr}` |
| `janusvln` | VLM + VGGT | 内置 `qwen_vl` runtime | 自包含 checkpoint（内嵌 VGGT） |
| `internvla-n1` | 双系统 VLA | 内置 `internnav` runtime | System 1/2 权重向作者获取 |
| `onevla` | VLA + flow head | 内置 OneVLA runtime | `run/checkpoints/*.pt` + `run/config.yaml` + `run/dataset_statistics.json` |
| `gavln` | SigLIP + VGGT | 内置 GA-VLN runtime | 模型 + `vision_tower`（SigLIP）+ `vggt_path` |

示例——StreamVLN 权重：

```bash
huggingface-cli download mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln \
  --local-dir checkpoints/streamvln
```

所有内置方法的推理实现都在 `extensions/methods/` 中，只需安装 Python 依赖并下载权重。
NaVid / Uni-NaVid 还需 EVA 权重：把 `eva_vit_g.pth` 放入 checkpoint 目录，或通过
`vision_tower` 指向已有权重文件。InternVLA-N1 的辅助 DepthAnything 权重也放入
checkpoint 目录（见[方法环境](../docs/deployment.zh-CN.md#方法环境)）。
缺少 checkpoint 的运行会产出类型化的 checkpoint-missing 失败，不会被静默跳过。

## 4. 下载后只登记一次

```bash
python -m nav_eval configure \
  --method streamvln \
  --checkpoint checkpoints/streamvln \
  --method-python envs/nav_streamvln/bin/python \
  --simulator habitat024 \
  --environment-python envs/nav_streamvln/bin/python \
  --data-root data
```

该命令创建或合并 git 忽略的永久 `configs/local.json`。多方法登记按插件 ID 合并，
添加方法不会覆盖已有的方法或仿真器条目。`plan` 与 `run` 自动加载它。已有资源映射只可作为一次性导入输入：

```bash
python -m nav_eval configure --from configs/resources/my-host.json
```

首次运行会记录权重与资产的内容摘要；替换权重后混入旧 run 会被拒绝。

## 全集规模

全量 split 的 episode 数：R2R val_unseen 1,839 / val_seen 778；RxR val_unseen
guide 11,006（多语言子集通过 `benchmark_settings.languages` 过滤）；ObjectNav MP3D
val 2,195；VLNVerse fine 825 / coarse 835。实验配置默认带小子集——删掉
`episodes` / `episode_limit` 即全量运行。

## 许可证

| 数据 | 许可 / 获取方式 |
|---|---|
| R2R-CE / RxR-CE episodes | 上游发布，衍生自 Matterport3D 全景——适用 MP3D 条款 |
| MP3D 场景 | Matterport3D Terms of Use |
| ObjectNav MP3D episodes | CC BY-NC-SA 3.0 US（衍生自 Matterport3D） |
| HM3D 场景 + episodes | Matterport 研究协议 |
| HSSD 场景 | CC BY-NC 4.0 |
| hab3 episodes / Spot / 人形机器人 | HF `ai-habitat/*`（episodes CC-BY-NC-4.0；人形动作受 SMPL body license 约束；Spot 自带许可） |
| 模型权重 | 各上游仓库许可证 |

下载都落在不入库的 `data/` 目录；本机路径通过已忽略的 `configs/local.json` 与仓库隔离。`asset_files()`
记录每次运行实际使用的 episode 文件、场景、navmesh 与 URDF——跨主机比较时请把
数据来源与 run 记录一起锁定。
