# 资产获取:数据、权重与环境 / Asset Acquisition

English summary: this repository ships **no datasets, scenes, weights or upstream
source**. Everything below is fetched from the network and placed in local
(git-ignored) directories, then wired into a run through a host resource map
(`configs/resources/*.json`). Relative paths in resource maps resolve from the
repository root, so run all commands from the repo root as the docs do.

## 目录约定

所有资产放在仓库下这些目录(已被 `.gitignore` 排除,绝不入库),或资源映射指向的任何本地路径:

```text
data/
├── datasets/r2r/{val_seen,val_unseen}/    # R2R-CE splits
├── datasets/rxr/val_unseen/               # RxR-CE splits
└── scene_datasets/mp3d/                   # MP3D 场景 (91 个, 约 21 GB)
weights/<method>/                          # 方法 checkpoint
source/<upstream>/                         # 上游方法源码 checkout
envs/<name>/bin/python                     # 上游依赖环境的解释器
```

## 1. 数据集(全部来自上游官方发布)

| 本地路径 | 内容 | 上游来源 |
|---|---|---|
| `datasets/r2r/val_unseen/val_unseen.json.gz` | R2R-CE val_unseen 全集(1839 episodes / 11 场景) | [R2R-CE 官方发布](https://github.com/jkrishnavs/nav-habitat)(habitat 0.2.x 数据布局) |
| `datasets/r2r/val_unseen/val_unseen_gt.json.gz` | 官方 GT reference(评测证据由 adapter 现场采集,GT 备用) | 同上 |
| `datasets/r2r/val_seen/` | R2R-CE val_seen 全集(778 episodes / 53 场景) | 同上 |
| `datasets/rxr/val_unseen/val_unseen_guide.json.gz` | RxR-CE val_unseen guide 全集(11006 episodes / 11 场景;en/hi/te 多语言) | [RxR-CE 官方发布](https://github.com/rvlab/rxr-ce) |
| `datasets/rxr/val_unseen/val_unseen_guide_gt.json.gz` | RxR 官方 GT(nDTW 等备用) | 同上 |
| `scene_datasets/mp3d/` | 全部 91 个 MP3D 场景,覆盖 R2R/RxR 全部 split | [habitat DATASETS.md](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md)(需签署 MP3D 许可后用官方脚本下载) |

R2R/RxR 按上游 release 的目录结构原样放置即可;MP3D 需先在
[MP3D 官网](https://niessner.github.io/Matterport/) 签署许可,再用 habitat 页面提供的
下载脚本获取 `mp3d` 场景目录。相同 MP3D 只存一份;split、许可与来源必须锁定。

RxR 的 habitat dataset 装载器(`RxR-VLN-CE-v1` 注册)逐字复制自 StreamVLN 仓库
`streamvln/habitat_extensions/rxr_vln_dataset.py`,来源注明于
`extensions/benchmarks/rxr_ce/dataset.py` 文件头。

## 2. 方法权重(checkpoint)

| 方法 ID | 基座模型 | 上游代码(`repo_path` 指向它) | 权重来源 |
|---|---|---|---|
| `streamvln` | LLaVA-Video | [InternRobotics/StreamVLN](https://github.com/InternRobotics/StreamVLN) | [mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln](https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln)(benchmark 复现版 checkpoint) |
| `navid` | 7B 视频 VLM | [jzhzhang/NaVid-VLN-CE](https://github.com/jzhzhang/NaVid-VLN-CE) | [Jzzhang/NaVid](https://huggingface.co/Jzzhang/NaVid) |
| `uni_navid` | 7B 视频 VLM | [jzhzhang/Uni-NaVid](https://github.com/jzhzhang/Uni-NaVid) | [Jzzhang/Uni-NaVid](https://huggingface.co/Jzzhang/Uni-NaVid) 中 `uninavid-7b-full-224-video-fps-1-grid-2` 子目录 |
| `navila` | VLM | [AnjieCheng/NaVILA](https://github.com/AnjieCheng/NaVILA) | 上游 README 链接的 Hugging Face collection |
| `awarevln` | InternVL2-8B | [GWxuan/AwareVLN](https://github.com/GWxuan/AwareVLN) | 上游 README 中的 checkpoint 链接 |
| `navida` | Qwen2.5-VL-3B | NaVIDA(arXiv 2601.18188) | **暂未公开发布**,发布后在此补充链接 |

多数方法的适配器逐行复用上游预处理代码,因此除 checkpoint 外还需要
`git clone` 对应上游仓库作为 `repo_path`。NaVIDA 权重未公开期间,未提供
checkpoint 时框架产出类型化的 checkpoint-missing 失败,不会静默跳过。

## 3. 运行环境(上游各自安装,控制面不代装)

每个方法/仿真器保留上游依赖环境;在资源映射中为每个角色指定该环境的解释器:

| 角色 | 环境要点 |
|---|---|
| `navida` 推理 | Python ≥3.10 + PyTorch + Transformers + flash-attn + qwen-vl-utils(按上游 NAVIDA 依赖) |
| `awarevln` / `streamvln` / `navid` / `navila` 推理 | 按各自上游仓库 README 安装(版本以上游发布为准) |
| `habitat017` / `habitat024` / `habitat030` 仿真 | [habitat-sim](https://github.com/facebookresearch/habitat-sim) / [habitat-lab](https://github.com/facebookresearch/habitat-lab) 对应版本 tag(0.1.7 / 0.2.4 / 0.3.0);habitat024 需在 `pythonpath` 加入上游 habitat-lab/baselines 源码,habitat017 需把 habitat_sim `_ext` 加入 `library_paths`(CLI 已内置处理,资源文件示例见 `configs/resources/example.json`) |
| `isaacsim500` 仿真 | [NVIDIA Isaac Sim 5.0](https://developer.nvidia.com/isaac/sim);资源映射需指定其 pip 预打包目录(`pythonpath`)与 `challenge_repo` |

## 4. 接线:资源映射

```bash
cp configs/resources/example.json configs/resources/my-host.json
# 填入:解释器(envs/*/bin/python 或任何绝对路径)、GPU、weights/、source/、data 路径
python -B -m nav_eval plan --config configs/experiments/navida-r2r.json --resources configs/resources/my-host.json
```

`example.json` / `parallel.example.json` 使用仓库相对路径演示 `weights/`、`source/`、
`envs/`、`data` 的引用方式;你自己的资源映射(建议命名 `my-host.json`,机器专属,
不入库)可自由使用绝对路径。首个 run 会记录权重与资产的内容摘要,替换权重后混入
旧 run 会被拒绝。

InternVLA-N1 不在本仓库实现;外部插件需自行记录 System 1 与 System 2 的权重版本和
摘要,不写入运行时镜像。

## 5. 复现入口

```bash
MODE=plan bash scripts/inference.sh navida r2r configs/resources/my-host.json   # 只校验配置
bash scripts/inference.sh navida r2r configs/resources/my-host.json             # 采集并自动评分
python -B -m nav_eval resume --run runs/<run-id>                                # episode 级恢复
```

全集规模:R2R val_unseen 1839 / val_seen 778 / RxR val_unseen guide 11006
(RxR 多语言子集通过实验配置的 `benchmark_settings.languages` 过滤;详见
[README](../README.zh-CN.md))。
