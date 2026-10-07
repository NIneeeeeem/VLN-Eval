# Assets: Datasets, Scenes and Weights

[简体中文](README.zh-CN.md) | English

Nav-Eval ships code only — datasets, scenes and weights are downloaded from
their official sources into local (git-ignored) directories. After the first
download, register their paths once in `configs/local.json`; this page lists
every download source and the exact target layout.

Start with the fresh `nav_streamvln`, `nav_vlm` and `nav_habitat030` prefixes
in the [installation guide](../README.md). Public evaluation downloads:

```bash
python -B scripts/download_assets.py objectnav instance_imagenav hm3d_ovon \
  multion_mp3d multion_objects multion_hssd
python -B scripts/download_hssd_scene.py --scene 102816036
python -B scripts/download_assets.py goat_bench
python -B scripts/download_goat_cache.py
```

Archives and URL/SHA256/file receipts live in `data/downloads/`. Downloads
resume, extract only evaluation splits and preserve existing files. GOAT's
Drive episodes and goal caches are separate from licensed HM3D scenes; see
[specialized recipes](../configs/benchmarks/README.md). Failed downloads return
nonzero. If the official Hugging Face endpoint is unreachable, prefix those
commands with `HF_ENDPOINT=https://hf-mirror.com`; receipts record the endpoint.
The HSSD helper fetches one complete real scene and all referenced objects,
collision assets and semantic lexicon into `data/scene_datasets/fphab/`.
Scene 102816036 contains five minival episodes; example YAMLs restrict the
selection to it. Fetch other scenes with `--scene`, then remove `content_scenes`
for full minival. SocialNav's humanoid/episode assets are separate.

Specialized benchmarks (OVON, GOAT-Bench, MultiON, HSSD SocialNav assets) and
model-free smoke checks have their own recipes in
[configs/benchmarks/README.md](../configs/benchmarks/README.md).

## Directory layout

All assets live under these git-ignored directories (or another local path
registered with `nav_eval configure`). Relative paths resolve from the
repository root:

```text
data/
├── datasets/r2r/{val_seen,val_unseen}/     # R2R-CE splits
├── datasets/rxr/val_unseen/                # RxR-CE splits
├── datasets/objectnav/{mp3d,hm3d}/         # ObjectNav splits
└── scene_datasets/{mp3d,hm3d}/             # scene meshes + navmeshes
checkpoints/<method>/                           # method checkpoints
```

Method inference source is bundled. Prepare Python dependency environments and
register their interpreters once with `nav_eval configure`.

## 1. Episode datasets

| Target path | Content | Source |
|---|---|---|
| `datasets/r2r/val_unseen/val_unseen.json.gz` | R2R-CE val_unseen — 1,839 episodes / 11 scenes | [R2R_VLNCE_v1-3](https://drive.google.com/file/d/1T9SjqZWyR2PCLSXYkFckfDeIs6Un0Rjm/view) ([nav-habitat](https://github.com/jkrishnavs/nav-habitat) layout) |
| `datasets/r2r/val_seen/val_seen.json.gz` | R2R-CE val_seen — 778 episodes / 53 scenes | same archive |
| `datasets/rxr/val_unseen/val_unseen_guide.json.gz` | RxR-CE val_unseen guide — 11,006 episodes / 11 scenes (en/hi/te) | [RxR_VLNCE_v0](https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view) ([rxr-ce](https://github.com/rvlab/rxr-ce)) |
| `datasets/objectnav/mp3d/v1/val/val.json.gz` | ObjectNav MP3D v1 val — 2,195 episodes | [habitat CDN](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip) |
| `datasets/objectnav/hm3d/v2/val/val.json.gz` | ObjectNav HM3D v2 val | [habitat CDN](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip) |
| `datasets/instance_imagenav/hm3d/v2/<split>/<split>.json.gz` | InstanceImageNav HM3D v2 (`val` / `val_mini`) + `content/*.json.gz` | [habitat CDN](https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v2/instance_imagenav_hm3d_v2.zip) |
| `datasets/ovon/hm3d/v1/val_seen/val_seen.json.gz` | OVON episodes and content shards | [official archive](https://huggingface.co/datasets/nyokoyama/hm3d_ovon/resolve/main/hm3d.tar.gz) |
| `datasets/goat_bench/hm3d/v1/val_seen/val_seen.json.gz` | GOAT episodes | [official Drive archive](https://drive.google.com/file/d/1N0UbpXK3v7oTphC4LoDqlNeMHbrwkbPe/view) |
| `datasets/multinav/3_ON/val/val.json.gz` | Original MP3D MultiON | [official archive](https://aspis.cmpt.sfu.ca/projects/multion/multinav.zip) |
| `datasets/minival/minival.json.gz` | HSSD MultiON and content shards | [official minival](https://aspis.cmpt.sfu.ca/projects/langmon/minival) |
| `objects/multion/` | MP3D MultiON goal cylinders | [official objects](https://aspis.cmpt.sfu.ca/projects/multion/objects.zip) |
| `goat-assets/goal_cache/` | GOAT object/language/image embeddings | [official caches](https://huggingface.co/datasets/axel81/goat-bench) |

Extract each archive into a staging directory first, then copy the split
folders to the target layout (avoid an extra archive-name level). The
`*_gt.json.gz` companions are optional — scoring reads evidence captured by
the binding, not GT files.

ObjectNav MP3D download and extraction (evaluation splits only):

```bash
python -B scripts/download_assets.py objectnav
```

Verify the frozen episode selection without loading a simulator — a
`missing scenes` error lists the MP3D scenes still to fetch:

```bash
python -B -c "
from pathlib import Path
from extensions.benchmarks.vln.r2r_ce.dataset import select_episodes
root = Path('data').resolve()
print(len(select_episodes(root / 'datasets/r2r/val_unseen/val_unseen.json.gz', root)))
"
```

## 2. Scenes

### MP3D (R2R-CE, RxR-CE, ObjectNav MP3D)

MP3D meshes are license-gated by Matterport:

1. Accept the terms at <https://niessner.github.io/Matterport/> and get the
   download script from
   [habitat DATASETS.md](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md).
2. Download scenes into `data/scene_datasets/mp3d/<scan>/<scan>.glb` (with
   navmeshes `.navmesh` and, for semantic tasks, `.ply`/`.scn`).

R2R val_unseen uses 11 scenes; the full 91-scene set (~21 GB) also covers
val_seen, RxR and ObjectNav MP3D — store it once and share it across
benchmarks. The VLN-CE [data guide](https://github.com/jacobkrantz/VLN-CE#data)
describes the expected layout.

### HM3D (ObjectNav HM3D, InstanceImageNav, OVON, GOAT)

HM3D requires a Matterport research agreement:
<https://matterport.com/habitat-matterport-3d-research-dataset>. Generate an
API token at <https://my.matterport.com/settings/account/devtools>, then use
the habitat-sim downloader from a habitat environment (token id = username,
secret = password):

```bash
envs/nav_habitat030/bin/python -B -m habitat_sim.utils.datasets_download \
  --username "$MATTERPORT_TOKEN_ID" --password "$MATTERPORT_TOKEN_SECRET" \
  --uids hm3d_val_v0.2 --data-path data/
```

`hm3d_minival_v0.2` fetches the minival subset; `hm3d_train_v0.2` adds
training scenes. HM3D semantic tasks need the version-matched semantic
annotations, not only visual meshes. Keep tokens local — never in JSON or Git.

### HSSD (SocialNav, MultiON HSSD)

HSSD scenes are on HuggingFace ([hssd/hssd-hab](https://huggingface.co/datasets/hssd/hssd-hab),
CC BY-NC 4.0); the HSSD SocialNav asset bundle is fetched with the
habitat-sim downloader — see
[HSSD SocialNav setup](../docs/benchmarks.md#social-navigation-hssd_socialnav).

## 3. Model weights

| Method | Base model | Reference implementation (bundled) | Weights |
|---|---|---|---|
| `streamvln` | LLaVA-Video (Qwen1.5) | [InternRobotics/StreamVLN](https://github.com/InternRobotics/StreamVLN) | [mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln](https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln) |
| `navid` | 7B video VLM | [jzhzhang/NaVid-VLN-CE](https://github.com/jzhzhang/NaVid-VLN-CE) | [Jzzhang/NaVid](https://huggingface.co/Jzzhang/NaVid) |
| `uni_navid` | 7B video VLM | [jzhzhang/Uni-NaVid](https://github.com/jzhzhang/Uni-NaVid) | [Jzzhang/Uni-NaVid](https://huggingface.co/Jzzhang/Uni-NaVid) — subfolder `uninavid-7b-full-224-video-fps-1-grid-2` |
| `navila` | open VLM | [AnjieCheng/NaVILA](https://github.com/AnjieCheng/NaVILA) | HF collection linked in the upstream README |
| `awarevln` | InternVL2-8B | [GWxuan/AwareVLN](https://github.com/GWxuan/AwareVLN) | checkpoint links in the upstream README |
| `navida` | Qwen2.5-VL-3B | loads via Transformers — no repo | not yet public (arXiv 2601.18188) |
| `activevln` | Qwen2.5-VL | bundled Transformers policy | `checkpoints/activevln/{rl,sft}_{r2r,rxr}` |
| `janusvln` | VLM + VGGT | bundled `qwen_vl` runtime | self-contained checkpoint (embedded VGGT) |
| `internvla-n1` | dual-system VLA | bundled `internnav` runtime | System 1/2 checkpoints from the authors |
| `onevla` | VLA + flow head | bundled OneVLA runtime | `run/checkpoints/*.pt` + `run/config.yaml` + `run/dataset_statistics.json` |
| `gavln` | SigLIP + VGGT | bundled GA-VLN runtime | model + `vision_tower` (SigLIP) + `vggt_path` |

Example — StreamVLN checkpoint:

```bash
huggingface-cli download mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln \
  --local-dir checkpoints/streamvln
```

All built-in inference implementations are bundled under `extensions/methods/`.
Install the method's Python dependencies and download weights; no separate
method repository is needed. NaVid / Uni-NaVid additionally need EVA weights:
place `eva_vit_g.pth` inside the checkpoint directory or set `vision_tower` to
an existing weight file. InternVLA-N1's auxiliary DepthAnything weights belong
inside its checkpoint directory (see [method environments](../docs/deployment.md#method-environments)).
A run with a missing checkpoint fails with a typed checkpoint-missing error —
it is never silently skipped.

## 4. Register once after downloading

```bash
python -m nav_eval configure \
  --method streamvln \
  --checkpoint checkpoints/streamvln \
  --method-python envs/nav_streamvln/bin/python \
  --simulator habitat024 \
  --environment-python envs/nav_streamvln/bin/python \
  --data-root data
```

The command creates or merges the ignored, permanent `configs/local.json`.
Multi-method registrations are keyed by plugin, so adding another method does
not replace existing method or simulator entries. `plan` and `run` load it
automatically. Existing resource maps are legacy import input only:

```bash
python -m nav_eval configure --from configs/resources/my-host.json
```

The first run records content digests of weights and assets; swapping weights
under an existing run is rejected.

## Split sizes

Full-split episode counts: R2R val_unseen 1,839 / val_seen 778; RxR val_unseen
guide 11,006 (multilingual subsets filter via `benchmark_settings.languages`);
ObjectNav MP3D val 2,195; VLNVerse fine 825 / coarse 835. Experiment configs
ship with small subsets — remove `episodes` / `episode_limit` for full runs.

## Licenses

| Data | License / access |
|---|---|
| R2R-CE / RxR-CE episodes | upstream releases derived from Matterport3D panoramas — MP3D terms apply |
| MP3D scenes | Matterport3D Terms of Use |
| ObjectNav MP3D episodes | CC BY-NC-SA 3.0 US (derived from Matterport3D) |
| HM3D scenes + episodes | Matterport research agreement |
| HSSD scenes | CC BY-NC 4.0 |
| hab3 episodes / Spot / humanoids | HF `ai-habitat/*` (episodes CC-BY-NC-4.0; humanoid motions under the SMPL body license; Spot ships its own license) |
| Model weights | their upstream repositories' licenses |

Downloads land in your untracked `data/` tree; their local paths stay out of
the repository through ignored `configs/local.json`. `asset_files()` records the episode
files, scenes, navmeshes and URDFs each run actually used — lock data
provenance alongside run records when comparing across hosts.
