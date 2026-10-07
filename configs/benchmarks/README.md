# Benchmark recipes and base runs

Run from the repository root. Scripted base smoke selects one episode,
renders RGB/depth, turns once, sends STOP/FOUND and computes metrics. Low
navigation scores are expected: this checks execution, not model quality.
Run it when setting up a new task.

## Environments and configurations

Use the [three fresh core prefixes](../../README.md): Habitat 0.2.4 in
`envs/nav_streamvln`, Habitat 0.3.0 in `envs/nav_habitat030`. OVON, GOAT and
both MultiON variants share them; no additional SDK fork is needed.

| Task | 0.2.4 configuration | 0.3.0 configuration | Assets |
|---|---|---|---|
| R2R | [r2r_ce_habitat024.json](r2r_ce_habitat024.json) | [r2r_ce.json](r2r_ce.json) | R2R val_unseen, MP3D |
| RxR | [rxr_ce_habitat024.json](rxr_ce_habitat024.json) | [rxr_ce.json](rxr_ce.json) | RxR guide val_unseen, MP3D |
| ObjectNav | [objectnav_habitat024.json](objectnav_habitat024.json) | [objectnav.json](objectnav.json) | ObjectNav v1, semantic MP3D |
| InstanceImageNav | [instance_imagenav_habitat024.json](instance_imagenav_habitat024.json) | [instance_imagenav.json](instance_imagenav.json) | ImageNav v2 val_mini, HM3D v0.2 |
| OVON | [hm3d_ovon.json](hm3d_ovon.json) | [hm3d_ovon_habitat030.json](hm3d_ovon_habitat030.json) | OVON val_seen, HM3D-Sem |
| GOAT | [goat_bench.json](goat_bench.json) | [goat_bench_habitat030.json](goat_bench_habitat030.json) | GOAT val_seen, HM3D-Sem, goal caches |
| MultiON MP3D | [multion_mp3d.json](multion_mp3d.json) | [multion_mp3d_habitat030.json](multion_mp3d_habitat030.json) | 3-ON val, MP3D, cylinders |
| MultiON HSSD | [multion_hssd_habitat024.json](multion_hssd_habitat024.json) | [multion_hssd_habitat030.json](multion_hssd_habitat030.json) | minival, complete HSSD scene |

HSSD SocialNav remains 0.3.0 only because it uses Habitat 3 humanoid and
rearrangement APIs. VLNVerse needs a separate `nav_isaac` prefix. SCAND
offline scoring needs no additional environment.

## Download public assets and task source

```bash
python -B scripts/setup_benchmark.py hm3d_ovon
python -B scripts/setup_benchmark.py goat_bench
python -B scripts/download_assets.py objectnav instance_imagenav hm3d_ovon \
  multion_mp3d multion_objects multion_hssd
python -B scripts/download_hssd_scene.py --scene 102816036
```

The asset downloader selects evaluation splits/content shards, resumes
partial downloads, preserves existing files, validates archives and records
URLs/SHA256 in `data/downloads/`. For public Hugging Face assets,
`HF_ENDPOINT=https://hf-mirror.com` is an optional endpoint; receipts record
it. R2R/RxR sources and licensed MP3D/HM3D instructions: [Assets](../../data/README.md).

The HSSD helper downloads actual scene/object references at revision
`4369cb9876214c7fbebcf552eb532380e4d287e4`. The example YAML selects only
scene `102816036`; remove `content_scenes` after downloading all minival
scenes. It is a complete scene, not a replacement scene or empty fixture.

GOAT uses the official Google Drive episode archive and split-matched caches:

```bash
python -B scripts/download_assets.py goat_bench
python -B scripts/download_goat_cache.py
```

The cache paths are explicit in [goat-habitat024.yaml](tasks/goat-habitat024.yaml)
and [goat-habitat030.yaml](tasks/goat-habitat030.yaml). Embeddings are feature
vectors, not RGB goal images. Unavailable downloads are reported as failures.

The task installer pins OVON/GOAT revisions, restricts package initializers
to environment registration and preserves originals in `.upstream`. It
avoids unrelated training-policy dependencies, installs immutable 024/030
task YAMLs and shares `data/`. Recipes and hashes are recorded in
`configs/environments/benchmark-runtimes.json` and `nav-eval-install.json`.
Use separate OVON/GOAT processes because registry names overlap. These
recipes do not support upstream policy training. `--archive` accepts an
already downloaded official source snapshot.

MultiON uses the portable repository task; never add an upstream Habitat
fork to `PYTHONPATH`. MP3D cylinders use the public rigid-object manager.
STOP maps to FOUND: wrong FOUND ends the episode; correct FOUND advances the
sequence; finding the last goal ends it. Both variants use 0.25m movement,
30-degree turns and 256x256 RGB-D at 79-degree HFOV. HSSD uses the challenge's
0.5m threshold.

## Run a base smoke

```bash
CUDA_VISIBLE_DEVICES=0 envs/nav_streamvln/bin/python -B -m nav_eval benchmarks smoke \
  --config configs/benchmarks/multion_mp3d.json --output runs/base/multion_mp3d-024.json
CUDA_VISIBLE_DEVICES=0 envs/nav_habitat030/bin/python -B -m nav_eval benchmarks smoke \
  --config configs/benchmarks/multion_hssd_habitat030.json --output runs/base/multion_hssd-030.json
```

Choose a configuration from the table. `passed` verifies a real episode,
observations, termination and metrics; `blocked` lists missing assets or
modules. `--check-only` discovers paths/modules without rendering.
[Model baseline commands](../../README.md).

## Other backends

VLNVerse uses `python -B scripts/setup_benchmark.py vlnverse`, Isaac Sim
5.0 + InternUtopia and isolated `nav_eval run` workers, not inline smoke.
[VLNVerse setup](../../docs/benchmarks.md#vln-verse-vlnverse).
HSSD SocialNav needs additional humanoid/episode assets:
[SocialNav setup](../../docs/benchmarks.md#social-navigation-hssd_socialnav).

SCAND's synthetic fixture checks offline evaluator wiring (ADE 0.25m, FDE 0.5m):

```bash
python -B -m nav_eval benchmarks evaluate-offline scand \
  --reference configs/benchmarks/scand-reference.jsonl \
  --predictions configs/benchmarks/scand-predictions.jsonl --output runs/base/scand.json
```

Real [SCAND recordings](https://www.cs.utexas.edu/~xiao/SCAND/) go in `data/scand/`.
Use the documented ROS1 exporter or supply JSONL with matching episode IDs,
odometry header timestamps and coordinate frames.
