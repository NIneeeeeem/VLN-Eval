# Benchmark Catalog and Setup

[简体中文](benchmarks.zh-CN.md) | English

This page lists every benchmark plugin shipped with the framework and how to
run it: where episodes and scenes come from, how to place them under `data/`,
what to register once in `configs/local.json`, and which experiment config to use.
Model-free environment checks live in the
[smoke recipes](../configs/benchmarks/README.md); dataset and scene download
sources in [Assets](../data/README.md).

All shipped benchmarks are continuously embodied (metric movement in Habitat
or Isaac); discrete viewpoint-graph navigation is out of scope. Every plugin
follows the shared contract: frozen episode selection, evidence capture per
step, offline metrics that read only committed evidence, and
`benchmark_settings` that enter the comparison key. Setup adds data and
configuration — the CLI and runner need no changes.

| Benchmark | Task family | Simulator bindings | Episodes source | Scenes |
|---|---|---|---|---|
| `r2r_ce` | continuous VLN | habitat024/030 | VLN-CE episodes | MP3D |
| `rxr_ce` | multilingual continuous VLN | habitat024/030 | RxR-CE episodes | MP3D |
| `vlnverse` | VLN (Isaac) | isaacsim500 | challenge repo | challenge assets |
| `objectnav` | ObjectNav (MP3D/HM3D) | habitat024/030 | habitat CDN | MP3D / HM3D |
| `instance_imagenav` | instance image-goal navigation | habitat030 | Habitat InstanceImageNav v2 | HM3D-Sem |
| `hm3d_ovon` (`ovon`) | open-vocabulary ObjectNav | habitat024 / habitat030 | OVON task-only port | HM3D-Sem |
| `goat_bench` (`goat`) | multimodal lifelong navigation | habitat024 / habitat030 | GOAT task-only port | HM3D-Sem |
| `multion` | sequential multi-object navigation | habitat024 / habitat030 | portable MP3D / HSSD task | MP3D / HSSD |
| `scand` | offline real-world SocialNav diagnostic | none | SCAND recordings | real robot data |
| `hssd_socialnav` | social navigation (habitat 3.0) | habitat030 | HuggingFace `ai-habitat` | HSSD (hab3) |

### Simulator bindings

| Binding | Runtime requirement | Benchmarks |
|---|---|---|
| `habitat030` | fresh `nav_habitat030`, stock Habitat 0.3.0 | all Habitat tasks; SocialNav requires this version |
| `habitat024` | fresh `nav_streamvln`, stock Habitat 0.2.4 | R2R, RxR, ObjectNav, InstanceImageNav, OVON, GOAT, MultiON |
| `isaacsim500` | Isaac Sim 5.0.0 pip environment `envs/nav_isaac` + downloaded task runtime (see [VLN-VERSE](#vln-verse-vlnverse)) | `vlnverse` |

A binding's `simulator_version` must match the packages importable by its
runtime `python`. `plugins check` / `plan` validate manifests and settings;
run the per-environment import checks in the deployment guide before a real
run.

## Continuous VLN (`r2r_ce`, `rxr_ce`)

Instruction-following navigation in VLN-CE form: discrete platform primitives
(forward 0.25 m, 15°/30° turns, RGB-D 640×480) in continuous MP3D scenes. One
binding serves all splits; `benchmark_settings.benchmark_id` selects the data:

| `benchmark_id` | Episode file under `<data-root>/datasets/` | Frozen selection |
|---|---|---|
| `r2r_val_unseen` | `r2r/val_unseen/val_unseen.json.gz` | 1,839 episodes / 11 scenes |
| `r2r_val_seen` | `r2r/val_seen/val_seen.json.gz` | 778 episodes / 53 scenes |
| `rxr_val_unseen` | `rxr/val_unseen/val_unseen_guide.json.gz` | 11,006 episodes / 11 scenes (en/hi/te) |

Use `"benchmark": "rxr_ce"` for RxR-CE and `"benchmark": "r2r_ce"` for
R2R-CE; both offer the habitat024/030 bindings. The `*_gt.json.gz`
companions are optional — scoring reads evidence captured by the binding.
Episode and scene sources, sizes and licenses are tabulated in
[data/README.md](../data/README.md); the single 91-scene MP3D set also covers
ObjectNav MP3D.

1. Verify the frozen selection without loading the simulator (also checks
   that every referenced scene exists on this host):

   ```bash
   python -B -c "
   from pathlib import Path
   from extensions.benchmarks.vln.r2r_ce.dataset import select_episodes
   root = Path('data').resolve()
   print(len(select_episodes(root / 'datasets/r2r/val_unseen/val_unseen.json.gz', root)))
   "
   ```

   A `missing scenes` error lists the absent MP3D scenes; copy them under
   `data/scene_datasets/mp3d/` or select an explicit `episodes` subset.

2. Register fresh `nav_streamvln` for `habitat024` and `nav_habitat030` for
   `habitat030`. Both cover all three splits; use the repository installer.

3. Run an existing experiment (e.g. `configs/experiments/navid-r2r.json`,
   `streamvln-r2r.json`) or copy one and change
   `benchmark_settings.benchmark_id` for another split. RxR may additionally
   filter English with `"languages": ["en-IN", "en-US"]` inside `benchmark_settings`.

## ObjectNav (`objectnav`)

Object-goal navigation following the official habitat ObjectNav protocol
([ObjectNav Revisited](https://arxiv.org/abs/2006.13171)). The goal is an
**object category** delivered through `EpisodeContext.goal`
(`kind: object_category`); language-only methods render their own prompt from
the category (NaVid does this — see below). Evidence records habitat's own
`DistanceToGoal` measure (view-point geodesic distance — the same quantity
upstream success/SPL use); the metric set `objectnav_standard` computes
SR / SPL / Soft-SPL / NE / Oracle-SR with formulas byte-identical to habitat
(Soft-SPL cross-checked against habitat's own measure on a real episode:
exact match to 1e-9).

Protocol defaults: `success_distance_m: 0.1` (upstream task default,
overridable in `benchmark_settings`), 500 control steps, forward 0.25 m,
turns 15° or 30°, RGB-D 640×480.

### Split A — ObjectNav MP3D v1 (no new scenes needed)

The 11 val scenes (`2azQ1b91cZZ`, `8194nk5LbLH`, `EU6Fwq7SyZv`, `QUCTc6BB5sX`,
`TbHJrupSAjP`, `X7HyMhZNoso`, `Z6MFQCViBuw`, `oLBMNvg9in8`, `pLe4wQe7qrG`,
`x8F5xyUWy9e`, `zsNo4HB9uLZ`) are a subset of the MP3D scenes any R2R setup
already has — if `r2r_ce` runs on this host, no scene download is required.

1. Download the official episode archive (one zip, train+val+val_mini; the
   path segment is `m3d`, not `mp3d`) — see the recipe in
   [Assets](../data/README.md#1-episode-datasets).

2. Verify the selection without loading the simulator (episode ids are
   positional over habitat's load order — main file, then `content/` in
   sorted scene order):

   ```bash
   python -B -c "
   from pathlib import Path
   from extensions.benchmarks.objectnav.objectnav.dataset import select_episodes
   root = Path('data').resolve()
   print(len(select_episodes(root / 'datasets/objectnav/mp3d/v1/val/val.json.gz', root)))
   "
   ```

3. Resource map: nothing new — the `habitat030` (or `habitat024`) runtime
   with `settings.data_root` already covers this benchmark.

4. Run (NaVid renders the category goal into its own fixed prompt, part of
   the method bundle identity):

   ```bash
   python -B -m nav_eval plan \
     --config configs/experiments/navid-objectnav.json
   python -B -m nav_eval run \
     --config configs/experiments/navid-objectnav.json \
     --output runs/objectnav-smoke
   python -B -m nav_eval evaluate --run runs/objectnav-smoke/<run-id>
   ```

   Remove `"episodes"`/`"episode_limit"` from the experiment for the full
   2,195-episode split. A model-free binding smoke:

   ```bash
   python -m nav_eval benchmarks smoke --config configs/benchmarks/objectnav.json \
     --output runs/benchmark-smoke/objectnav.json
   ```

### Split B — ObjectNav HM3D v2 (challenge edition)

1. Episodes (~248 MiB; layout `objectnav_hm3d_v2/{train,val,val_mini}`) —
   [habitat CDN](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip),
   extracted so that `<data-root>/datasets/objectnav/hm3d/v2/val/val.json.gz`
   exists.

2. HM3D scenes are license-gated by Matterport — see
   [Assets: HM3D](../data/README.md#2-scenes) for the token + downloader
   recipe. Use `hm3d_val_v0.2` for the full val scene set; scenes stay under
   `data/scene_datasets/hm3d/`.

3. Change the experiment to
   `"benchmark_settings": {"benchmark_id": "objectnav_hm3d_val"}` and run as
   above.

### Not integrated (yet)

- **Gibson ObjectNav**: no official episode download exists (the old habitat
  CDN path returns 403 and no habitat-lab release ever shipped the episodes);
  community mirrors use an unofficial `gibson/v1.1` layout and Gibson scenes
  need the StanfordVL agreement. To use it, place community episodes at
  `data/datasets/objectnav/gibson/v1.1/` and re-add the family mapping.
- **HSSD / ProcTHOR ObjectNav**: episodes are officially hosted on Dropbox
  (`objectnav_hssd-hab_v0.2.3.zip`), scenes on HuggingFace `hssd/hssd-hab`;
  integration is straightforward via a new `BENCHMARK_SPLITS` entry once the
  episodes have been fetched and their layout verified.

## InstanceImageNav (`instance_imagenav`)

The Habitat 0.3 binding targets the upstream
`benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml` task and its
two supported splits: `instance_imagenav_hm3d_val` and
`instance_imagenav_hm3d_val_mini`. It has a 1000-step budget, 0.25 m forward
actions, 30 degree turns and camera tilts. The public goal is
`image_reference`, backed by the rendered `instance_imagegoal` sensor. The
goal image retains native HxWx3 geometry and HFOV; goal camera pose, object
identity/category and world coordinates are evaluator-private.

Place the upstream episodes at
`<data-root>/datasets/instance_imagenav/hm3d/v2/<split>/<split>.json.gz` with
the sibling `content/*.json.gz` files, and licensed HM3D scenes below
`<data-root>/scene_datasets/`. Public episode IDs are scene-qualified
(`scene-relative-path::upstream-id`) because upstream IDs are scene-local.
Nondefault `content_scenes_path` metadata is rejected so Habitat's loader
behavior cannot silently change.

An image-capable method must declare both
`accepts_goals: ["image_reference"]` and `requires_sensors` containing
`instance_imagegoal`; plan resolution rejects missing declarations before
launch. The built-in language methods do not declare these, so they are not
offered as InstanceImageNav policies — bring an image-goal method via the
[integration guide](integration.md).

## Social navigation (`hssd_socialnav`)

Habitat 3.0's social navigation task
([paper](https://arxiv.org/abs/2310.13724), task type
`RearrangePddlSocialNavTask-v0`): a Spot robot finds and approaches a
simulated human wandering through an HSSD home. The binding drives the robot
with platform primitives converted 1:1 into `agent_0_base_velocity` control
steps (velocity = primitive displacement × `ctrl_freq`, clipped exactly like
habitat's `BaseVelAction`); the human is stepped by the upstream scripted
`OracleNavRandCoordAction` and is never visible to the method beyond RGB-D.

Two metric layers, by design:

- `socialnav_standard` (offline, evidence-only): position-based SR / SPL /
  Soft-SPL / NE / Oracle-SR on the captured robot→human `DistToGoal`
  samples — directly comparable with the other benchmarks in this repo.
- The evidence parity layer additionally records habitat's official
  `nav_seek_success`, `nav_to_pos_success`, `rot_dist_to_goal`, collision
  counters and the full `social_nav_stats` dict. The habitat 3.0 paper
  reports Finding Success / SPS / Following Rate / Collision Rate — read
  those numbers from the parity layer.

Protocol defaults: `success_distance_m: 1.5` (upstream `nav_to_pos_succ`
default), `max_task_steps: 750`, forward 0.25 m, 15° turns.

### Setup (all assets are HuggingFace `ai-habitat` datasets)

Run the downloader with the habitat-sim python of the `habitat030` environment
(`--username/--password` are HF credentials; all UIDs below are free but
license-acknowledged on the HF pages):

```bash
python -m habitat_sim.utils.datasets_download --username <hf-user> --password <hf-token> \
  --uids hab3-episodes --data-path <data-root>/          # val/social_rearrange.json.gz (~1.2k episodes)
python -m habitat_sim.utils.datasets_download --username <hf-user> --password <hf-token> \
  --uids hab3_bench_assets --data-path <data-root>/       # hab3-hssd scenes + benchmark configs
python -m habitat_sim.utils.datasets_download --username <hf-user> --password <hf-token> \
  --uids ycb --data-path <data-root>/                     # YCB object configs
python -m habitat_sim.utils.datasets_download --username <hf-user> --password <hf-token> \
  --uids hab_spot_arm --data-path <data-root>/            # robot URDF/meshes
python -m habitat_sim.utils.datasets_download --username <hf-user> --password <hf-token> \
  --uids habitat_humanoids --data-path <data-root>/       # female_2 humanoid (agent_1)
```

Then check the layout the binding expects (it resolves everything relative to
`data_root`):

```text
<data-root>/
├── datasets/hssd/rearrange/val/social_rearrange.json.gz
├── objects/{ycb,amazon_berkeley,google_object_dataset}/configs/
├── robots/hab_spot_arm/urdf/hab_spot_arm.urdf
├── humanoids/humanoid_data/female_2/female_2.urdf
└── scene_datasets/<hab3-hssd scenes>
```

(The `hab3-episodes` and `hab3_bench_assets` UIDs symlink/clone their trees
into `data/`; move or symlink the pieces into the layout above.)

Run:

```bash
python -B -m nav_eval run \
  --config configs/experiments/navid-socialnav.json
```

## VLN-VERSE (`vlnverse`)

Instruction-following navigation in 33 synthetic kujiale scenes rendered by
NVIDIA Isaac Sim (VLN-VERSE EMR challenge data): `vlnverse_fine_val_unseen`
(825 episodes) and `vlnverse_coarse_val_unseen` (835 episodes), served by the
`isaacsim500` binding. The metric set `vlnverse_standard` computes
SR / SPL / NE / OSR on the captured metric-space positions.

Prepare the local task runtime once with
`python -B scripts/setup_benchmark.py vlnverse`. No challenge checkout is needed.
All assets live under the registered `data_root`:

- Episodes: `<data-root>/datasets/vlnverse/{fine,coarse}_val_unseen.json.gz`
  (re-exported from the challenge repository).
- Scenes: `<data-root>/scene_data/vlnverse/<scan>/` (kujiale scene data).
- Robot USD: `<data-root>/Embodiments/` (vln-pe robot).

The `isaacsim500` runtime in `configs/local.json` must provide:

| Field | Value |
|---|---|
| `python` | `envs/nav_isaac/bin/python` (Python 3.11 / Isaac Sim 5.0.0) |
| `settings` | `data_root` |
| `cwd` | Omit; the binding uses its repository-local runtime |
| `env` | `ISAAC_PATH=<pip-installed isaacsim directory>`, `OMNI_KIT_ACCEPT_EULA=YES`, `PYTHONNOUSERSITE=1`; written by `scripts/register_isaac.py` |
| `pythonpath` / `library_paths` | Omit; the new prefix contains its own Python and CUDA dependencies |
| `timeout_s` / `min_free_memory_mib` | Generous: Kit startup is slow (1500 s / 12 GiB used on the reference host) |

Follow the [Isaac setup in the README](../README.md#specialized-benchmarks) for a fresh environment and a single-scene StreamVLN smoke. Start from `configs/experiments/navida-vlnverse.json` (`benchmark: vlnverse`,
`simulator: isaacsim500`) and select the split via
`benchmark_settings.benchmark_id`. This benchmark needs no MP3D assets.

Use `streamvln-vlnverse-diagnostic.json` or `internvla-n1-vlnverse-diagnostic.json`
in `configs/experiments/` for the fixed 12-route, three-scene diagnostic. These
use a horizontal initial camera (`camera_pitch_deg: 0`) and the USD's actual
90° HFOV; omitting camera pitch retains the USD's −30° mount. The default
controller is physical H1. Experimental `robot_flash: true` uses upstream
discrete flash movement with collision checks enabled, a separate protocol.

The R2R/RxR and VLNVerse bindings allow 500 navigation actions by default.
Camera probes consume only `max_control_steps`: its default is 5000 when
`allow_tilt` is enabled, otherwise `max_navigation_steps`. Explicit control
limits, including smoke limits, take precedence. VLNVerse also has the upstream
physics watchdog `max_task_steps`. Records retain legacy `steps`/`control_tick`
and add `navigation_steps`, `control_steps` and `environment_end_reason`.
See [measured results and protocol limits](vlnverse-validation.md).

## SCAND (`scand`): offline recorded-data evaluation

SCAND has no simulator binding in this framework. Its evaluator compares an
aligned predicted trajectory with a reference JSONL trajectory. Every row has
exactly `episode_id`, `timestamp_s`, `frame_id` and `position_xy_m`;
reference and prediction must contain identical episode IDs, timestamps and
coordinate frames. It reports equal-episode-weighted ADE/FDE.

```bash
python -B -m nav_eval benchmarks evaluate-offline scand \
  --reference data/scand/reference.jsonl --predictions runs/scand/predictions.jsonl \
  --output runs/scand/diagnostic.json

# In a ROS1 environment, export one explicitly selected nav_msgs/Odometry topic.
python -B -m nav_eval benchmarks export-scand \
  --bag recording.bag --topic /odom --episode-id trial-001 \
  --output runs/scand/trial-001.jsonl
```

The exporter uses each message's odometry header timestamp and does not
overwrite an existing output file. ADE/FDE measure imitation error only —
they are not closed-loop success or social-compliance metrics. The full
recorded-bag workflow is in the
[smoke recipes](../configs/benchmarks/README.md#scand-offline-check-without-a-simulator).

## Roadmap (researched, not integrated)

| Benchmark | Family | Status | Source of truth |
|---|---|---|---|
| ImageNav HM3D v3 | image-goal navigation | the shipped adapter targets InstanceImageNav HM3D v2; v3 needs loader/data parity plus an image-capable method | `…/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip` (habitat CDN) |
| REVERIE-CE / CVDN-CE / SOON-CE | VLN variants | no canonical official download; conversions (REVE-CE, RAL 2022) circulate through paper repos — provenance would need freezing first | REVE-CE paper repos |
| Arena-RoNav (Arena 5.0) | social navigation on Isaac | would follow the `vlnverse` isaac binding pattern; repo (`Arena-Rosnav/arena-rosnav`) needs a stable eval API audit | [5.arena-rosnav.org](https://5.arena-rosnav.org/) |
| ObjectNav HSSD / ProcTHOR | object navigation | Dropbox-hosted episodes; blocked on fetching and layout verification | habitat-lab `DATASETS.md` |

## Licenses and provenance

| Data | License/access |
|---|---|
| ObjectNav MP3D v1 episodes | MP3D Terms of Use + CC BY-NC-SA 3.0 US (derived from Matterport3D) |
| ObjectNav HM3D v2 episodes | distributed for use with HM3D; scenes gated by Matterport agreement |
| hab3 episodes / assets / Spot / humanoids | HF `ai-habitat/*` (CC-BY-NC-4.0 for episodes; humanoid motions additionally under the SMPL body license; Spot ships its own license) |
| HSSD scenes | HF `hssd/hssd-hab`, CC BY-NC 4.0 |

Downloads land in your untracked `data/` tree; the resource map keeps host
paths out of the repository. `asset_files()` records the episode files,
scenes, navmeshes and URDFs actually used by each run; identity digests cover
plugin code — lock data provenance separately when comparing across hosts.
