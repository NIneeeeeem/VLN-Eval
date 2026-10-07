# Specialized Habitat benchmarks

[Benchmark setup](benchmarks.md) · [中文目录](benchmarks.zh-CN.md)

Use the [concrete smoke configurations and download recipes](../configs/benchmarks/README.md)
for a model-free environment check. They include versioned OVON/GOAT YAMLs,
matching cache splits, and both MultiON variants. The JSON below illustrates a
full method experiment; its placeholders are not needed for the smoke command.

The bindings below execute upstream Habitat tasks. They require separately
installed upstream code, licensed scenes, episodes, and task-specific assets
(setup recipes: [smoke recipes](../configs/benchmarks/README.md)).

中文说明：以下是已接入的专用评测适配器，需分别配置上游环境与数据。GOAT 图像目标
使用显式声明的上游特征；MultiON 原始 MP3D 版与 HSSD 版是两个独立绑定。这些任务
的目标类型与传感器要求各不相同——方法需按[接入指南](integration.md)显式声明能力。

| Benchmark | Simulator ID / runtime | Public goal | Required task sensors | Captured metric set |
|---|---|---|---|---|
| `hm3d_ovon` | `habitat024` / `habitat030` + task-only `ovon` | `language`: free-form category | `task_goal` | `upstream_habitat_standard`: success, SPL |
| `goat_bench` | `habitat024` / `habitat030` + task-only `goat_bench` | `goal_sequence` | `task_goal`, `goal_embedding` | `goat_captured`: composite/partial/modality success, composite SPL |
| `multion` (MP3D) | `habitat024` / `habitat030` + portable task | `goal_sequence`: object categories | `task_goal` | `multion_captured`: progress, PPL, success, MSPL |
| `multion` (HSSD) | `habitat024` / `habitat030` + portable task | `goal_sequence`: language | `task_goal` | `multion_captured` |

All values in these metric sets are explicitly named `captured_*`: they read
measurements committed by the selected upstream task. Missing/nonfinite fields
are unavailable, never silently treated as zero. GOAT preserves nested
subtask results in evidence. MP3D MultiON's `percentage_success`/`pspl` and the
HSSD fork's `progress`/`ppl` remain separate upstream protocols despite sharing
display names. Keep runs from different simulator bindings separate.

## Configuration

Choose an upstream checkout and interpreter for each runtime. Do not combine
the Habitat versions in one Python environment. Add the checkout to the worker
`pythonpath` and set the environment worker's `cwd` to that checkout, since its
configs reference relative scene/cache paths. The Nav-Eval worker requires
Python 3.9 or later. Use the fresh core prefixes and task-only recipes;
the original MultiON Habitat fork is replaced by the portable task plugin.
No dependencies, scenes, or weights are installed by
selecting a benchmark.

For example, use a host resource map with:

```json
{
  "runtimes": {
    "habitat024": {
      "python": "envs/nav_streamvln/bin/python",
      "pythonpath": ["<OVON_ROOT>"],
      "cwd": "<OVON_ROOT>"
    }
  }
}
```

An experiment for a separately registered, compatible method has this shape:

```json
{
  "benchmark": "hm3d_ovon",
  "simulator": "habitat024",
  "method": "your_ovon_policy",
  "track": "native",
  "episode_limit": 1,
  "benchmark_settings": {
    "repo_root": "<OVON_ROOT>",
    "task_config": "configs/<resolved-task>.yaml",
    "dataset_path": "data/datasets/ovon/val_seen/val_seen.json.gz",
    "asset_paths": ["ovon", "config", "data/scene_datasets", "data/caches"]
  }
}
```

`<OVON_ENV>` and `<OVON_ROOT>` are named placeholders, not executable paths.
The paths above are illustrative: every listed file/directory must exist in
your checkout. `task_config` and `dataset_path` are existing paths relative to
`repo_root`; the dataset path must name a concrete split file, not a template.
The task YAML must resolve the upstream dataset, task, actions, sensors, and
measurements. A baseline experiment containing a resolved `habitat` node is
also accepted by structured-config bindings. Missing execution fields fail
explicitly. Depth normalization is disabled so public depth is in meters.

`asset_paths` must enumerate additional dependencies used by your task: upstream
source/config directories, object assets, goal-feature caches, scene metadata,
and pre-rendered observations as applicable.
Files in these paths are hashed recursively. The selected main episode file,
standard `content/*.json.gz` shards, existing scene files, adjacent navmeshes,
and resolved task config are captured automatically. Custom dataset shard
layouts must be included explicitly. Omitting external assets leaves their
contents outside attestation — include them for reproducible comparison.

Episode IDs exposed by these bindings are `scene-relative-path::upstream-id`,
sorted deterministically before filtering. `episode_limit`/`episodes` at the
experiment level use the normal runner selection. Duplicate qualified IDs fail.
The simulator keeps its original episode IDs internally.

```bash
python -m nav_eval benchmarks list
python -m nav_eval benchmarks inspect GOAT-Bench
python -m nav_eval configure --from resources.json
python -m nav_eval plan --config experiment.json
python -m nav_eval run --config experiment.json --output runs/specialized
```

## Method contract and protocols

Every method must explicitly declare the accepted goal kinds in
`capabilities.accepts_goals` and consume the task sensors in
`capabilities.requires_sensors`. `plan` rejects mismatched declarations before
worker launch — a language-only VLN policy does not automatically qualify as
an image-goal or lifelong policy.

`task_goal` is a uint8 tensor containing UTF-8 JSON:

```json
{"schema":"nav-eval-task-goal/1","active_index":0,"count":3,"kind":"language","value":"find the blue lamp"}
```

For sequences, the reset goal describes the task-goal sensor and count. Only
the currently active target is public. A method's `stop` action means the
upstream subtask completion action (`subtask_stop` in GOAT, FOUND in MultiON).
The same method session remains active across target changes, preserving
memory. Completion is taken from the actual task index, not inferred from
every STOP being successful. In MultiON an incorrect FOUND may end the episode.
The generic rollout respects the binding's explicit episode decision budget.

### OVON and GOAT

[OVON](https://github.com/naokiyokoyama/ovon) uses its own registration package,
episodes, open-vocabulary categories and HM3D semantic scene assets. The binding
preserves free-form category text; it does not coerce labels into fixed
ObjectNav classes.

[GOAT's upstream sensors](https://github.com/Ram81/goat-bench/blob/main/goat_bench/task/sensors.py)
include `GoatGoalSensor` (`goat_subtask_goal`). Enable this sensor and its object,
language and image caches in the task config. Its finite float32 vector of
length 1024 is exported as `goal_embedding`, with source `estimated` and
representation metadata identifying the upstream sensor. This is **not an RGB
goal image**. Image subtasks reference this sensor using `embedding_reference`;
language and category subtasks also retain their public text in `task_goal`.
Privileged target instance IDs, viewpoints and evaluator distances stay private.

Defaults are 0.25m forward and 30-degree turns/tilts, 500 episode steps for
OVON and 5000 for GOAT. GOAT's adapter enforces the total episode budget; it
does not impose an additional per-subtask timeout. Camera defaults are OVON
640x480/90 degrees and GOAT 640x360/90 degrees — all explicit adapter
settings, overridable through the experiment config.

### MultiON

Select `habitat024` or `habitat030` for both the
[original MP3D task](https://github.com/saimwani/multiON) and
[HSSD challenge task](https://github.com/3dlg-hcvc/multion-challenge).
The portable plugin preserves ordered FOUND progression and termination;
MP3D cylinders use the public rigid-object manager, and the worker exposes
only the active category/language goal. It removes inherited PointNav oracle
sensors and unrelated measurements. MP3D uses the published 1.5m FOUND
threshold; HSSD uses 0.5m distance to navigable viewpoints. Both variants use
2500 steps, 0.25m/30 degrees and 256x256 RGB-D at 79-degree HFOV.

The source references historically used incompatible Habitat forks; these
task ports share the installed stock SDKs. OVON/GOAT still need task snapshots,
episode data, GOAT caches and licensed HM3D. MultiON needs real episodes and
scenes, plus MP3D cylinders. Download commands and dual-version examples are
in the [recipes](../configs/benchmarks/README.md).
