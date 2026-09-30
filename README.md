# Nav-Eval

A unified evaluation platform for vision-language navigation (VLN) methods on Linux.
Methods, simulators, benchmarks and metrics are all attached as manifest plugins; the
control plane only handles config resolution, dependency isolation, execution, resume and
offline scoring — it contains no model or simulator implementation of its own.

🌐 [简体中文](README.zh-CN.md) | English

📚 [Documentation](docs/README.md) | 🧭 [Methods](#supported-methods) | 🌍 [Simulators & Benchmarks](#simulators--benchmarks) | ⚡ [Quickstart](#quickstart)

---

## Why Nav-Eval?

The pain point of VLN evaluation is coupling: every paper ships its own scripts, a pinned
simulator version, and incomparable measurement conventions — results stop reproducing the
moment you switch machines. Nav-Eval separates the evaluation pipeline from the thing
being evaluated:

- **Reproducible** — the plan freezes benchmark, simulator, binding, observations, metrics
  and seed; source, plugins, weights and assets are locked by content digests; episodes
  commit atomically and runs resume from episode boundaries without re-running inference.
- **Isolated** — methods and simulators run in their own Python environments and processes;
  a method only receives the public observations declared in its manifest; shape, dtype,
  units and actions are validated at the inference boundary, and ground-truth evidence
  never leaks to the model.
- **Honestly graded** — "entry point imports", "episodes run end to end" and
  "paper-level reproduction" are distinct validation levels; results carry their
  diagnostic/unverified labels explicitly and never stand in for one another.

## Quickstart

The control plane depends only on the standard library (Python >= 3.9); models and
simulators use their own upstream environments:

```bash
git clone https://github.com/NIneeeeeem/VLN-Eval.git && cd VLN-Eval

# 1. List registered plugins
python -B -m nav_eval plugins list

# 2. Create a host resource map from the template (interpreters, GPUs, weights, data)
cp configs/resources/example.json configs/resources/my-host.json

# 3. Validate the plan (does not start models or simulators)
python -B -m nav_eval plan \
  --config configs/experiments/navida-r2r.json \
  --resources configs/resources/my-host.json

# 4. Collect + score offline
python -B -m nav_eval run \
  --config configs/experiments/navida-r2r.json \
  --resources configs/resources/my-host.json
python -B -m nav_eval evaluate --run runs/<run-id>
```

If `plan` succeeds, plugin discovery and resource planning are correct; real runs
additionally need the resource file to point at valid weights, data and upstream
environments.

## Bash inference and evaluation

Use two shared entrypoints for all model/dataset pairs:

```bash
# Fill in interpreters, GPUs, checkpoint/source and data paths before running.
cp configs/resources/example.json configs/resources/my-host.json
# Plan only: no model loading or simulation (not a runtime/asset check).
MODE=plan bash scripts/inference.sh navida r2r configs/resources/my-host.json

# Run ONE desired pair, using a resource map appropriate to that pair:
bash scripts/inference.sh streamvln r2r configs/resources/my-host.json
bash scripts/inference.sh streamvln vlnverse configs/resources/my-host.json
bash scripts/inference.sh navida r2r configs/resources/my-host.json
bash scripts/inference.sh navida vlnverse configs/resources/my-host.json

# Only after integrating an external InternVLA-N1 method plugin:
CONFIG=/absolute/path/internvla-r2r.json bash scripts/inference.sh internvla-n1 r2r configs/resources/my-host.json
CONFIG=/absolute/path/internvla-vlnverse.json bash scripts/inference.sh internvla-n1 vlnverse configs/resources/my-host.json

# Use the actual run directory printed as output_dir, including <run-id>.
bash scripts/eval.sh runs/navida-r2r/<run-id>
python -B -m nav_eval resume --run runs/navida-r2r/<run-id>
```

| Model | R2R-CE val_unseen | VLNVerse fine val_unseen |
|---|---|---|
| `streamvln` | `habitat024`, native track | `isaacsim500`, standardized track; runtime unverified |
| `navida` | `habitat030`, native track | `isaacsim500`, standardized track |
| `internvla-n1` | External plugin required | External plugin and compatible Isaac observations/actions required |

`inference.sh MODEL DATASET RESOURCES [OUTPUT_PARENT]` uses
`configs/experiments/MODEL-DATASET-full.json`. These configs omit episode filters/limits;
they retain the `diagnostic` claim and do not certify paper reproduction. Existing
small-sample configs are unchanged. InternVLA-N1 is **not implemented in this repository**:
its two configs are integration templates, not runnable support. Without a registered
plugin the script exits explicitly. An external config must register method ID
`internvla-n1` through `plugin_dirs` and declare compatible sensors/actions (and a
controller if needed); see [plugin integration](docs/integration.md).

Usage rules:

- Set `PYTHON=/path/to/python` for the control plane (default `python`); model/simulator
  interpreters, GPUs and assets come from the resource JSON. `example.json` is a
  NaVIDA/Habitat template, not a ready-made map for every pair. Adapt both roles, or key
  `runtimes` by method/simulator ID in your own host map. Isaac additionally needs
  `challenge_repo`.
- `MODE=plan` validates configuration only; `MODE=run` (default) collects trajectories
  **and automatically scores them**. Default output is `runs/MODEL-DATASET/<run-id>`;
  the optional fourth argument changes the parent, not the generated run ID.
- `CONFIG=/path/to/experiment.json` selects a custom experiment, for example a smoke
  test (`episode_limit: 1`), specific episodes, parallelism or VLNVerse coarse
  (`benchmark_settings.benchmark_id: vlnverse_coarse_val_unseen`). Its method/benchmark
  must match the command. R2R uses benchmark ID `r2r_ce` internally.
- Shell path arguments and `CONFIG` are relative to the caller's working directory;
  paths **inside JSON** should be absolute (relative ones resolve from the repository).
- `eval.sh RUN_DIR` re-scores frozen trajectories without inference or GPU allocation;
  metrics default to the run's frozen metric set. Optional `--metric-set FILE` and
  `--plugin-dir DIR` are forwarded. Results go to `RUN_DIR/evaluations/<evaluation-id>/`
  (`summary.json`, `episodes.jsonl`); collection artifacts are preserved. Resume must
  use `nav_eval resume`, not a fresh invocation of `inference.sh`.

Script inventory:

| Script | Purpose |
|---|---|
| `inference.sh`, `eval.sh` | Public collection and offline scoring entrypoints |
| `compare_inference.py` | Compares complete runs: trajectories, resource locks and timing; rejects invalid speed comparisons |
| `check_preprocessing_rollout.py` | Prepares a preprocessing reference plugin (`preprocessing_reference.py`) and compares its rollout against the real method |

Diagnostic audit scripts from earlier development phases are not part of inference or
evaluation and are not shipped with the repository.

## Installation

```bash
# Control plane: the standard library is enough; an editable install also works
pip install -e .

# Methods and simulators: there is no unified install
```

Each real method/simulator keeps its own upstream dependency environment (Torch,
Transformers, Habitat, Isaac, ...) and you assign interpreters, GPUs and paths per role in
the resource map; the control plane does not install them for you. See the
[deployment guide](docs/deployment.md) (including Docker) for details. Where to obtain
every method checkpoint, dataset and upstream environment is documented in
[data/README.md](data/README.md).

## Usage Examples

> More examples in [`configs/experiments/`](configs/experiments) and
> [`configs/resources/`](configs/resources).

### Serial evaluation

```bash
python -B -m nav_eval run \
  --config configs/experiments/navida-r2r.json \
  --resources configs/resources/my-host.json
```

### Parallel inference (replica data parallelism)

Set `parallelism` in the experiment config and assign GPUs per replica with `replicas` in
the resource file:

```bash
python -B -m nav_eval run \
  --config configs/experiments/navida-r2r-parallel.json \
  --resources configs/resources/my-host.json
```

All replicas share one dynamic episode queue; each replica still executes strictly in
order internally, so methods do not need to become thread-safe. Multi-GPU template:
[parallel.example.json](configs/resources/parallel.example.json).

### Shared model with batching

Adapted methods (currently only NaVIDA) can serve multiple environments from one set of
weights and overlap environment execution with inference:

```bash
python -B -m nav_eval run \
  --config configs/experiments/navida-r2r-shared.json \
  --resources configs/resources/my-host.json
```

This example preserves upstream decoding, uses one input per generation, and isolates
CPU/CUDA random state per session. Larger tensor batches require explicit greedy decoding
and a separate trajectory-equivalence check; they are not enabled by default.

### Resume and re-scoring

```bash
python -B -m nav_eval resume --run runs/<run-id>      # episode-level resume
python -B -m nav_eval evaluate --run runs/<run-id>    # offline re-scoring of existing trajectories
```

For all options: `python -B -m nav_eval --help`.

## Supported Methods

| Method ID | Implementation | Runtime resources | Current status |
|---|---|---|---|
| `navida` | `extensions/methods/navida` | checkpoint, GPU | JPEG cache parity checked on two R2R trajectories; full-split reproduction unverified |
| `awarevln` | `extensions/methods/awarevln` | checkpoint, upstream source, GPU | preprocessing parity checked on two R2R trajectories; full-split reproduction unverified |
| `streamvln` | `extensions/methods/streamvln` | checkpoint, upstream source, GPU | R2R/Habitat 0.2.4 runs enabled; upstream trajectory parity checked on two episodes; full-split reproduction unverified |
| `navila` | `extensions/methods/navila` | checkpoint, upstream source, GPU | preprocessing parity checked on two R2R trajectories; paper reproduction unverified |
| `navid` | `extensions/methods/navid` | checkpoint, upstream source, EVA vision tower, GPU | corrected 30° turns; two R2R trajectories completed; paper reproduction unverified |
| `uni_navid` | `extensions/methods/navid` (variant manifest) | same as NaVid | official online cache, 30° turns and HFOV 120; two R2R trajectories completed; paper reproduction unverified |

Methods without a bound implementation, synthetic strategies and test fixtures are not
listed as supported. `plugins check` only proves that the manifest and Python entry point
load — not that weights, simulation assets or paper-level results are usable.

## Simulators & Benchmarks

### Simulators

| Simulator ID | Implementation | Upstream version | Sensors offered |
|---|---|---|---|
| `habitat017` / `habitat024` / `habitat030` | `extensions/simulators/habitat` | Habitat-Lab 0.1.7 / 0.2.4 / 0.3.0 | rgb, depth |
| `isaacsim500` | `extensions/simulators/isaacsim500` | Isaac Sim 5.0 | rgb |

### Benchmarks

| Benchmark | Data | Available bindings | Metric set |
|---|---|---|---|
| `r2r_ce` | R2R val_seen / val_unseen | `habitat017` / `habitat024` / `habitat030` | `r2r_ce_standard` (SR, SPL, NE, OSR) |
| `rxr_ce` | RxR val_unseen | `habitat024` / `habitat030` | `r2r_ce_standard` |
| `vlnverse` | VLNVerse fine/coarse val_unseen | `isaacsim500` | `vlnverse_standard` |

A benchmark declares its real task binding per simulator via `bindings[simulator_id]`;
registering a simulator alone does not make any benchmark available on it.

## Project Organization

```text
Nav-Eval/
├── nav_eval/            # control plane (stdlib only)
│   ├── plugins/         #   manifest discovery, entry loading, bundle identity digests
│   ├── planning/        #   sensor/action/binding/evidence/resource config resolution
│   ├── execution/       #   launchers, preflight, replica pool, dynamic scheduling, run/resume
│   ├── rollout/         #   the single episode action loop
│   ├── evaluation/      #   standalone offline scoring and aggregation
│   ├── storage/         #   atomic attempt commits, file locks, derived JSONL
│   └── sdk/             #   shared cross-method contracts and worker boundary (no concrete methods)
├── extensions/          # plugin bundles, one directory per kind
│   ├── methods/         #   navida, navid, navila, awarevln, streamvln
│   ├── simulators/      #   habitat (017/024/030), isaacsim500
│   ├── benchmarks/      #   r2r_ce, rxr_ce, vlnverse (bindings live in each bindings/)
│   └── metrics/         #   r2r_ce_standard, vlnverse_standard
├── configs/
│   ├── experiments/     # experiment configs: benchmark + simulator + method + track + seed
│   └── resources/       # host resource maps: interpreters, GPUs, weights, data paths
├── scripts/             # inference.sh, eval.sh and comparison tools
├── docs/                # project documentation
└── runs/                # run artifacts: attempts, trajectories, evidence, scores
```

Example experiment config:

```json
{
  "schema_version": "nav-eval-experiment/1",
  "benchmark": "r2r_ce", "simulator": "habitat030", "method": "navida",
  "track": "native", "launcher": "python", "transport": "binary",
  "benchmark_settings": {"benchmark_id": "r2r_val_unseen"},
  "episode_limit": 1, "seed": 0, "claim": "diagnostic"
}
```

## Adding Plugins

All plugins are discovered automatically from `extensions/**/manifest.json` or from
`plugin_dirs` in the experiment config — there is no second central registry:

| Kind | Location | Guide |
|---|---|---|
| Method (model) | `extensions/methods/<id>/` | [Add a method](docs/integration.md#add-a-method) |
| Simulator | `extensions/simulators/<id>/` | [Add a simulator](docs/integration.md#add-a-simulator) |
| Benchmark | `extensions/benchmarks/<id>/` | [Add a benchmark](docs/integration.md#add-a-benchmark) |
| Benchmark-simulator binding | `bindings/` inside the benchmark bundle | same guide |
| Metric | `extensions/metrics/<id>/` | [Add a metric](docs/integration.md#add-a-metric) |
| Controller | `extensions/controllers/<id>/` or external `plugin_dirs` | [Add a controller](docs/integration.md#add-a-controller) |

Minimal method bundle:

```text
extensions/methods/<id>/
├── manifest.json   # ID, entry point, sensors, actions, resource requirements
├── adapter.py      # create(config) factory
└── service.py      # model lifecycle and method implementation
```

`nav_eval.plugins.Registry` discovers manifests automatically; you never touch the CLI,
the runner or any central list. For the full guide (manifest fields, service contracts and
the acceptance checklist) see [plugin integration](docs/integration.md).

## FAQ

**Which methods does Nav-Eval support?**
Six built-in method IDs (see the table above) covering upstream implementations such as
NaVid/NaVILA/NaVIDA/AwareVLN/StreamVLN. Any method satisfying the service contract can be
attached via a bundle or external `plugin_dirs`.

**Which simulators and benchmarks are supported?**
Habitat-Lab (0.1.7/0.2.4/0.3.0) and Isaac Sim 5.0; R2R-CE, RxR-CE and VLNVerse. Adding
new simulators/benchmarks is covered in the [integration guide](docs/integration.md).

**What does a passing `plugins check` prove?**
Only that the manifest and Python entry point load — not that weights, simulation assets
or paper-level results are usable. The acceptance checklist is in the
[integration guide](docs/integration.md).

**How do I run on a new machine?**
Create a host resource map from `configs/resources/example.json` and fill in interpreters,
GPUs, weights and data paths. Missing resources, bindings, sensors or action compatibility
fail explicitly at plan/preflight time; the platform never silently falls back to a
synthetic implementation.

**How do I go parallel / multi-GPU?**
Set `parallelism` in the experiment and assign GPUs per replica via `replicas` in the
resource file; model and rendering can sit on different cards. Resource budgeting, resume
and throughput conventions are in the [deployment guide](docs/deployment.md#parallel-inference).

**What if an evaluation is interrupted?**
`resume` recovers from episode boundaries: after verifying that plan, source, plugins,
weights and assets match, it only processes unfinished items; completed attempts are never
re-run. `evaluate` can re-score offline at any time.

## Documentation

| Doc | Contents |
|---|---|
| [Plugin integration](docs/integration.md) | Full guides for adding methods/simulators/benchmarks/metrics/controllers |
| [Architecture](docs/architecture.md) | Execution pipeline, plugin boundaries, protocol and authenticity |
| [Deployment](docs/deployment.md) | Resource maps, parallel inference, shared models, Docker, resume and throughput |
| [Protocol](docs/protocol.md) | Worker sessions, public observations, action and failure semantics, transport |

Documentation is also available in [简体中文](README.zh-CN.md#文档).

## License

Released under the [Apache License 2.0](LICENSE).

## Acknowledgement

The evaluation framework design references
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
