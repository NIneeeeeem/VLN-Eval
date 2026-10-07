# Plugin Integration Guide

[简体中文](integration.zh-CN.md) | English

All plugins are discovered automatically from built-in `extensions/**/manifest.json` or
from `plugin_dirs` in the experiment config; method bundles may additionally declare
close variants as `extensions/**/*.manifest.json`. There is no second central registry —
adding a plugin never requires touching the CLI, the runner or any central list.

```bash
python -B -m nav_eval plugins list
python -B -m nav_eval plugins inspect navida --kind method
python -B -m nav_eval plugins check navida --kind method
```

A manifest must contain `schema_version: nav-eval-plugin/1`, `kind`, `id` and a string
`version`. Duplicate IDs within the same kind are an error; external plugins cannot
silently override built-in ones.

Responsibility boundaries per plugin kind:

| Kind | Location | Responsibility |
|---|---|---|
| method | `extensions/methods/<id>/` | the full algorithm state: prompt, preprocessing, history, maps, action queue |
| simulator | `extensions/simulators/<id>/` | SDK lifecycle, native actions and geometry |
| benchmark | `extensions/benchmarks/<task>/<id>/` | data, task config, public observations, action translation, termination and evidence |
| metric | `extensions/metrics/<id>/` | offline scoring that only reads committed evidence |
| controller | `extensions/controllers/<id>/` or external `plugin_dirs` | stateless action conversion |

A benchmark declares its real task binding per simulator via `bindings[simulator_id]`;
registering a simulator alone does not make any benchmark available on it. Simulator-private
objects may be used inside an environment worker, but they must never enter method RPCs.

## Add a method

Create `extensions/methods/<id>/` with at least:

```text
extensions/methods/<id>/
├── manifest.json   # ID, entry point, sensors, actions, resource requirements
├── adapter.py      # create(config) factory
└── service.py      # model lifecycle and method implementation
```

Example manifest:

```json
{
  "schema_version": "nav-eval-plugin/1",
  "kind": "method",
  "id": "my_method",
  "version": "1",
  "entrypoint": "extensions.methods.my_method.adapter:create",
  "settings": [],
  "defaults": {},
  "capabilities": {
    "accepts_goals": ["language"],
    "requires_sensors": ["rgb"],
    "emits_actions": ["primitive", "stop"]
  },
  "requires": {
    "gpu": true,
    "paths": ["checkpoint"]
  },
  "validation": {"level": "unverified"}
}
```

`adapter.py` exposes `create(config)` and returns a service implementing
`call(operation, payload)`. Model dependencies must be imported lazily inside the factory
or `prepare()`, so that manifest discovery never loads Torch, Transformers or a simulator.

Keep method inference implementations inside the bundle's `runtime/` package,
using qualified or relative imports. Built-in methods do not accept `repo_path`
or modify `sys.path`; users only supply model assets and dependency environments.

Related methods may share pure helpers in `extensions/methods/shared/<family>/`,
while keeping model classes, registration, conversation templates and episode state
inside their own bundles. Existing runtime import paths can re-export shared functions.
Declare these directories with `"code_dependencies": ["../shared/image_io", "../shared/vila"]`.
Each entry must name a unique, existing direct child of the sibling `shared/` directory;
absolute paths, traversal and symlinks are rejected. Discovery reads metadata without
importing helpers. The declared directories and optional `shared/__init__.py` contribute
to the consumer's bundle digest, including provenance and notices. External Docker
bundles copy these dependencies alongside the bundle. Undeclared sibling families do
not affect that digest; run-resume still checks the project's global source identity.

`code_dependencies` declares fingerprints and file transport; Python module resolution
follows the plugin's existing import contract. Built-in helpers use qualified
`extensions.methods.shared.*` imports. External file entrypoints should use installed
qualified packages or explicit file loading rooted at `__file__`, for example
`runpy.run_path(str(Path(__file__).parent.parent / "shared/family/helper.py"))`.
That relative layout is preserved when Docker copies the external bundle. Declaring a
directory alone does not make an unqualified `shared.*` module importable.

Declare optional filesystem settings (for example `vision_tower` or `base_vlm`) in
`resource_paths`; do not duplicate them in `resource_settings`. Required paths belong in
`requires.paths`. `resource_settings` remains for legacy non-path deployment settings.
Host paths are frozen before a worker changes directories;
Docker paths remain container-relative. Declared method paths also participate
in resource hashing, so replacing an auxiliary model invalidates resume.
Declare additional files read implicitly by an upstream loader using
`resource_companions`, for example
`"resource_companions": {"checkpoint": ["../config.yaml", "../dataset_statistics.json"]}`.
Names are relative to a file resource's parent (or to a directory resource itself);
these exact files are required and hashed regardless of extension. YAML files
also participate in directory fingerprints. Unlisted external dependencies are
not covered automatically; preserve and declare them when adding a method.

Simulator manifests declare import probes with `requires.modules`; a benchmark
binding may add registration modules in its own `requires.modules`. Smoke checks
read these declarations instead of branching on plugin IDs. Set
`requires.main_thread: true` for simulators with a main-thread event loop: run them
through isolated workers, not inline smoke. GPU backends must set `requires.gpu`.

For optional actions, a binding can declare
`"action_requirements": {"camera_tilt": {"allow_tilt": true}}`. Planning freezes
the enabled action set from benchmark settings and uses it for both capability
negotiation and live validation. A method requiring tilt must explicitly enable
`benchmark_settings.allow_tilt`; otherwise planning fails before model loading.

Register an external runtime with its discovery directory, then retain that directory in
the experiment's `plugin_dirs`; `configure` records only the local interpreter/assets,
while `plugin_dirs` makes the bundle portable at plan and worker time:

```bash
python -B -m nav_eval configure --plugin-dir /opt/vendor/nav-plugin --method vendor-method \
  --method-path auxiliary_model=/models/vendor-aux
python -B -m nav_eval configure --plugin-dir /opt/vendor/nav-plugin --simulator vendor-sim
```

External bundles are also supported by `benchmarks smoke --plugin-dir <directory>` and
`matrix --plugin-dir <directory> --track native`. A frozen plan includes only the method
and simulator selected by that experiment, never unrelated local runtime inventory.

Method service operations:

- `describe`: declare role, sensors, actions and transition capability.
- `reset`: receive the public context and initialize episode state.
- `act`: receive a public observation; return episode, sequence and a non-empty action list.
- `observe_transition`: only if declared — feedback for every actually executed action.
- `close_episode`: clean up episode state; loaded weights may be kept.

The generic worker handles readiness, exclusive sessions, generations, public inputs and
action validation. Concrete methods must not copy the RPC server, nor add method-name
branches to `nav_eval/sdk` or the CLI.

Default `parallelism` works by replicating isolated processes and does not require the
method to implement a batch API. Each model copy stays resident across episodes, so
`reset/close_episode` must fully clean history, caches and pending action queues. New
methods following this contract adapt to replica parallelism automatically, without
runner changes.

The optional shared-model mode requires the manifest's `capabilities.batching` to declare
`"independent_greedy"`, and the service to provide
`act_batch(payloads) -> list[reply]`. It receives public observations from different
sessions and must return the same number of standard `act` replies in input order, each
keeping its own episode_id and observation_sequence. Single-sample `act` must remain
available.

Only model parameters may be shared; mutable state such as history, action queues and maps
must be isolated per session — one session's reset must never wipe another trajectory.
Batched calls are executed serially by the SDK and are mutually exclusive with
reset/transition/close; the SDK reuses the existing WorkerService boundary for per-session
validation and never allows two concurrent calls within one session. `independent_greedy`
does not support sampling strategies that depend on a global RNG sequence, nor does it
automatically guarantee token-by-token floating-point equivalence. When submitting an
adaptation, provide comparisons across different history lengths, pending actions, session
reuse, single failing input, and real closed-loop serial vs. batch runs. See
`extensions/methods/navida/service.py` for a reference implementation.

For batch sizes above 1, live `describe.batching` must be `independent_greedy`.
For singleton sharing it may instead be `independent_sessions`, provided RNG state is
also isolated per session. Declare settings required for multi-input batching with
`capabilities.batching_requires`, e.g. `{"decoding": "greedy"}`; planning validates these
against merged method defaults/settings. Shared mode defaults to max_batch_size=1.

Only close variants that share the same implementation and dependencies should share a
bundle. Keep the main `manifest.json` and add `<variant>.manifest.json` for the variant;
the factory selects the variant from `config["plugin"]["id"]`. Do not build new central
dispatch tables for merely similar methods.

## Add a simulator

Create `extensions/simulators/<id>/` with a manifest and a backend implementation:

```text
extensions/simulators/<id>/
├── manifest.json
└── backend.py
```

Example manifest (see `extensions/simulators/habitat030/manifest.json`):

```json
{
  "schema_version": "nav-eval-plugin/1",
  "kind": "simulator",
  "id": "my_sim",
  "version": "1",
  "entrypoint": "extensions.simulators.my_sim.backend:MyBackend",
  "defaults": {"version": "1.2.3"},
  "capabilities": {
    "offers_sensors": ["rgb", "depth"]
  },
  "requires": {
    "gpu": true,
    "paths": ["data_root"]
  },
  "validation": {"level": "unverified"}
}
```

Key points:

- `entrypoint` points at a Backend class (not a factory function) implementing
  `initialize(task_config) / reset(episode, seed) / step(native_action) / close()`.
- `offers_sensors` declares the sensors the simulator can provide; the sensor requirements
  of benchmark bindings are checked against it for compatibility.
- Keys in `requires.paths` (e.g. `data_root`) are given concrete paths by the host
  resource map.
- Simulator SDK dependencies must likewise be imported lazily; manifest discovery never
  loads Habitat/Isaac.
- Simulators with main-thread requirements (e.g. Isaac/InternUtopia) handle them via the
  service's `run_main()` — no CLI special-casing.

Reference implementations: `extensions/simulators/habitat/backend.py` (one backend serves
the 017/024/030 manifests, distinguished by `defaults.version`) and
`extensions/simulators/isaacsim500/backend.py` (a single Kit App that switches scenes via
reset).

A new simulator only participates in evaluation once a benchmark adds a binding for it;
see the next section.

## Add a benchmark

Create `extensions/benchmarks/<task>/<id>/` with a manifest, binding implementations and dataset
registration:

```text
extensions/benchmarks/<task>/<id>/
├── manifest.json
├── dataset.py           # dataset registration (if needed)
└── bindings/
    └── <simulator_id>.py
```

Choose a task family from the [benchmark extension guide](benchmark-extensions.md).
Keep benchmark IDs independent of the directory; shared helpers belong in `shared/`.

Example manifest (excerpt from `extensions/benchmarks/vln/r2r_ce/manifest.json`):

```json
{
  "schema_version": "nav-eval-plugin/1",
  "kind": "benchmark",
  "id": "my_benchmark",
  "version": "1",
  "default_simulator": "habitat030",
  "splits": ["my_val_unseen"],
  "defaults": {"benchmark_id": "my_val_unseen", "success_distance_m": 3},
  "settings": ["benchmark_id", "success_distance_m", "simulator_version"],
  "observation": {"width": 640, "height": 480, "hfov": 90},
  "bindings": {
    "habitat030": {
      "entrypoint": "extensions.benchmarks.vln.my_benchmark.bindings.habitat:create",
      "accepts_actions": ["primitive", "stop"],
      "capture_schema": "habitat-r2r-evidence/1",
      "provides_evidence": [
        "trajectory.goal_distances_m",
        "trajectory.positions_xyz_m",
        "reference.geodesic_start_to_goal_m",
        "reference.success_radius_m"
      ],
      "clock": "discrete_control_ticks",
      "action_semantics": {"forward_m": 0.25, "turn_rad": 0.2617993877991494},
      "settings": {"simulator_version": "0.3.0"}
    }
  },
  "metrics": ["r2r_ce_standard"],
  "validation": {"level": "unverified"}
}
```

Binding fields:

- `entrypoint`: points at a `create(config)` factory returning the benchmark service.
- `accepts_actions`: the action types this binding can actually execute; if a method's
  actions are incompatible, a controller must be selected explicitly — the runner never
  converts actions implicitly.
- `capture_schema` / `provides_evidence`: the evidence schema and fields captured;
  metrics match on `requires_evidence` and return unavailable for missing fields instead
  of degrading.
- `clock`: `discrete_control_ticks` or physical seconds; discrete tasks must not pass off
  action counts as physical seconds.
- `action_semantics`: action magnitude semantics (meters, radians); part of the comparison
  key.

The benchmark service provides `describe / episodes / reset / step / finish /
close_episode`. `finish` returns the execution record and evidence and does not compute
metrics during collection. `asset_files()` must list the data, scenes and NavMesh actually
used; `runtime_identity()` may record the SDK build.

Reference implementations: the Habitat binding in `extensions/benchmarks/vln/r2r_ce/bindings`,
the RxR dataset registration in `extensions/benchmarks/vln/rxr_ce/dataset.py`, and the Isaac
binding in `extensions/benchmarks/vln/vlnverse/bindings`.

## Add a metric

A metric manifest declares `requires_evidence` and a versioned `metric_set`:

```json
{
  "schema_version": "nav-eval-plugin/1",
  "kind": "metric",
  "id": "my_standard",
  "version": "1",
  "requires_evidence": [
    "trajectory.goal_distances_m",
    "reference.geodesic_start_to_goal_m"
  ],
  "metric_set": {
    "schema_version": "nav-eval-metric-set/0.1",
    "id": "my_standard",
    "evidence_schema": "habitat-r2r-evidence/1",
    "metrics": [{"entrypoint": "nav_eval.evaluation.metrics:MyMetric"}]
  }
}
```

Metrics read committed evidence only; missing fields return unavailable. A new metric can
re-score existing trajectories from the stored evidence. When geodesic distances are
unavailable, null and the reason are stored and metrics depending on that evidence return
unavailable rather than degrading to Euclidean scores.

Reference implementations: `extensions/metrics/r2r_ce_standard` and
`extensions/metrics/vlnverse_standard`.

## Add a controller

A controller manifest explicitly declares its input and output actions. When a method's
actions are incompatible with a binding, a controller must be selected; the runner does
not convert actions implicitly.

Controllers should be stateless pure action-transform functions; parallel replicas may
call them concurrently in the control process, so they must not keep episode history in
module globals.

## External bundles

The experiment config may provide absolute paths:

```json
{"plugin_dirs": ["third_party/plugins"]}
```

Relative `.py` entry points must not escape the bundle. Identity digests cover the `.py`
and `.json` files inside the bundle; models, data and other large resources must be locked
separately via the resource map and `asset_files()`.

## Acceptance checklist

`plugins check` only covers the first item; full acceptance includes at least:

1. Manifest and entry-point import (`plugins check`).
2. Public contracts: `describe` declarations match the manifest; sensor/action/evidence
   fields are complete.
3. Upstream behavior comparison on fixed inputs: behavior matches the original upstream
   implementation after integration.
4. Real closed-loop episodes: an isolated worker completes the full
   reset/act/step/finish flow.
5. Offline scorer comparison: evidence fields are correctly consumed by the metric set.

Paper-level reproduction additionally requires certified runs on the full split — record
that tier in the manifest's `validation.level`.
