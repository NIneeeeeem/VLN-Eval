# Nav-Eval Plugin Architecture

[简体中文](architecture.zh-CN.md) | English

## The single execution pipeline

```text
experiment + permanent local installation
  -> manifest discovery
  -> resolve / preflight
  -> resident environment slots + isolated/shared model workers
  -> dynamic episode queue -> shared episode kernel
  -> committed attempts
  -> offline evaluation
```

`nav_eval.plugins.Registry` scans `extensions/**/manifest.json` plus any `plugin_dirs`
given explicitly by the experiment; method bundles may add `<variant>.manifest.json` for
close variants of a shared implementation. Discovery never imports model or simulator
dependencies; factories are only loaded once workers start.

| Location | Responsibility |
|---|---|
| `nav_eval/sdk` | common types, method lifecycle, worker facade, bounded batching, binary framing |
| `nav_eval/plugins` | manifest discovery, entry loading, bundle identity digests |
| `nav_eval/planning` | resolution of sensors, actions, bindings, evidence and resource configs |
| `nav_eval/execution` | launchers, total-resource preflight, resident replica pool, dynamic scheduling, run/resume |
| `nav_eval/storage` | atomic attempt commits, file locks, digests and derived JSONL |
| `nav_eval/rollout/generic.py` | the single episode action loop and error attribution |
| `nav_eval/evaluation` | standalone offline scoring and aggregation |
| `extensions/<kind>/<id>` | concrete plugin manifest, factory and implementation |

## Plugin boundaries

Plugin kinds are simulator, benchmark, method, metric and controller. A benchmark declares
its real task binding via `bindings[simulator_id]`; registering a simulator alone does not
make every benchmark available on it.

The simulator backend manages the SDK lifecycle, native actions and geometry. The
benchmark binding manages data, task configuration, public observations, action
translation, termination and evidence. Simulator-private objects may be used inside the
same environment worker, but they must not enter method RPCs.

A method bundle owns the complete algorithm state, including prompts, preprocessing,
history, maps, KV cache and action queues. The unified call chain is:

```text
WorkerService -> MethodBoundaryAdapter -> ServiceRuntime -> method service
```

`MethodBoundaryAdapter` lives in `nav_eval/sdk/method.py`; it only provides the common
contract and is not a method registry. Every reset creates a fresh generation; stale
replies, wrong episodes, missing transitions, private sensors and undeclared actions are
all rejected.

In shared mode, `sdk/batching.py` reuses this boundary chain per session, feeds the act
calls of different sessions into a bounded queue, and then calls the concrete method's
`act_batch`. Methods must declare the capability explicitly; reset, transition and close
are mutually exclusive with model computation. The SDK holds no method history or
implementation prompts and has no method-name branches.

## Protocol and authenticity

Experiments must declare `track: native | standardized`. Native uses the method's
observation defaults; standardized uses the benchmark defaults. Explicit observation
overrides enter the comparison key.

The plan freezes benchmark, simulator, binding, task settings, observations, controller,
metrics, seed and episode selection. Absolute paths live in the permanent `configs/local.json`;
execution results record digests of source, plugins, models and assets.

Discrete tasks use `control_tick` and never pass action counts off as physical seconds.
When geodesic distances are unavailable, null and the reason are stored, and metrics
depending on that evidence return unavailable. GT and geometry evidence only enters
evaluation artifacts and is never sent to the method.

Each plugin manifest's `validation.level` records what has been verified for it: an
importable entry point, a resolvable config, real episodes run, or certified full-split
runs.

## Deployment, storage and resume

Methods and environments use separate interpreters or containers pinned by digest. The
resource map specifies Python, GPU, environment variables and read-only paths per role;
prepare actually loads the model or scene.

`execution/pool.py` manages replica lifecycles; `runner.py` only dispatches one episode at
a time to each idle replica and commits results on a single control thread. `parallelism`
defaults to 1, and adding replicas does not change the intra-method inference order.
Replicas initialize concurrently and verify identical runtimes and assets before draining
the queue. Task durations of different episodes may overlap. By default each episode
occupies one pair of isolated processes; in shared mode each environment stays isolated
while several sessions share one model, the method's mutable state is kept per session,
and the scheduler coordinates failure restarts around the shared dependency.

Every episode attempt is first written to a temporary file, fsynced, then atomically
replaced. Policy errors are not retried; infrastructure errors are retried only under the
frozen configuration. Resume verifies the plan, source, plugins, weights, assets and the
episode list, then processes only unfinished items.

JSON/base64 is the reference transport; binary uses a versioned header and lossless
tensor buffers. Both go through the same worker contract and do not change model
preprocessing semantics.

For integration details see [integration.md](integration.md); for runtime resources and
resume see [deployment.md](deployment.md).
