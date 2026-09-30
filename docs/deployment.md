# Linux Deployment and Inference

[简体中文](deployment.zh-CN.md) | English

Experiments and machine resources are configured separately. Existing Python/Conda
environments can be used directly; the Docker launcher requires access to the daemon plus
a pre-built and verified image digest. This repository does not publish real model or
simulation images.

## Control-plane verification

```bash
python -m nav_eval plugins list
python -m nav_eval doctor
python -m nav_eval plan \
  --config configs/experiments/navida-r2r.json \
  --resources configs/resources/my-host.json
```

These commands only verify plugin discovery, entry points and resource planning; they do
not start models or simulators. Real methods keep their upstream dependency environments,
checked during the preflight/prepare stages of `run`.

## Real runs

```bash
python -m nav_eval plan --config configs/experiments/navida-r2r.json --resources configs/resources/my-host.json
python -m nav_eval run --config configs/experiments/navida-r2r.json --resources configs/resources/my-host.json
python -m nav_eval run --config configs/experiments/navida-vlnverse.json --resources configs/resources/my-host.json
python -m nav_eval resume --run runs/<run-id>
python -m nav_eval evaluate --run runs/<run-id>
```

Machine-specific resource maps (`*.this-host.json`) are not tracked in the repository.
On any machine, build your own resource file from `configs/resources/example.json`.
Before running, check the free VRAM of the physical
GPUs listed there; GPU allocation only affects this run's workers and does not stop other
processes.

Resources can be placed under `runtimes` keyed by plugin ID, or overridden with
`method/environment`. Each role may specify python, gpu, pythonpath, library_paths, cwd,
env, timeout_s, min_free_memory_mib, and settings such as checkpoint/repo_path/data_root/
challenge_repo. Resource settings only accept the manifest's requires.paths/
resource_settings; experiment parameters such as cameras, actions and success thresholds
must go into the experiment config — they cannot bypass comparison keys and compatibility
checks through the local map. Model and rendering processes may use different GPUs;
physical cards are mapped to in-process 0 via CUDA_VISIBLE_DEVICES.

Preflight checks paths, interpreters, GPUs and disk. Model loading in isolated processes
and environment initialization run concurrently; episodes only start after all replicas
pass prepare and identity verification. Per-replica logs go to
`method.log/environment.log` and failure summaries to `failure.json`. The Python launcher
cleans up separate process groups, including subprocesses started by interpreter scripts.
Models stay resident across episodes; method history and caches are cleaned per the
upstream reset rules.

## StreamVLN on R2R

A host resource map binds the StreamVLN v1-3 checkpoint and Habitat 0.2.4:

```bash
python -B -m nav_eval run \
  --config configs/experiments/streamvln-r2r.json \
  --resources configs/resources/my-host.json
```

This is a diagnostic subset of 33 distinct routes, three from each of the 11
val_unseen scenes. Remove `episodes` to evaluate the full 1839-episode split.
The reference host map selected GPU 4 and checked a combined 32000 MiB
model/rendering budget. A long trajectory reached 27258 MiB (26.6 GiB) for the
model process, so this budget was raised after the initial experiment; short
smokes underestimate capacity. Wait for sufficient memory or select another GPU.
Update paths and device allocation on another host.

Defaults are `preprocess_mode: lazy` and `tokenizer_mode: reuse`: retain every raw
RGB observation, preprocess only selected current/history frames, and reuse a
dedicated prompt tokenizer copy while executing upstream prompt construction.
Model precision, generation settings and random conjunction draws are preserved.
Use `eager` and `upstream` for the baseline. The paired smoke configurations are
`streamvln-r2r-smoke.json` and `streamvln-r2r-lazy-validation.json`.

```bash
python -B scripts/compare_inference.py BASE_RUN OPTIMIZED_RUN \
  --streamvln-preprocessing --output comparison.json
```

This exception allows only those two preprocessing settings to differ; actions,
private evidence, metrics and identity locks must still match. StreamVLN supports
independent replicas, but does not declare shared sessions or batched generation.

## Parallel inference

Add `"parallelism": 2` to the experiment config and two explicit assignments to the
resource file:

```json
{
  "replicas": [
    {"method": {"gpu": 0}, "environment": {"gpu": 0}},
    {"method": {"gpu": 1}, "environment": {"gpu": 1}}
  ]
}
```

These entries merge with the existing `runtimes` and `method/environment` resource
configuration; they do not replace weight and interpreter settings on their own. You can
edit [parallel.example.json](../configs/resources/parallel.example.json) directly. The
number of `replicas` must equal `parallelism`, and every GPU-requiring role must spell
out its device. A replica's method and environment may sit on different cards, and
multiple replicas may explicitly share one card. Replicas may only override deployment
fields — not checkpoints, method settings or benchmark settings.

The default `inference.mode: replicas` gives each replica its own method/environment
process pair, with the resident model loaded exactly once. All replicas share one episode
queue and pick up the next task upon completing the current one; long trajectories never
hold other idle replicas waiting. Each worker still holds exactly one session, and within
an episode the observe/reason/act/feedback order is strict. That is why NaVid's upstream
global history or StreamVLN's caches, for example, do not need to become thread-safe.
Each weight copy occupies its own VRAM in this mode; sharing weights requires the
explicit configuration below.

GPU roles must configure a positive `min_free_memory_mib`, the conservative VRAM budget
of that worker. Preflight sums the budgets of all model and rendering processes by
physical GPU UUID and checks that the total fits into available VRAM. Numeric indices and
UUIDs pointing at the same card are merged. This is a pre-launch capacity check, not a
system-level VRAM reservation — leave headroom for the longest history and other jobs.
The benefit of co-locating replicas on one card depends on model size and GPU load; do
not assume doubling concurrency doubles speed.

An example (replica allocations come from your host map, built from
[parallel.example.json](../configs/resources/parallel.example.json)):

```bash
python -B -m nav_eval run \
  --config configs/experiments/navida-r2r-parallel.json \
  --resources configs/resources/my-host.json
```

That example pins GPU 7, two replicas and 4 diagnostic episodes. Before scaling to a full
split, confirm all scenes exist and adjust the episode selection. Other integrated
methods use the same `parallelism/replicas` configuration by swapping the base resource
map; per-method numerical equivalence under real parallelism still has to be verified.

Parallel logs land in `workers/000/`, `workers/001/`, etc.; `replicas.lock.json` locks
each replica's runtime and assets. Before execution, all replicas must agree on identity
and the episode list; restarts affect only the worker pairs that hit infrastructure errors
or failed cleanup, and identity is re-checked. The scheduler commits attempts centrally;
results, scoring and resume stay within one run directory. Ctrl-C reaps the process groups
started by this run; committed attempts are kept, and uncommitted tasks resume from
episode boundaries. The `local` launcher stays single-replica to avoid mixing RNG and
simulation threads.

## Shared model and batching

Adapted methods can serve multiple independent environments from one model; among the
production methods only NaVIDA declares this capability today. Add the following to the
experiment; resources still reuse `replicas` to pin the devices of each environment and
the shared model:

```json
{
  "parallelism": 2,
  "inference": {"mode": "shared", "max_batch_size": 1, "max_wait_ms": 0}
}
```

```bash
python -B -m nav_eval run \
  --config configs/experiments/navida-r2r-shared.json \
  --resources configs/resources/my-host.json
```

The deployment resources merged from all `replicas[].method` must be identical; the model
is loaded once and its VRAM budget counted once, while environment budgets accumulate.
In this mode `parallelism` caps environments/sessions; `max_batch_size` caps the number
of merged requests per batch, defaults to 1, and must lie between 1 and parallelism.
The default overlaps environments with singleton inference. Values above 1 enable actual
tensor batching and require a separate numerical-equivalence check. `max_wait_ms` is the
upper bound for waiting for more requests once the first one is ready, defaulting to
5 ms; the last episode need not wait for other environments. Queueing and GPU execution
may still take longer — it is not an RPC latency cap.

Only the current requests of different episodes are merged; within an episode the loop
still waits for execution and new observations. History, action queues, generations and
observation sequences are isolated per session. NaVIDA's multi-input uses left padding
and keeps the original JPEG, frame sampling and action parsing; KV or
responses are never reused across episodes. Merged queue requests may contain pending
actions, so the actual model batch count should be read from the generation counters.
Model logs go to `workers/shared-method/method.log`; environment logs stay in
`workers/000/` and so on.

Infrastructure or cleanup failures pause picking up new episodes, wait for all in-flight
attempts to finish and commit, then restart the shared model and environments together
and re-verify identity. Policy errors only terminate the affected episode and do not
restart the model. A shared-model process crash affects all in-flight episodes using it;
use the default replica mode when stronger failure isolation is needed. Quantization,
tensor parallelism, vLLM/SGLang replacement and stochastic sampling batches are not
enabled in this round.

NaVIDA defaults to `decoding: upstream`, preserving `use_model_defaults=True`. The local
checkpoint overrides the apparent do_sample=False with sampling defaults. Shared
singleton inference preserves that behavior and saves/restores CPU/CUDA RNG per session.
Batch sizes above 1 require explicit `method_settings: {"decoding": "greedy"}`, which
disables those overrides while retaining BOS/EOS/pad token defaults. Greedy changed real
trajectories in validation, so it must use a separate baseline. Larger batches are
diagnostic options, not a claim of token-level or closed-loop equivalence.

## Simulation and assets

Isaac/InternUtopia uses a single Kit App and switches the selected scene via reset. The
binding defaults allow `max_control_steps=500` control decisions and `max_task_steps=25000`
upstream task steps; the latter is not the number of model decisions. Both belong to
benchmark_settings and enter the comparison key. Upstream termination reasons and task
counters are recorded separately. `navida-vlnverse-two-scenes.json` is a deployment
diagnostic config with 20 decisions per scene and must not be used to report full
benchmark scores. The exact rc/build identity of the local 5.0 installation is kept in
the lock file; model initialization, Kit startup, scene resets and actual inference time
are accounted separately.

The first run computes content digests of weights and selected assets, timed separately
from model initialization; the presence of a file does not mean its identity was
verified. The run directory contains the resolved config, per-role minimal configs,
environment/model/scoring locks, episode attempts, trajectories, private evidence,
timing and independent evaluations.

## Docker

The experiment selects `launcher: docker`, with per-role resource settings:

```json
{
  "image": "registry.example/navida@sha256:<64-hex real digest>",
  "container_python": "python",
  "gpu": 0,
  "mounts": [
    {"source": "<host-weights-dir>", "target": "/weights", "read_only": true}
  ],
  "settings": {"checkpoint": "/weights/navida"}
}
```

Host-side mount sources are the only place host paths appear, and they live only in your
untracked resource map. Path checks happen on the host; in-container resources use the
mounted container paths, avoiding implicit path translation. Selected external plugins are loaded into
the container with the code snapshot, without extra mounts. Data/weights use read-only
mounts; model and shader caches need explicitly writable mounts. Source and each role's
config are mounted read-only; the whole project is never mounted, and the method does
not receive the environment's role config. Images must pre-install the respective SDK
and original model dependencies; if the image is absent, Docker pulls it by the pinned
digest.

The two containers connect to the local control process via host loopback ports; no
public network API is exposed. The launcher only stops containers it created with random
names for this run. Docker command construction is testable, but with insufficient daemon
privileges a real container run must be marked unverified.

## Resume and throughput

The unit of recovery is the episode; the default infrastructure retry allowance is 0 —
set `max_infrastructure_retries` in the original experiment when retries are needed.
Policy errors are not retried; uncommitted episodes rerun from scratch. Completed
attempts are never re-inferred, and mixing into an old run is refused when source code,
plugins, models or assets change. After an infrastructure failure, workers are rebuilt
before the next allowed attempt, re-verifying runtimes, assets and the episode list.
Offline scoring after an interruption reads the frozen metric config directly and does
not depend on previously generated score reports.

`transport: binary` uses lossless buffer framing; `json` is the reference protocol.
Lossy compression, quantization or engine replacement are not enabled.
`shards/shard_index` splits the same frozen set across independent runs; each shard may
additionally set `parallelism`. Episodes are filtered/limited first, then sharded, then
dynamically assigned by the replica queue — different replicas never collect the same
episode twice. Device capacity across different runs is still coordinated by the caller.

`timing.json` stores cold start, reset, act and step times; episode records include
total RPC time. NaVIDA additionally reports preprocess/generate/decode times, the actual
generation count, a histogram of generation batch sizes and input/output token counts;
executing a pending action is not counted as a model generation. Each episode commits
only new attempts, and JSONL is rebuilt on exit or resume.

`timing.json.sessions` records per run/resume the start time, rollout wall time, the
committed decisions/completions, plus each replica's
`attempts/episode_wall_time_s/busy_fraction` and phase timings. `decisions_per_second`
uses the actual rollout wall time; a pending action is still a decision, not a
generation. The top-level `method/environment` timings keep only a replica-0
compatibility view. Each session's `method_workers` provides `current/retired` counts
de-duplicated by actual model worker; the replicas' `method_worker_index` records
ownership — in shared mode the same model seen by every environment must not be counted
repeatedly. `batcher.batch_sizes/requests/queue_wait_s` describes queue batches and total
request wait time, while `phases.generation_batch_sizes` holds the actual model
generation batches.

`collection.json.episodes_per_hour` is completions divided by accumulated rollout wall
time, including concurrency, resets, communication, commits and any restarts in between;
`collection_episodes_per_hour` additionally includes resource hashing, loading and
identity verification. Neither includes process teardown at the end or offline scoring.
`episode_wall_time_sum_s` keeps the plain sum of per-episode times and must not be used
as the denominator of concurrent throughput. On resume, wall times accumulate across
sessions and completed episodes are not double-counted; if a hard interrupt loses timers
or counts cannot be reconciled, throughput returns null. Numerical equivalence,
cross-GPU consistency and throughput gains each need their own measurement.
