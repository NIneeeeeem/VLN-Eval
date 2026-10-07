# Linux Deployment and Inference

[简体中文](deployment.zh-CN.md) | English

Nav-Eval separates *what to run* (experiment configs, portable, in Git) from
*where to run it* (permanent `configs/local.json`, machine-specific, git-ignored). Every
method keeps its upstream dependency environment; the framework launches the
model and the simulator as separate processes and wires them together.

This page walks through the complete path: verify the control plane → build
the environments → download assets → register the installation once → run, scale and
containerize. Asset download links live in [Assets](../data/README.md);
per-benchmark setup lives in [Benchmarks](benchmarks.md).

After downloading assets, register weights with `python -m nav_eval configure --method <model> --checkpoint <path>`
and simulators with `--simulator <plugin> --data-root data --environment-python <interpreter>`.
The installation is permanent in
`configs/local.json`; `plan/run` load it automatically. Repository-local environments
can be installed with `bash scripts/setup_environment.sh <method-or-simulator>`.
Select models and benchmarks in Bash:

```bash
METHOD=streamvln BENCHMARK=r2r_ce GPU=0 bash scripts/eval.sh
GPU=0 bash scripts/eval_suite.sh
# Per-invocation replica allocation; configure sufficient memory budgets first.
METHOD=streamvln BENCHMARK=r2r_ce GPUS=0,1 bash scripts/eval.sh
```

`scripts/eval.sh <run-directory>` retains offline scoring.

## Overview

A typical evaluation involves:

| Piece | Where it lives | Example |
|---|---|---|
| Experiment config | `configs/experiments/*.json` | `navida-r2r.json` |
| Permanent installation | `configs/local.json` (git-ignored) | registered once with `configure` |
| Method environment | shared Conda prefix | `envs/nav_streamvln` or `envs/nav_vlm` |
| Simulator environment | `envs/nav_habitat030/` etc. | one per Habitat version |
| Datasets and scenes | `data/` (git-ignored) | `data/datasets/r2r/`, `data/scene_datasets/mp3d/` |
| Weights | `checkpoints/<method>/` (git-ignored) | `checkpoints/streamvln/` |
| Run outputs | `runs/<output>/<run-id>/` | `episodes.jsonl`, evaluations |

Prerequisites: Linux, an NVIDIA GPU, Conda, and the datasets/scenes for your
benchmark ([Assets](../data/README.md)).

## Control-plane verification

From the repository root, no GPU or simulator needed:

```bash
python -B -m nav_eval plugins list
python -B -m nav_eval doctor
python -B -m nav_eval plan \
  --config configs/experiments/navida-r2r.json \
  --gpu 0
```

`plugins list` and `doctor` check plugin discovery, interpreters and tooling.
`plan` resolves the full experiment — benchmark, simulator, binding, sensors,
actions and resource settings — and reports incompatibilities before anything
launches. Models and simulators are loaded only during the preflight/prepare
stages of `run`.

## First run walkthrough (NaVIDA on R2R-CE)

NaVIDA is the simplest method to set up: it loads directly through
Transformers, so the method environment is plain `pip`. The same steps apply
to every other method — swap in its environment from
[Method environments](#method-environments). The NaVIDA checkpoint is not yet
public ([arXiv 2601.18188](https://arxiv.org/abs/2601.18188)); until it is
released, run this walkthrough with any model whose weights you have, or use
the [benchmark smoke recipes](../configs/benchmarks/README.md) which need no
weights at all.

**1. Habitat 0.3.0 simulator environment:**

```bash
# Run from the cloned Nav-Eval repository root.
export NAV_EVAL_ROOT="$PWD"
mkdir -p "$NAV_EVAL_ROOT/envs"
bash scripts/setup_environment.sh habitat030
```

**2. Method environment:**

```bash
export CUDA_HOME=/usr/local/cuda-12.8
bash scripts/setup_environment.sh vlm
MODEL_PY="$NAV_EVAL_ROOT/envs/nav_vlm/bin/python"
```

**3. Assets** under the repository root (relative paths resolve from the
directory you invoke `nav_eval` in):

```text
data/
├── datasets/r2r/val_unseen/val_unseen.json.gz
└── scene_datasets/mp3d/<scene>/<scene>.glb
checkpoints/navida/        # NaVIDA checkpoint (config + tokenizer + weight shards)
```

**4. Permanent installation** — save as `configs/local.json`
(git-ignored; absolute paths also work):

```json
{
  "method": {
    "python": "envs/nav_vlm/bin/python",
    "gpu": 0,
    "min_free_memory_mib": 12000,
    "timeout_s": 900,
    "settings": {"checkpoint": "checkpoints/navida"}
  },
  "environment": {
    "python": "envs/nav_habitat030/bin/python",
    "gpu": 0,
    "min_free_memory_mib": 4000,
    "timeout_s": 900,
    "settings": {"data_root": "data"}
  }
}
```

Choose a physical card with `GPU`/`GPUS` in Bash. The installation settings below are saved once.

**5. Run.** The shipped `configs/experiments/navida-r2r.json` selects one
episode; remove `episode_limit` for the full 1,839-episode split:

```bash
python -B -m nav_eval plan --config configs/experiments/navida-r2r.json \
  --gpu 0
python -B -m nav_eval run --config configs/experiments/navida-r2r.json \
  --gpu 0 --output runs/navida-r2r
python -B -m nav_eval evaluate --run runs/navida-r2r/<run-id>
```

When `runs/navida-r2r/<run-id>/` contains `episodes.jsonl` and an evaluation
with SR/SPL, the environment is ready. Per-role logs are in
`workers/*/method.log` / `environment.log`; failures are summarized in
`failure.json`.

## Real runs

The four commands cover the whole lifecycle:

```bash
python -m nav_eval plan    --config <experiment.json> --gpu 0
python -m nav_eval run     --config <experiment.json> --gpu 0 --output <dir>
python -m nav_eval resume  --run runs/<output>/<run-id>   # continue unfinished episodes
python -m nav_eval evaluate --run runs/<output>/<run-id>   # re-score saved evidence, no inference
```

Notes on the permanent installation:

- Build yours from [example.json](../configs/resources/example.json). Keys
  `method` / `environment` set the two default roles; `runtimes` keyed by
  plugin ID configures others. Each role accepts `python`, `gpu`,
  `pythonpath`, `library_paths`, `cwd`, `env`, `timeout_s`,
  `min_free_memory_mib`, and `settings` such as `checkpoint` / `vision_tower` /
  `data_root`.
- `plan` and `run` accept `--gpu <index>`: the physical GPU for both roles
  this invocation, overriding the file. Keep static inventory — interpreters,
  weights, data paths — in the file and pick the card per run; the
  `scripts/` wrappers forward it from the `GPU` environment variable.
- Resource settings accept the manifest's `requires.paths` /
  `resource_settings` keys. Experiment parameters — cameras, actions, success
  thresholds — belong in the experiment config, where they enter the
  comparison key.
- Model and rendering processes may sit on different GPUs; a physical card is
  mapped to in-process device 0 via `CUDA_VISIBLE_DEVICES`.
- VLNVerse registers the Isaac interpreter and `data_root`; install its local
  task runtime with `python -B scripts/setup_benchmark.py vlnverse`. See
  [Benchmarks: VLN-VERSE](benchmarks.md#vln-verse-vlnverse).

Execution behavior:

- Preflight checks paths, interpreters, GPUs and disk before launch. Model
  loading and environment initialization run concurrently; episodes start
  after all replicas pass prepare and identity verification.
- Models stay resident across episodes; `reset`/`close_episode` clean history
  and caches following the upstream reset rules.
- The launcher cleans up whole process groups, including subprocesses started
  by interpreter scripts. Ctrl-C keeps committed attempts; uncommitted work
  resumes from episode boundaries.
- Start a new run after changing weights, decoding, camera geometry or the
  simulator — `resume` continues only the same experiment.

## StreamVLN on R2R

`configs/experiments/streamvln-r2r.json` runs a 33-episode subset (three
routes from each of the 11 val_unseen scenes); remove `episodes` for the full
1,839-episode split. The method environment doubles as the habitat024
simulator environment (recipe in the [README quickstart](../README.md) and
below under [Method environments](#method-environments)).

```bash
python -B -m nav_eval run \
  --config configs/experiments/streamvln-r2r.json \
  --gpu 0
```

Defaults are `preprocess_mode: lazy` and `tokenizer_mode: reuse`: keep every
raw RGB observation, preprocess only the selected current/history frames, and
run upstream prompt construction against a dedicated tokenizer copy. Model
precision, generation settings and random draws follow upstream. `eager` /
`upstream` reproduce the method's own preprocessing behavior exactly.

Set `min_free_memory_mib` from a conservative model-plus-simulator budget —
a short smoke underestimates the memory a long trajectory needs. StreamVLN
supports independent replicas; it does not declare shared sessions or batched
generation. Compare runs only when experiment, assets, method identity and
metric settings are identical.

## Default decoding and batch organization

Every method ships exactly one frozen default decoding, recorded in
`runtime_identity()`; decoding must not change between baselines, and
equivalence comparisons must not mix decoding settings.

Default decoding per method:

| Method | Default decoding |
|---|---|
| `navid` / `uni_navid` | Upstream checkpoint sampling — generation delegates to the upstream agent untouched (`extensions/methods/navid/service.py`) |
| `navida` | `decoding: upstream` (`use_model_defaults=True`; checkpoint sampling overrides apply). Optional `decoding: greedy`, required for batch sizes above 1 |
| `streamvln` | Greedy (`do_sample=false`, `num_beams=1`) |
| `navila` | Greedy (`do_sample=false`, `temperature=0.0`) |
| `awarevln` | Greedy (`do_sample=false`, `temperature=0.0`) |

Batch organization:

- Default `inference.mode: replicas` — every replica owns a resident model,
  each worker holds exactly one session, and observe/reason/act proceeds
  strictly in order within an episode, so generation is always singleton.
  Upstream global histories (NaVid) and caches (StreamVLN) need no
  thread-safety changes.
- Explicit `inference.mode: shared` — one model serves several environment
  sessions. `max_batch_size` defaults to 1 and preserves singleton numerical
  behavior; values above 1 require the `independent_greedy` capability plus
  the method's `batching_requires` settings (today only NaVIDA, which then
  requires `decoding: greedy`). See
  [Shared model and batching](#shared-model-and-batching).

## Parallel inference

Add `"parallelism": 2` to the experiment and two assignments to the resource
map (edit [parallel.example.json](../configs/resources/parallel.example.json)
directly if you prefer):

```json
{
  "replicas": [
    {"method": {"gpu": 0}, "environment": {"gpu": 0}},
    {"method": {"gpu": 1}, "environment": {"gpu": 1}}
  ]
}
```

Rules:

- `replicas` entries merge with the existing `runtimes` / `method` /
  `environment` configuration — interpreters, weights and settings carry
  over; each entry pins only deployment fields (GPU, memory, timeouts).
- The number of `replicas` must equal `parallelism`, and every GPU-requiring
  role must spell out its device. A replica's method and environment may sit
  on different cards; several replicas may share one card.
- Replicas may not override checkpoints or method/benchmark settings.

Example:

```bash
python -B -m nav_eval run \
  --config configs/experiments/navida-r2r-parallel.json \
  --gpus 0,1
```

`navida-r2r-parallel.json` pins one GPU, two replicas and 4 diagnostic
episodes. All replicas share one episode queue and pick up the next task on
completion, so long trajectories never hold other replicas idle. Other
methods use the same `parallelism`/`replicas` configuration with their own
base resource map.

GPU roles take a positive `min_free_memory_mib` — the conservative VRAM
budget of that worker. Preflight sums budgets by physical GPU UUID (numeric
indices and UUIDs pointing at the same card are merged) and checks the total
fits available VRAM. Leave headroom for the longest history and other jobs;
the speedup from co-locating replicas on one card depends on model size and
GPU load.

Operational details:

- Per-replica logs land in `workers/000/`, `workers/001/`, …;
  `replicas.lock.json` locks each replica's runtime and assets.
- Before execution, all replicas agree on identity and the episode list.
  Restarts affect only worker pairs that hit infrastructure errors or failed
  cleanup; identity is re-checked. Results, scoring and resume stay within
  one run directory.
- The `local` launcher stays single-replica to keep RNG and simulation
  threads isolated.

## Shared model and batching

Adapted methods can serve multiple independent environments from one resident
model (among the shipped methods, NaVIDA declares this capability). Add to
the experiment:

```json
{
  "parallelism": 2,
  "inference": {"mode": "shared", "max_batch_size": 1, "max_wait_ms": 0}
}
```

```bash
python -B -m nav_eval run \
  --config configs/experiments/navida-r2r-shared.json \
  --gpus 0,1
```

- Deployment resources merged from all `replicas[].method` must be identical;
  the model is loaded once and its VRAM budget counted once, while
  environment budgets accumulate.
- `parallelism` caps environments/sessions; `max_batch_size` caps merged
  requests per batch (1 … parallelism, default 1); `max_wait_ms` bounds the
  wait for more requests once the first is ready (default 5 ms — a queueing
  bound, not an RPC latency cap).
- Only current requests from *different* episodes merge; within an episode
  the loop still waits for execution and new observations. History, action
  queues, generations and observation sequences are isolated per session.
  NaVIDA's multi-input uses left padding and keeps the upstream JPEG/frame
  sampling/action parsing; KV and responses are never reused across episodes.
- Model logs go to `workers/shared-method/method.log`; environment logs stay
  in `workers/000/`, … Read the real model batch count from the generation
  counters — merged queue requests may contain pending actions.
- Infrastructure or cleanup failures drain in-flight attempts, commit, then
  restart model and environments together and re-verify identity. Policy
  errors terminate only the affected episode.
- NaVIDA defaults to `decoding: upstream` with per-session RNG
  save/restore. Batch sizes above 1 require
  `method_settings: {"decoding": "greedy"}` — greedy changed real
  trajectories in validation, so keep it as a separate baseline. Quantization,
  tensor parallelism and vLLM/SGLang serving are on the roadmap, not in this
  release.

## Simulation and assets

Isaac/InternUtopia uses a single Kit App and switches scenes via reset.
Binding defaults allow `max_control_steps=500` control decisions and
`max_task_steps=25000` upstream task steps; both are benchmark settings and
enter the comparison key. `navida-vlnverse-two-scenes.json` is a two-scene
deployment diagnostic (20 decisions per scene). The Isaac rc/build identity
is kept in the lock file; model init, Kit startup, scene resets and inference
time are accounted separately.

Habitat platform notes:

- ActiveVLN/OneVLA starter configs share the fresh Habitat 0.2.4 prefix.
  Install only 0.2.4 and 0.3.0; specialized task recipes use these same SDKs.
- R2R/RxR bindings on both Habitat versions offer the measured `pose` sensor (xyz +
  quaternion, habitat world frame) required by GA-VLN.
- The R2R/RxR bindings on both versions accept opt-in `camera_tilt` actions
  (`benchmark_settings.allow_tilt`) for InternVLA-N1's ground-view probes —
  each tilt consumes one control tick of the 500-step budget.

The first run computes content digests of weights and selected assets. The
run directory contains the resolved config, per-role minimal configs,
environment/model/scoring locks, episode attempts, trajectories, private
evidence, timing and independent evaluations.

## Method environments

Install from fresh `envs/nav_*` prefixes using the commands below. No pre-existing
model environment is required. The three core prefixes are `nav_streamvln`
(including Habitat 0.2.4), `nav_vlm` and `nav_habitat030`.

Built-in model code is bundled in `extensions/methods/<method>/runtime/`.
Only third-party Python dependencies, weights and HF caches belong to the method
environment. No method checkout, editable method install, source `pythonpath`
or source `cwd` is needed. Dependency files below record the installed versions.

For a shared installation, use `bash scripts/setup_environment.sh streamvln habitat024`
for StreamVLN / GA-VLN plus Habitat, and `bash scripts/setup_environment.sh vlm`
for the other built-in methods. These create `envs/nav_streamvln` (Python 3.9,
Transformers 4.45.1) and `envs/nav_vlm` (Python 3.10, Transformers 4.57.0).
The model-specific files below are alternative legacy recipes, not additions to
the shared requirements. Python version differences and namespaced VLM forks
alone do not require separate environments.

| Method | Installation | Notes |
|---|---|---|
| NaVIDA | Fresh install — [commands above](#first-run-walkthrough-navida-on-r2r-ce) | Transformers-only; no upstream repo |
| StreamVLN | `configs/environments/streamvln-inference.txt` | Model code and prompt helpers are bundled; Habitat is needed only by the environment worker |
| NaVid / Uni-NaVid | `configs/environments/navid-inference.txt` | Set `vision_tower` to EVA weights, or place `eva_vit_g.pth` inside the checkpoint directory; CLIP processor configs are bundled |
| NaVILA / AwareVLN | `configs/environments/vila-inference.txt` | Python ≥3.10; isolated method namespaces contain each VLM fork |
| ActiveVLN | Any Qwen2.5-VL-capable env; weights under `checkpoints/activevln/{rl,sft}_{r2r,rxr}` | In-process Transformers serving; sampling (t=0.2, top_p=0.8) is platform-seeded per episode |
| JanusVLN | `configs/environments/janusvln-inference.txt` | VGGT is bundled; its KV cache is cleared per episode; greedy 24 tokens |
| InternVLA-N1 | `configs/environments/internvla-n1-inference.txt` | Ground-view probes use `camera_tilt` (`benchmark_settings.allow_tilt`, habitat030); NavDP flow head is platform-seeded |
| OneVLA | `configs/environments/onevla-inference.txt` | `checkpoint` = `run/checkpoints/*.pt`; `run/config.yaml` + `run/dataset_statistics.json` must exist; set resource `base_vlm` when needed; default attention is `sdpa` |
| GA-VLN | `configs/environments/gavln-inference.txt` | Needs `pose` (habitat024/030), `vision_tower` (SigLIP) and `vggt_path`; 8-step KV windows reset via `model.reset_for_env`; R2R only |

InternVLA-N1's auxiliary depth weights live inside the configured checkpoint
directory: `depth_anything_v2_metric_hypersim_vits.pth` for asynchronous NextDiT,
or `depth_anything_v2_vits.pth` for asynchronous NavDP. Source code and depth resize
are independent of Habitat. A method's third-party dependency installation remains
separate from the simulator SDK; CUDA wheels must match the selected interpreter.

StreamVLN environment:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
bash scripts/setup_environment.sh streamvln habitat024
```

Source revisions and notices live in each runtime's
`provenance.json` and `notices/`; the root Apache license does not override their
terms. StreamVLN declares CC BY-NC-SA 4.0; JanusVLN/GA-VLN have no discovered
top-level license grant in the recorded revisions.
Export environment locks for reproduction (`build/` is git-ignored):

```bash
mkdir -p "$NAV_EVAL_ROOT/build/env-locks"
for role in nav_vlm nav_habitat030; do
  conda list -p "$NAV_EVAL_ROOT/envs/$role" --explicit \
    > "$NAV_EVAL_ROOT/build/env-locks/$role-conda.txt"
  "$NAV_EVAL_ROOT/envs/$role/bin/python" -m pip freeze \
    > "$NAV_EVAL_ROOT/build/env-locks/$role-pip.txt"
done
```

## Docker

No prebuilt images are shipped — build one from your working environments
with the recipe below. The recommended layout is a single combined image
holding the method and Habitat environments side by side; the host runs the
control plane and starts worker containers from it. Launcher implementation:
`nav_eval/execution/launchers.py`.

Planning guide: one native method needs **two runtime environments** (method +
Habitat 0.3.0), packaged as one combined image or two. Four-session
`replicas` then runs 4 method + 4 simulator containers (four model copies);
NaVIDA `shared` runs 1 + 4. Shared installation uses two model dependency groups,
`streamvln` and `vlm`, plus SDK environments as needed. StreamVLN with Habitat
0.2.4 can use the same interpreter for both roles; separate worker processes
still load the model and simulator.

### Host preparation

Install the NVIDIA driver, Docker Engine and NVIDIA Container Toolkit per the
[NVIDIA installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html);
the invoking user needs daemon access. Initial runtime configuration (admin):

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
nvidia-smi && docker info
CUDA_IMAGE='your-cuda-image@sha256:REPLACE_WITH_REAL_DIGEST'
docker run --rm --gpus device=0 "$CUDA_IMAGE" nvidia-smi
```

Do not install host kernel drivers inside images; a working `nvidia-smi` does
not by itself verify headless rendering — test with a real episode.

### Build a combined image

Snapshot your working environments in a git-ignored build directory:

```bash
mkdir -p "$NAV_EVAL_ROOT/build/docker-streamvln" && cd "$NAV_EVAL_ROOT/build/docker-streamvln"
METHOD_ENV="$NAV_EVAL_ROOT/envs/nav_vlm"
SIM_ENV="$NAV_EVAL_ROOT/envs/nav_habitat030"
conda list -p "$METHOD_ENV" --explicit > conda-explicit.txt
"$METHOD_ENV/bin/python" -m pip freeze > pip-freeze.txt
conda pack -p "$METHOD_ENV" -o model.tar.gz
conda pack -p "$SIM_ENV" -o habitat030.tar.gz
```

Dockerfile beside both archives (base image pinned by digest, with the system
libraries Habitat needs, including EGL/OpenGL):

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY model.tar.gz habitat030.tar.gz /tmp/
RUN mkdir -p /opt/envs/nav_vlm /opt/envs/nav_habitat030 \
    && tar -xzf /tmp/model.tar.gz -C /opt/envs/nav_vlm \
    && /opt/envs/nav_vlm/bin/python /opt/envs/nav_vlm/bin/conda-unpack \
    && tar -xzf /tmp/habitat030.tar.gz -C /opt/envs/nav_habitat030 \
    && /opt/envs/nav_habitat030/bin/python /opt/envs/nav_habitat030/bin/conda-unpack \
    && rm /tmp/model.tar.gz /tmp/habitat030.tar.gz
ENV PATH=/opt/envs/nav_vlm/bin:$PATH
ENV PYTHONUNBUFFERED=1
```

```bash
BASE_IMAGE='your-compatible-base@sha256:REPLACE_WITH_REAL_DIGEST'
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t nav-eval-streamvln-habitat030:repro .
```

The launcher references images by **RepoDigest** (`repo@sha256:<64 hex>`),
not a tag or image ID. Publish to your registry and record the digest
([Docker publishing](https://docs.docker.com/reference/cli/cli/image/push/)).

### Resource map and launch

The experiment sets `launcher: docker`; each role replaces `python` with
`image` + `container_python`, keeps settings/budgets, and adds mounts and env:

```json
{
  "method": {
    "image": "registry.example/nav-eval-streamvln-habitat030@sha256:REPLACE_WITH_64_HEX",
    "container_python": "/opt/envs/nav_vlm/bin/python",
    "min_free_memory_mib": 24000,
    "timeout_s": 1800,
    "shm_size": "8g",
    "mounts": [
      {"source": "/srv/nav-eval/assets/checkpoints/streamvln", "target": "/srv/nav-eval/assets/checkpoints/streamvln", "read_only": true},
      {"source": "/srv/nav-eval/cache", "target": "/srv/nav-eval/cache", "read_only": false}
    ],
    "env": {
      "HF_HOME": "/srv/nav-eval/cache/huggingface",
      "XDG_CACHE_HOME": "/srv/nav-eval/cache/xdg",
      "HF_HUB_OFFLINE": "1",
      "TRANSFORMERS_OFFLINE": "1",
      "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
      "PYTHONPATH": "/opt/nav-eval"
    },
    "settings": {
      "checkpoint": "/srv/nav-eval/assets/checkpoints/streamvln"
    }
  },
  "environment": {
    "image": "registry.example/nav-eval-streamvln-habitat030@sha256:REPLACE_WITH_64_HEX",
    "container_python": "/opt/envs/nav_habitat030/bin/python",
    "min_free_memory_mib": 4000,
    "timeout_s": 1800,
    "shm_size": "8g",
    "mounts": [
      {"source": "/srv/nav-eval/assets/data", "target": "/srv/nav-eval/assets/data", "read_only": true},
      {"source": "/srv/nav-eval/cache", "target": "/srv/nav-eval/cache", "read_only": false}
    ],
    "env": {
      "XDG_CACHE_HOME": "/srv/nav-eval/cache/xdg",
      "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics"
    },
    "settings": {"data_root": "/srv/nav-eval/assets/data"}
  },
  "replicas": [
    {"method": {"gpu": 0}, "environment": {"gpu": 0}},
    {"method": {"gpu": 1}, "environment": {"gpu": 1}},
    {"method": {"gpu": 2}, "environment": {"gpu": 2}},
    {"method": {"gpu": 3}, "environment": {"gpu": 3}}
  ]
}
```

Launcher behavior:

- Host preflight and hashing read asset paths before containers start: mount
  assets at the **same absolute path on both sides**, including external
  symlink targets.
- The code snapshot is mounted read-only at `/opt/nav-eval` (runtime packages
  + selected external plugins). Prepare the vision/tokenizer cache before
  enabling the offline flags.
- Workers run as host UID:GID; create writable cache dirs as that user.
  `container_python` must start directly — the image ENTRYPOINT is replaced.
- Container cwd is fixed at `/opt/nav-eval`; resource `cwd` affects only
  host-launched processes. `pythonpath`/`library_paths` are not translated
  into container paths — use `env.PYTHONPATH` / `env.LD_LIBRARY_PATH`,
  keeping `/opt/nav-eval` in PYTHONPATH.
- Each worker selects one physical GPU and uses local device numbering.
  Headless Habitat needs the `graphics` driver capability
  ([NVIDIA driver capabilities](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html));
  `shm_size` does not increase GPU memory.
- Worker HTTP ports are published dynamically on the host loopback — no
  public port 8000, host network or Docker socket mount. This is a
  single-host launcher; Isaac, external VLM services and multi-host execution
  need separate recipes.

Validate a CUDA op inside the image first, then run real episodes:

```bash
METHOD_IMAGE='registry.example/nav-eval-streamvln-habitat030@sha256:REPLACE_WITH_REAL_DIGEST'
docker run --rm --gpus device=0 --entrypoint /opt/envs/nav_vlm/bin/python \
  "$METHOD_IMAGE" -c 'import torch, transformers; print(torch.ones(1, device="cuda").cpu())'
python -B -m nav_eval plan \
  --config configs/experiments/my-streamvln-docker.json \
  --gpu 0
python -B -m nav_eval run \
  --config configs/experiments/my-streamvln-docker.json \
  --gpu 0 \
  --output runs/docker-streamvln-r2r
```

### Optional: one outer container

Run the control plane inside the combined image with `launcher: "python"`:
the framework then starts isolated Python workers instead of extra containers
(no Docker socket mount). Four replicas still mean four model + four
simulator processes with the same memory cost. Set
`method.python=/opt/envs/nav_vlm/bin/python`,
`environment.python=/opt/envs/nav_habitat030/bin/python`, keep
replicas/budgets/assets, and drop the image/container_python/mounts fields:

```bash
docker run --rm --init --gpus all --shm-size 8g \
  --user "$(id -u):$(id -g)" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e HF_HOME=/srv/nav-eval/cache/huggingface -e XDG_CACHE_HOME=/srv/nav-eval/cache/xdg \
  -v "$NAV_EVAL_ROOT:/work/nav-eval:ro" \
  -v "$NAV_EVAL_ROOT/runs:/work/nav-eval/runs:rw" \
  -v "/srv/nav-eval/assets:/srv/nav-eval/assets:ro" -v "/srv/nav-eval/cache:/srv/nav-eval/cache:rw" \
  -w /work/nav-eval --entrypoint /opt/envs/nav_vlm/bin/python \
  nav-eval-streamvln-habitat030@sha256:REPLACE_WITH_REAL_DIGEST \
  -B -m nav_eval run \
  --config configs/experiments/my-container-python.json \
  --gpu 0 \
  --output runs/container-streamvln-r2r
```

Resource GPU IDs must match devices visible inside the container. The outer
container sees both roles' assets, so deployment isolation is weaker than
separate role mounts.

## Reproduction records

For a citable run, keep alongside the run directory: experiment JSON, a
redacted host resource map, repository/upstream commits and local patches,
checkpoint revisions, asset/vision/tokenizer/data hashes, dependency lists,
image digests, interpreter paths, GPU/driver details, run locks, trajectories,
failures and scores. A combined image has one digest for both roles; a digest
does not freeze mounted weights or caches — those are covered by the asset
hashes. Start a new run when changing model, decoding, geometry or simulator;
`resume` only continues the same experiment.
