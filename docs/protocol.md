# Protocol and Lifecycle

[简体中文](protocol.zh-CN.md) | English

The current wire envelope is `nav-eval/0.2`; new experiment configs use
`nav-eval-experiment/1`, plugin declarations use `nav-eval-plugin/1`, and atomically
committed artifacts use `nav-eval-rollout/0.4`. The scope is trusted single-machine
evaluation — TLS, public-network authentication, multi-tenant scheduling and
step-level exactly-once delivery are out of scope.

## Workers and sessions

A worker first prints its local endpoint on startup; a live process is not a ready one.
`prepare` loads the real model or initializes the real scene, after which
`describe.ready=true`; `/health` returns 503 until then. `describe` carries the role,
plugin identity, sensors/actions, capture schema, clock, session capacity and codec; the
control side verifies all of it against the resolved plan.

```text
prepare -> describe / attest -> episodes
  -> environment.reset(episode_id, seed, session_id)
  -> method.reset(public_context, session_id)
  -> [method.act -> environment.step -> public transition]*
  -> environment.finish -> close_episode -> atomic attempt commit
  -> offline evaluate
```

By default a worker serves exactly one active session at a time. Each attempt uses a
different session_id; the method's internal generation is regenerated as well.
MethodBoundaryAdapter checks episode, generation and observation sequence numbers and
rejects late results; transition-capable methods receive public feedback for every
executed sub-action. Methods keep weights resident, while reset/close cleans history,
maps, caches, pending actions and other algorithm state.

Parallel runs create multiple independent worker pairs that receive whole episodes via a
dynamic queue. The exclusive-session constraint per worker is unchanged; the control side
only assigns the next task to a replica after the previous attempt commits. All replicas
verify identical assets, runtimes and episode lists before executing; every record keeps
a `replica_index`.

An explicit `inference.mode: shared` lets multiple environments connect to one method
worker. The method must declare the `independent_greedy` batching capability and
implement `act_batch`; the SDK creates an independent existing method boundary per
session. The external RPC format is unchanged, `describe.session_capacity` becomes the
configured limit, and internally only ready act calls from different sessions are merged,
with bounded request counts and wait times; a session can never have two outstanding
calls at once. `act_batch` is a multi-session inference batch, not the action chunk a
single decision may return. The shared-inference `method_worker_index` de-duplicates
timing; on failure, in-flight work is drained before a coordinated restart.
Shared mode defaults to max_batch_size=1. Its live service may declare
`independent_sessions` with isolated RNG; larger batches require `independent_greedy`.
Different decoding settings must not be mixed in an equivalence baseline.

## Public observations and time

Observations contain `episode_id / sequence / sim_time_s / sensors / sensor_specs` and
optionally `control_tick`. Every sensor declares modality, dtype, shape, unit, frame,
source and calibration. Validation covers shape/dtype, RGB ranges, camera size and hfov;
depth units must be m. `rendered / measured / estimated` are allowed sources; privileged
ground truth must not be used as input.

Tasks with a simulation clock use `sim_time_s` in seconds. The discrete
Habitat/VLNVerse tasks currently set `sim_time_s=null` and record control steps via
`control_tick`, avoiding passing off action counts as physical seconds. Goal coordinates,
reference paths, geodesic distances and scoring information only enter the environment's
private evidence.

Python process isolation separates roles; it is not a security sandbox for untrusted
code. Docker mounts only the code snapshot, the current role's config and explicitly
configured resources by default — keep private benchmark data out of the method
container.

## Actions and failures

An action batch carries episode, observation_sequence and 1–16 actions.
primitive.forward is in meters, left/right in radians; twist uses m/s, rad/s and seconds.
STOP is a separate action. camera_tilt/wait have types but still require an actual
binding to execute. Controller conversions must be selected explicitly. NaN, Inf,
illegal magnitudes, unknown actions and frames are never silently clipped into legal
actions.

The Habitat R2R/RxR bindings explicitly advertise 15° and 30° turns in
`supported_action_semantics`. `track: native` selects the method's declared
magnitude and freezes it into the binding, environment configuration and
`compare_key`. NaVid / Uni-NaVid use 30°; the other current methods use 15°.
One 30° turn remains one control step. `standardized` retains the benchmark's
default magnitude and rejects conflicts during planning. Native runs with
different magnitudes do not have identical evaluation protocols.

terminated/truncated both stop the remaining chunk. `PolicyViolation` is recorded as a
policy_error; protocol errors, timeouts and worker failures are infrastructure_errors.
Policy failures are not retried; the infrastructure retry budget is frozen with the
config. The first completed/policy_error attempt of an episode is the canonical attempt;
otherwise the last infrastructure failure is shown, and all committed attempts are kept.

HTTP 400 marks protocol or policy errors, 500 marks internal worker errors; full
exceptions are kept in role logs. Steps whose execution state cannot be confirmed are not
re-sent. Recovery starts at episode boundaries; it is not mid-simulation-state recovery
and does not promise exactly-once execution. Monitoring or collection failures never
overwrite committed episode results.

## Lossless tensor transport

The reference codec is JSON/base64: explicit `tensor_b64 / dtype / shape / byte_order`,
supporting uint8, float32 and int32, with the wire fixed to little-endian. Non-native
endianness of ndarrays is converted explicitly.

`transport: binary` is negotiated via describe, then sends a 4-byte header length, a JSON
header and the raw tensor buffers. Decoding validates lengths, indices and bounds and
restores identical tensors. JSON is limited to 8 MiB and binary to 64 MiB; image content,
sizes, compression and history-frame selection are unchanged. NaVIDA's own JPEG prompt
images are algorithm preprocessing and still follow upstream behavior.

HTTP binds to loopback by default; Docker publishes to random ports on the host loopback.
The implementation targets trusted single-machine evaluation; TLS, authentication,
multi-tenant resource scheduling and cross-machine clusters are out of scope.
