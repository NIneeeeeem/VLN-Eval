# StreamVLN experimental tensor batching

Opt-in plugin ID `streamvln_batch`, registered with
`plugin_dirs: ["experimental/streamvln_batch"]`. Uses the existing shared inference
worker with `inference.mode: shared`, `max_batch_size: 2`, `4` or `8`, and independent
environment sessions. One GPU hosts one shared model; environment resources must
point to the same checkpoint/runtime as the production StreamVLN configuration.
The plugin requires the deployed **Transformers 4.45.1** runtime and rejects other
versions before loading weights. Its generation/cache APIs are version-sensitive.

> The batch driver scripts (`validate_streamvln_batch.py`,
> `run_streamvln_batch_full.py`, …) and the `runs/` evidence directories cited
> below belong to the development tree and are not shipped with the repository.

Implementation and acceptance plan:

1. Keep production `nav_eval/`, `extensions/` and upstream code unchanged so the
   running full-split suite retains its frozen identities.
2. Maintain independent prompt RNG, frame history, action queue, accumulated
   embeddings and KV cache per session. Preserve the 32-step cache reset.
3. Pack unequal past/suffix lengths with separate padding masks; retain real
   per-row rotary positions. Use upstream HF greedy generation and checkpoint
   logits processors, including repetition penalty and both EOS tokens.
4. Unpack KV rows without padding or tokens generated after each row's EOS.
   Release batch storage before the next round; keep no views into other sessions.
5. Verify mixed cached/fresh sessions, cache-reset boundaries, unequal output
   lengths, singleton tail, close/reset isolation and duplicate-request rejection.
6. Compare real R2R single-input actions/evidence against the production plugin;
   compare batch=2/4 against the same episode selection and report throughput,
   actual generation batch histogram, latency and peak GPU allocation. Any action
   divergence prevents an equivalence claim. Keep this experimental evaluation
   separate from the production multi-method suite.

This is an experimental plugin. Tensor shape changes can alter BF16 rounding and
greedy choices. A passing small test does not establish full-split parity.
Vision embedding preparation currently remains per-session; language-model
prefill and autoregressive decoding are tensor-batched. All queued actions are
served without unnecessarily running the model.

## Validation status (2026-09-29)

Ten targeted tests pass under the deployed StreamVLN interpreter on CPU.
The full CPU/worker regression suite reports 80 tests: 77 pass and three
version-specific integration tests skip in the control interpreter. Those
three tests pass in the StreamVLN environment. Two additional tests check
full-shard metric weighting, duplicate coverage rejection and startup reporting.
The real small Qwen tests exercise batch sizes 1, 2, 4 and 8 for three generation
rounds, unequal prefixes, mixed fresh/cached sessions, and resets. Generated IDs
match singleton generation; per-layer KV values match within atol=1e-6 and
rtol=1e-5. Separate tests cover early-EOS cache cropping and session isolation.
Ruff E9/F passes. Five experiment configurations resolve without loading models.
The ongoing full-suite source hash still matches its frozen source identity.

**Real GPU short validation passed after two compatibility repairs.** The
released checkpoint allocates four unused empty KV layers; packing/unpacking
preserves those slots. Installed FlashAttention returns five unpadding values
while HF 4.45.1 expects four; a scoped adapter retains the original four outputs
and restores the function after generation. No upstream/package files changed.

Four episodes (1, 4, 10, 16), all 219 actions, captured trajectory evidence and
per-episode metrics match production exactly. Rollout took 32.04 s versus 70.24 s
(2.19x). Actual generation batches were 47 singleton and 4 size-two batches;
this bounded comparison does not establish the throughput of full size-four
batches. Peak model allocation was 17,941 MiB; reservation was 24,358 MiB.
Evidence: `runs/streamvln-batch-validation-20260929-v3/production-comparison.json`.

The production suite is paused with 478/1839 StreamVLN episodes preserved.
Separate full validation was paused in `runs/streamvln-bs4-full-20260929` after
173 completed episodes: GPUs 1-4 each hosted one model with four
environments, max batch 4 and wait 10 ms. Four disjoint shards must cover all
1839 episodes exactly once before aggregate metrics/paper comparisons qualify
as full-split diagnostics. `scripts/run_streamvln_batch_full.py --report-only
--output runs/streamvln-bs4-full-20260929` refreshes the report without inference.

**The expanded sample disproves exact trajectory equivalence.** Among 173
common completed episodes, 30 (17.34%) have different actions/trajectory
evidence. Seven success outcomes flip (one success-to-failure, six in the other
direction). Paired SR is 49.71% vs 52.60%; SPL 44.14% vs 46.02%; NE 5.120 m vs
4.753 m. These are selected partial results, not full-split scores. See
`runs/streamvln-bs4-full-20260929/path-comparison.json`.

A four-environment shared-model bs=1 control matches all 15 completed episodes
overlapping the original run (1106 actions), supporting that the deviations
are related to tensor batching rather than session sharing alone. Numerical
changes from BF16/padded FlashAttention remain a hypothesis, not a proven
operator-level diagnosis. Same-GPU, eight-environment bs=1/4/8 comparisons
completed in `runs/streamvln-batch8-fair-20260929` after user authorization of
bs=8. All 16 episodes finished without errors in each stage. Rollout times were
137.99 / 132.96 / 135.13 seconds, respectively. Both batched stages changed
5/16 paths; SR was 50.00 / 43.75 / 56.25%. This deliberately diagnostic subset
is not representative of the full split. Actual bs=8 generation batches averaged
1.44 and peaked at 7; peak model allocation was 26.41 GiB. Full evaluation stays
paused because trajectory equivalence failed. See that directory's `REPORT.md`
and paired comparison JSON files for the final evidence and limitations.

The prepared bounded validation command is:

```bash
python -u -B scripts/validate_streamvln_batch.py \
  --output runs/streamvln-batch-validation-20260929 --gpu 2
```

It sequentially tests production singleton, experimental singleton, four shared
environments at batch=1, and the same four environments at batch=2 and batch=4.
Default episodes are 1, 4, 10 and 16. `--stage` selects one stage; `--episodes`
overrides the bounded selection. Existing stage runs are never overwritten.
The summary retains identity checks, action/evidence/metric differences, actual
generation batch histograms, episodes/h, actions/s, RPC latency, and experimental
model allocated/reserved CUDA peaks. It only reports an equivalent speedup when
the comparison checks pass. GPU peaks are PyTorch-process peaks, not whole-device
usage. Other GPUs share host CPU/storage, limiting timing isolation.
