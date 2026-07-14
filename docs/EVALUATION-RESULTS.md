# 2× DGX Spark DSpark stack evaluation

**Date:** 2026-07-14
**Rig:** 2× NVIDIA DGX Spark (GB10, aarch64, 128 GB unified each), TP=2 over a
200 Gb RoCE link. Head `spark-15d4` / `10.0.0.1`, worker `spark-1826` /
`10.0.0.2`, device `rocep1s0f0` / `enp1s0f0np0`, MTU 9000.
**Model:** `deepseek-ai/DeepSeek-V4-Flash-DSpark` (155.4 GiB checkpoint, 48
shards). Requires both nodes: it does not fit in one GB10's 121 GiB.

## TL;DR

**Winner: the fork stack (this repo), row F2** — fixed proposer guard +
`VLLM_DSPARK_SKIP_RAGGED_SPECULATION=1`, probabilistic drafting,
`MTP_NUM_TOKENS=3`. It is the only stack that serves the real model, and it
serves it **clean under speculation + concurrency** (the bug this evaluation
set out to validate). Left running and smoke-verified.

The upstream lane (`fulvius31/vllm@deepseek-v4-gb10-dspark-tp`) **builds, passes
its unit tests, and runs distributed on this fabric with a tiny model, but
cannot serve the real DeepSeek-V4 model**: its sparse-MLA decode kernel call is
incompatible with the FlashInfer version the branch itself pins. Details below;
this is a branch defect to report upstream, not a rig problem.

## Images / provenance

| artifact | id / ref |
| --- | --- |
| fork runtime | `vllm-dspark-runtime:dspark-nvfp4-stage-c` — `sha256:92309f340aa6…` (identical on both nodes) |
| fork base | `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready` — pulled digest `sha256:d8492e7677cf…` |
| upstream image | `vllm-upstream-gb10:dspark-tp` — `sha256:1dc8b133cc28…` (identical on both nodes) |
| upstream branch | `fulvius31/vllm@deepseek-v4-gb10-dspark-tp` commit `01d3975ab` |
| fork vLLM | `0.21.1rc1.dev339+g1967a5627bc3` · upstream vLLM `v0.1.dev18726+g01d3975ab` |

The rig arrived bare (no `.env.dspark`, no images on either node, no model
cache). All of the above was built/pulled/downloaded during this run; the model
was downloaded once on the head and shipped to the worker over RoCE, and each
image was built on the head and `docker save | ssh docker load`-ed to the
worker so both nodes carry byte-identical IDs.

## Phase 2 — fork stack corruption A/B (the repaired stack)

Soak: `scripts/dspark-corruption-soak.py`, 4 sessions × 12 turns × temp 0.6 for
c=4, single session for c=1. The server is restarted (cold prefix cache) before
**every** soak. `MTP_NUM_TOKENS=3` (spec on) for all rows.

| row | config | c=4 exit | c=4 tok/s | c=1 exit | c=1 tok/s | sampling |
| --- | --- | --- | --- | --- | --- | --- |
| **F1** | fixed guard only (no mitigation knobs) | **0 clean** | 40.4 | **0 clean** | 34.6 | probabilistic |
| **F2** | + `VLLM_DSPARK_SKIP_RAGGED_SPECULATION=1` | **0 clean** | **52.5** | **0 clean** | 36.9 | probabilistic |
| **F3** | + `GREEDY_VERIFICATION=1` (F2 reverted) | **0 clean** | 50.9 | **0 clean** | 36.1 | greedy-forced (temp 0.0) |

**All six soaks clean (exit 0), no corruption signatures.** The headline: **F1
is clean with zero mitigations** — full probabilistic speculation at
concurrency 4, the exact configuration that used to corrupt. So the in-tree
root-cause fix (the proposer guard: guard-fired steps schedule zero speculative
tokens instead of fabricating token-id-0 drafts) holds on its own; the
mitigation ladder is not required for correctness.

F2 adds `SKIP_RAGGED_SPECULATION` (skip speculation only on ragged mixed
prefill/decode steps; uniform decode steps keep full speculation). It is
clean and ~30 % faster at c=4 (52.5 vs 40.4) — the ragged steps it skips are
exactly the expensive/contended ones under concurrency. That makes it the
rule-preferred config: clean, still probabilistic, highest c=4 throughput.

F3 (`GREEDY_VERIFICATION=1`) is clean too but forces `temperature 0.0` /
`top_p 1.0` server-side for every client, so it is discounted by decision
rule 2 (greedy-forced). Its speed (50.9) is essentially F2's.

> Throughput note: the soak's "aggregate throughput" is decode-tokens ÷
> wall-clock-including-prefill over growing 12-turn sessions, so it reads lower
> than a raw single-stream decode rate (measured separately at **56.6 tok/s**,
> within the documented 52–57 band). It is a consistent yardstick across rows
> and stacks, not a peak decode number.

## Phase 3 — upstream lane build & gates (all passed)

1. **Unit tests in-image** (`tests/v1/spec_decode/test_dspark_local_argmax.py`):
   **11/11 passed** — first real execution of the branch's tests.
   (Harness note: the runbook command installs only `pytest`; `conftest.py`
   also imports `tblib` — add it or collection fails.)
2. **Platform:** torch 2.11.0+cu130, CUDA 13.0, device capability **(12, 1)**,
   FlashInfer 0.6.13 imports. Tiny model (`Qwen2.5-0.5B`) serves + chats single
   node.
3. **Image shipped:** identical ID `1dc8b133cc28` on both nodes.
4. **Tiny model, two nodes, TP=2, `NCCL_DEBUG=INFO`:** both nodes report
   `NET/IB : Using [0]rocep1s0f0:1/RoCE`, `Using network IB`, all channels
   `via NET/IB/0`, **zero `NET/Socket` lines** — RoCE confirmed, no TCP
   fallback. Two-node chat returned correct output.

## Phase 4 — upstream real model: BLOCKED (cannot serve)

With the real DeepSeek-V4 model, every upstream row crashes at the first decode
step:

```
TypeError: trtllm_batch_decode_sparse_mla_dsv4() got an unexpected
keyword argument 'swa_topk_lens'
  at vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py:769 (_forward_decode)
```

Root cause — the branch's DSV4 sparse-MLA decode code is **ahead of the
FlashInfer it pins**:

| | branch call site (`flashinfer_sparse.py:769`) | installed flashinfer **0.6.13** |
| --- | --- | --- |
| topk lengths kwarg | `swa_topk_lens=` | `sparse_topk_lens` |
| compressed dual-cache | `extra_sparse_indices=`, `extra_sparse_topk_lens=` | **absent** |
| required arg | (not passed) | `seq_lens` **required** |

The branch pins its own FlashInfer: `docker/versions.json`
`FLASHINFER_VERSION.default = 0.6.13`, Dockerfile `ARG FLASHINFER_VERSION=0.6.13`;
the image carries `flashinfer-{python,cubin,jit-cache} == 0.6.13`. The build is
correct — the model code (advanced by the branch's merge of upstream main / the
SM12x DeepSeek-V4 work) simply requires a newer FlashInfer than the branch
declares. 0.6.13 has no compressed/`extra_sparse_*` path at all.

Why the tiny-model gates passed: `Qwen2.5-0.5B` does not use the DSV4 sparse-MLA
kernel, so the mismatch only surfaces with the real model.

**Not remediated, deliberately.** Per the mission's ground rule ("don't
thrash") and the runbook's explicit warning not to override the pinned
FlashInfer (the SM120 sparse-MLA kernels ship with that wheel), bumping
FlashInfer is a multi-hour rebuild that would most likely trade this error for a
missing-SM120-kernel error, and 0.6.13 cannot be back-patched onto (no
compressed path). This should be filed against `fulvius31/vllm`: pin a
FlashInfer that matches the DSV4 decode signature, or gate the `extra_sparse_*`
path behind a `has_flashinfer_*` capability check.

Evidence: `bench-results/upstream-BLOCKER-flashinfer-mismatch.txt`,
`bench-results/upstream-U1-head-crash.log`.

## Phase 5 — decision

Applying the decision rule in order:

1. **Discard dirty c=4.** Upstream U1/U2/U3 cannot produce output at all →
   discarded. Fork F1/F2/F3 all clean at c=4 (and c=1).
2. **Prefer probabilistic over greedy-forced.** F1, F2 probabilistic; F3
   greedy-forced → keep F1, F2.
3. **Highest c=4 tok/s.** F2 (52.5) > F1 (40.4).

**Winner: fork stack, row F2.** (Rule 4 — prefer upstream on a tie — never
applies, because upstream is not serviceable on this model.)

Winning configuration, now running:

```
image  vllm-dspark-runtime:dspark-nvfp4-stage-c
spec   MTP_NUM_TOKENS=3, draft_sample_method=probabilistic
knobs  VLLM_DSPARK_SKIP_RAGGED_SPECULATION=1, GREEDY_VERIFICATION=0
serve  --kv-cache-dtype nvfp4_ds_mla, TP=2, max-model-len 1048576,
       gpu-memory-utilization 0.85, HF_HUB_OFFLINE=1
```

## Repairs made to get here (all pre-existing defects unless noted)

The rig could not run either stack two-node as delivered. Fixes applied:

1. **`GLOO_SOCKET_IFNAME` unset (both composes).** Torch's Gloo CPU process
   group ignores `NCCL_SOCKET_IFNAME`; left to choose it bound `enP7s7` (the
   wired-ethernet port, DOWN with no address on this wifi-only rig) and the
   worker died with `Unable to find address for: enP7s7`. This blocked **both**
   stacks two-node — very likely why the fork's historical runs worked (wired
   ethernet was plugged in then). Pinned Gloo to the fabric NIC in
   `docker-compose.dspark.yml` and `docker-compose.upstream.yml`.
2. **Runbook `--speculative-config "${…}"` → invalid JSON.** Under `bash -lc`
   the double quotes let bash strip the JSON's own quotes; vLLM received
   `{method:dspark,…}`. Would have failed every upstream row. Single-quoted it
   in the compose and the runbook.
3. **`HF_HUB_OFFLINE=0`** made every start hit huggingface.co even though the
   model is fully cached; when the worker's wifi flapped, DNS failed and the
   worker died. Set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` — the stack
   now needs only the RoCE link.
4. **Runbook `num_speculative_tokens:3` invalid on the branch.** The checkpoint
   declares `dspark_block_size=5` and the branch rejects spec depth < block
   size ("Smaller values produce incorrect output"). Upstream rows were set to
   5; the runbook examples were corrected. (This means the intended comparison
   would have been fork@3 vs upstream@5 — an asymmetry, but moot given the
   upstream blocker.)
5. **`bench-results/notes.md`** carries the full running log, including one
   self-inflicted harness bug (a `local a=$1 b=$2 c=…$b` unbound-variable under
   `set -u`) fixed in both row runners.

## Anomalies / caveats

- Upstream throughput/acceptance were never measured — the stack cannot serve
  the model. The fork's acceptance was not separately grepped; the soak's tok/s
  is the throughput meter used throughout.
- Each soak is a single run; corruption is stochastic. Confidence comes from
  6/6 clean fork soaks across the ladder, not any single row.
- The upstream build was restarted once after a git-clone TCP stall (35 min of
  zero progress while starved behind the model download; a fresh clone of the
  same repo took 20 s). BuildKit cache made the restart free.
