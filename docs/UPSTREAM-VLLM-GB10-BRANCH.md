# Running the upstream-vLLM branch (`fulvius31/vllm@deepseek-v4-gb10-dspark-tp`) on 2× DGX Spark

This runbook stands up **upstream vLLM** (main `7fc97042c` + our TP draft-loop
optimizations) on the two Sparks, as an alternative lane to this repo's
unholy-fusion overlay stack. The fork stack stays untouched; you can switch
back at any time.

Branch: <https://github.com/fulvius31/vllm/tree/deepseek-v4-gb10-dspark-tp>

## What the branch adds on top of upstream main

Upstream DSpark drafting pays, per draft block of N speculative tokens,
`1 + 2N` TP collectives (`1 + N` of them full-vocab) — every one crossing the
200 Gb RoCE link on this topology. The branch adds two default-off options:

| option (in `--speculative-config` JSON) | effect |
| --- | --- |
| `"replicate_markov_w1": true` | Full Markov-W1 copy per rank (~66 MB): the per-step embedding lookup becomes local, removing N all-reduces. Works with greedy **and** probabilistic drafting. |
| `"use_local_argmax_reduction": true` | Greedy drafting only: base logits and the per-step Markov bias stay vocab-sharded; only `[B, 2]` (value, index) pairs cross TP. Removes the `1 + N` full-vocab gathers. Also fixes an upstream startup crash when this documented flag was set with DSpark. |

With both on (greedy drafting): **zero vocab-scale collectives in the draft loop**.

## How upstream replaces the fork's pieces

| fork stack | upstream equivalent |
| --- | --- |
| `nvfp4_ds_mla` KV (Stage A/B/C) | `fp8_ds_mla` — byte-identical: the shipped Stage C stores fp8 numerics in the same 584 B/token envelope |
| B12X TileLang MoE (`VLLM_USE_B12X_*`) | `flashinfer_b12x` CuteDSL kernels (auto/kernel-config selected on SM12x) |
| `sm120.py` overlay attention | `DeepseekV4FlashInferSM120Attention` (auto-selected; needs FlashInfer with SM120 sparse-MLA kernels — pinned in the image) |
| Keys proposer patches + bind-mount | different upstream architecture (dedicated `DSparkSpeculator`, paged draft KV) — concurrency must be re-validated, nothing to port blindly |
| `VLLM_DSPARK_*` env vars | gone; the two new speculative-config fields above |
| model-runner selection | automatic: `method: dspark` force-selects the V2 GPU runner |

## Prerequisites

- **Build on a Spark, not the x86 workstation** — the Sparks are aarch64/sbsa.
  Expect a multi-hour first build and ~60 GB free for image + build cache.
- Your existing `.env.dspark` fabric values (NCCL_IB_HCA, NCCL_SOCKET_IFNAME,
  NCCL_IB_GID_INDEX, MASTER_ADDR, node IPs) carry over unchanged.
- The HF model cache from `prepare-dspark-model-cache.sh` is reused as-is.
- earlyoom disabled on both hosts (same recommendation as the main README).
- **Stop the fork stack first** (`./stop-deepseek-v4-flash-dspark.sh`) — both
  stacks need the same GPUs and port 8888.

## Step 1 — clone the branch (on the head Spark)

```bash
git clone --branch deepseek-v4-gb10-dspark-tp --depth 50 \
  https://github.com/fulvius31/vllm.git ~/vllm-gb10
cd ~/vllm-gb10
git log --oneline -1   # expect: c7032bf59 [Spec Decode] DSpark: cut TP collectives...
```

## Step 2 — build the image (on the head Spark)

`12.0f` is the CUDA family target that covers GB10 (sm_121).

```bash
cd ~/vllm-gb10
DOCKER_BUILDKIT=1 docker build \
  -f docker/Dockerfile \
  --target vllm-openai \
  --build-arg torch_cuda_arch_list='12.0f' \
  --build-arg max_jobs=16 \
  --build-arg nvcc_threads=2 \
  -t vllm-upstream-gb10:dspark-tp .
```

Notes:
- `max_jobs`/`nvcc_threads` tuned for the 20-core Grace CPU and 128 GB unified
  memory — raise cautiously; nvcc under parallel load is RAM-hungry.
- The image installs a pinned `flashinfer-jit-cache` wheel (see
  `docker/versions.json`); the SM120 sparse-MLA kernels ship with it.

## Step 3 — get the identical image onto the worker

Same rule as the fork stack: **bit-identical images on both nodes.**

```bash
docker save vllm-upstream-gb10:dspark-tp | \
  ssh <WORKER_HOST> docker load
```

(Or push to a registry both nodes can reach. Avoid rebuilding independently —
that's how the `vllm-dspark-runtime:clean` provenance incident happened.)

## Step 4 — compose file

Save as `docker-compose.upstream.yml` on **both** nodes (or rsync it). It
reuses your `.env.dspark` variables.

```yaml
services:
  vllm-upstream:
    image: vllm-upstream-gb10:dspark-tp
    network_mode: host
    ipc: host
    shm_size: "64gb"
    ulimits:
      memlock: -1
      stack: 67108864
    gpus: all
    devices:
      - /dev/infiniband:/dev/infiniband
    volumes:
      - ${HF_CACHE:-${HOME}/.cache/huggingface}:/cache/huggingface
    environment:
      HF_HOME: /cache/huggingface
      HF_HUB_OFFLINE: "1"
      VLLM_HOST_IP: "${VLLM_HOST_IP:-}"
      VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"
      FLASHINFER_CUDA_ARCH_LIST: "12.1a"
      PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
      NCCL_NET: "IB"
      NCCL_IB_DISABLE: "0"
      NCCL_IB_HCA: "${NCCL_IB_HCA}"
      NCCL_SOCKET_IFNAME: "${NCCL_SOCKET_IFNAME}"
      NCCL_IB_GID_INDEX: "${NCCL_IB_GID_INDEX:-}"
      NCCL_CROSS_NIC: "1"
      NCCL_CUMEM_ENABLE: "0"
      NCCL_DEBUG: "${NCCL_DEBUG:-WARN}"
      NCCL_NVLS_ENABLE: "0"
    entrypoint: ["bash", "-lc"]
    command:
      - >
        exec vllm serve ${DSPARK_MODEL:-deepseek-ai/DeepSeek-V4-Flash-DSpark}
        --served-model-name deepseek-v4-flash-dspark
        --host ${VLLM_HOST:-127.0.0.1}
        --port 8888
        --trust-remote-code
        --tensor-parallel-size 2
        --pipeline-parallel-size 1
        --kv-cache-dtype fp8_ds_mla
        --max-model-len ${MAX_MODEL_LEN:-200000}
        --max-num-seqs ${MAX_NUM_SEQS:-6}
        --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS:-8192}
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.80}
        --enable-prefix-caching
        --speculative-config "${SPECULATIVE_CONFIG}"
        --tokenizer-mode deepseek_v4
        --tool-call-parser deepseek_v4
        --enable-auto-tool-choice
        --reasoning-parser deepseek_v4
        --default-chat-template-kwargs '{"thinking":false}'
        --generation-config vllm
        --distributed-executor-backend mp
        --nnodes 2
        --node-rank ${NODE_RANK}
        --master-addr ${MASTER_ADDR}
        --master-port ${MASTER_PORT:-25000}
        ${HEADLESS:+--headless}
```

Two speculative profiles (set `SPECULATIVE_CONFIG` in the environment or
`.env.dspark`):

```bash
# Profile A — probabilistic drafting (closest to the fork's C12 profile).
# Only W1 replication applies (local argmax is greedy-only by design).
SPECULATIVE_CONFIG='{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic","replicate_markov_w1":true}'

# Profile B — greedy drafting, both optimizations (zero vocab-scale
# collectives in the draft loop):
SPECULATIVE_CONFIG='{"method":"dspark","num_speculative_tokens":3,"replicate_markov_w1":true,"use_local_argmax_reduction":true}'
```

For baseline A/B runs, drop the extra fields:
`'{"method":"dspark","num_speculative_tokens":3}'` (add
`"draft_sample_method":"probabilistic"` for the probabilistic baseline).

## Step 5 — launch (worker first, same as the fork stack)

On the **worker** Spark:

```bash
NODE_RANK=1 HEADLESS=1 VLLM_HOST_IP=<worker-fabric-ip> \
docker compose --env-file .env.dspark -f docker-compose.upstream.yml up -d
```

Then on the **head** Spark:

```bash
NODE_RANK=0 VLLM_HOST_IP=<head-fabric-ip> \
docker compose --env-file .env.dspark -f docker-compose.upstream.yml up -d
docker compose -f docker-compose.upstream.yml logs -f
```

Wait for `Application startup complete`, then:

```bash
curl -fsS http://127.0.0.1:8888/v1/models   # expect the served model + max_model_len
./smoke-deepseek-v4-flash-dspark.sh          # the repo's smoke test works unchanged
```

## First-time bring-up (do this ONCE, in this order)

The full model takes many minutes per boot attempt; a tiny model boots in
seconds. Isolate each variable before spending time on the big one.

1. **Prechecks** (both nodes): fork stack stopped, `nvidia-smi` healthy,
   fabric IPs ping, `/dev/infiniband` present, earlyoom disabled, ~80 GB free
   on the head for the build, model cache present.
2. **Build** (Step 2), then run the branch's unit tests **inside the image** —
   their first real execution (the dev workstation has no torch):

   ```bash
   docker run --rm --gpus all --entrypoint bash \
     -v ~/vllm-gb10/tests:/wd/tests vllm-upstream-gb10:dspark-tp -c \
     "cd /wd && (python3 -m pip install -q pytest || uv pip install --system -q pytest) \
      && python3 -m pytest tests/v1/spec_decode/test_dspark_local_argmax.py -v"
   ```
3. **Tiny-model, single node** — validates image/arch/kernels on GB10 with no
   distributed variables (download e.g. `Qwen/Qwen2.5-0.5B-Instruct` while the
   image builds):

   ```bash
   docker run --rm --gpus all --network host \
     -v $HF_CACHE:/cache/huggingface -e HF_HOME=/cache/huggingface \
     vllm-upstream-gb10:dspark-tp Qwen/Qwen2.5-0.5B-Instruct --port 8899
   curl -fsS http://127.0.0.1:8899/v1/models
   ```

   Failure here = platform problem (check `torch.cuda.get_device_capability()`
   returns `(12, 1)` in the container; check FlashInfer import).
4. **Ship the image to the worker** (Step 3); confirm the same image ID on
   both nodes (`docker images --digests`).
5. **Tiny-model, TWO nodes, TP=2** — validates NCCL-over-RoCE and the
   multi-node `mp` machinery in seconds per attempt. Use the Step 4 compose
   with `DSPARK_MODEL=Qwen/Qwen2.5-0.5B-Instruct`, `SPECULATIVE_CONFIG` unset
   (delete the `--speculative-config` line for this test), `MAX_MODEL_LEN=4096`,
   and `NCCL_DEBUG=INFO`. Worker first, then head. In the logs confirm NCCL
   reports an **IB/RoCE transport, not sockets**. Failure here = fabric/env
   problem (`NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `VLLM_HOST_IP` per node,
   master addr/port reachability).
6. **Real model, baseline profile** — `MAX_MODEL_LEN=200000`,
   `SPECULATIVE_CONFIG='{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'`
   (no new flags yet). Watch boot logs for the attention backend line (SM120
   sparse MLA), the KV pool size, and `Application startup complete`. Then
   `curl /v1/models` + the repo smoke test.
7. **Only now enable the branch features**, one at a time, per the validation
   matrix below. Record tok/s + acceptance at every step; the fork's 52–57
   tok/s single-stream is the reference.

## Step 6 — validation matrix

Run the same gates the fork stack was validated with, in this order:

1. **Baseline** (no new fields): single-stream tok/s + acceptance from the logs;
   compare against the fork's 52–57 tok/s and ~0.60 acceptance.
2. **`replicate_markov_w1: true`** (Profile A): acceptance must be unchanged
   (the replicated lookup is numerically identical); tok/s should tick up.
3. **Profile B** (greedy + both): acceptance comparable to the greedy baseline;
   biggest per-step latency win expected here.
4. **Concurrency**: 2/4/6 concurrent smoke prompts, then staggered arrivals —
   this exercises the code paths where the fork needed the Keys patches;
   upstream's architecture is different, so watch for crashes or acceptance
   collapse and report either upstream.
4b. **Corruption soak — the decisive gate.** The fork stack has a known
   speculation+concurrency corruption bug (README "Known issue"). Upstream
   reimplements that machinery entirely (paged draft KV, padded static
   batches, no hand-rolled ring buffers), so this test is what proves whether
   the bug class is actually gone:

   ```bash
   # Run against the upstream endpoint at production-like temperature.
   python3 scripts/dspark-corruption-soak.py \
     --base-url http://<head>:8888/v1 --model deepseek-v4-flash-dspark \
     --concurrency 4 --turns 12 --temperature 0.6 --label upstream-spec-on
   ```

   Exit 0 = clean, exit 2 = corruption detected (prints the signatures).
   Run the same command against the fork stack for the A/B. Restart the
   server between runs (prefix-cache reset); the harness always builds fresh
   client sessions, so a healthy server is never blamed for a poisoned
   client history.
5. **Collective count sanity**: rerun once with `NCCL_DEBUG=INFO` and confirm
   the per-step all-gather traffic drops between baseline and Profile B.
6. Only after all of the above: raise `MAX_MODEL_LEN` toward 1048576 (keep
   `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`) and re-check boot KV pool and headroom.

## Troubleshooting

- **Startup `ValueError: use_local_argmax_reduction is not compatible with
  draft_sample_method='probabilistic'`** — by design; use Profile A for
  probabilistic drafting.
- **`FLASHINFER_MLA_SPARSE_DSV4 SM120 requires FlashInfer's ...`** — the image's
  FlashInfer lacks the SM120 sparse-MLA kernels; rebuild with the pinned
  `docker/versions.json` FlashInfer (don't override its version).
- **Attention backend auto-selection picks something else** — force it with
  `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4` in the environment.
- **NCCL falls back to sockets** (decode ~3–5 tok/s): check
  `NCCL_DEBUG=INFO` for the transport line; verify `NCCL_IB_HCA`,
  `NCCL_IB_GID_INDEX`, and `/dev/infiniband` passthrough. This is the
  TCP-death mode — the fabric settings are load-bearing.
- **OOM-killed workers under load** — disable earlyoom on both hosts
  (`sudo systemctl stop earlyoom && sudo systemctl disable earlyoom`).
- **Model resolves to a download** — set `DSPARK_MODEL` to the exact repo id
  cached by `prepare-dspark-model-cache.sh` (check with
  `ls $HF_CACHE/hub/ | grep -i dspark`) and keep `HF_HUB_OFFLINE=1`.

## Rollback

The fork stack is untouched: `docker compose -f docker-compose.upstream.yml
down` on both nodes, then `./start-deepseek-v4-flash-dspark.sh` as before.

## Keeping the branch fresh

The branch is additive and default-off by construction (model-isolated files +
one config field + one guarded speculator branch), so tracking upstream is:

```bash
cd ~/vllm-gb10
git fetch https://github.com/vllm-project/vllm.git main
git rebase FETCH_HEAD          # conflicts, if any, will be small and local
docker build ...               # rebuild as in Step 2
```
