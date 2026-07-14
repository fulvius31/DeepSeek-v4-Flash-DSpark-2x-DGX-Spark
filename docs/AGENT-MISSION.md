# Agent mission: repair, validate, and pick the best DSpark stack on 2× DGX Spark

You are an autonomous coding agent running **on the head (master) DGX Spark**.
You can reach the **worker Spark over SSH** (key-based, no password). Your job
is to execute this mission end-to-end without asking the operator questions,
and to leave behind (a) a serving endpoint in the best known-good
configuration and (b) a committed report with every measurement.

Read this whole file before running anything.

## Environment facts

- Two DGX Sparks (GB10, aarch64, 128 GB unified each), TP=2 over a 200 Gb
  RoCE link. One GPU per node — **both stacks need both nodes; only one stack
  can run at a time** (same GPUs, same port 8888).
- This repo (the deployment recipe) lives on the head node; find it with
  `ls ~/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark || find ~ -maxdepth 3 -name
  docker-compose.dspark.yml`. `cd` into it; everything below is relative to it.
- `.env.dspark` holds cluster values. Read it first. `WORKER_HOST` is the
  worker's SSH address; `MASTER_ADDR`/`VLLM_HOST_IP`/`WORKER_VLLM_HOST_IP` are
  fabric IPs; NCCL_* are load-bearing.
- **Network identity of this rig** (validated in
  <https://contact.alessandrosangiorgi.net/posts/dgx-spark-roce-link-validation/>):
  - SSH user on both Sparks: `alessangiorgi`. Worker SSH target:
    `alessangiorgi@10.0.0.2` (over the point-to-point RoCE link; ssh omits the
    user when you are already logged in as `alessangiorgi`).
  - Head (master) fabric IP: `10.0.0.1`; worker fabric IP: `10.0.0.2`
    (`10.0.0.0/30` link subnet, MTU **9000**).
  - RDMA device: **`rocep1s0f0`**; its Ethernet netdev: **`enp1s0f0np0`**
    (the Spark exposes four RoCE devices across two PCI paths, `p1s0` and
    `P2p1s0`, dual-port each — the cabled/ACTIVE port on this rig is
    `p1s0f0`). So: `NCCL_IB_HCA=rocep1s0f0`,
    `NCCL_SOCKET_IFNAME=enp1s0f0np0`. If a NCCL start ever reports
    `NET/Socket`, verify the ACTIVE port first (`ip addr show enp1s0f0np0`,
    link state + 10.0.0.x address present on BOTH nodes) — a moved cable
    changes the `f0`/`f1` suffix.
  - Validated raw-link reference: ~109 Gbps bandwidth, ~1.45 µs latency. If a
    perftest shows far less, fix the link before benchmarking anything.
  - Extra NCCL env the link validation used (apply when diagnosing transport
    problems): `NCCL_NET_PLUGIN=none`, `NCCL_IB_MERGE_NICS=1`.
- Worker commands: `ssh alessangiorgi@10.0.0.2 '<cmd>'` (i.e.
  `ssh $WORKER_HOST ...`). The repo's own scripts already rsync/ssh to the
  worker (see `build-dspark-vllm-runtime.sh`,
  `start-deepseek-v4-flash-dspark.sh`) — prefer them over hand-rolled loops.
- Two stacks exist:
  1. **Fork stack** (this repo): overlay on the `unholy-fusion` image,
     launched by `./start-deepseek-v4-flash-dspark.sh`. Historical reference:
     52–57 tok/s single-stream, acceptance ≈ 0.60.
  2. **Upstream lane**: vLLM branch `deepseek-v4-gb10-dspark-tp` at
     `https://github.com/fulvius31/vllm` — upstream main + two TP draft-loop
     optimizations. Build/run per `docs/UPSTREAM-VLLM-GB10-BRANCH.md`.
- Known bug being validated: **speculation + concurrency output corruption**
  (README section "Known issue"). Root cause fixed in-tree (proposer guard
  fabricated token-id-0 drafts). Mitigation ladder: fixed guard (always on) →
  `VLLM_DSPARK_SKIP_RAGGED_SPECULATION=1` → `GREEDY_VERIFICATION=1` →
  `MTP_NUM_TOKENS=0`.
- The measurement tool is `scripts/dspark-corruption-soak.py` (stdlib-only).
  Exit 0 = clean, 2 = corruption (prints signatures), 1 = run error. It also
  reports aggregate tok/s — use it as the throughput meter too
  (`--concurrency 1` ≈ single-stream).

## Ground rules

1. **One stack at a time.** Always `./stop-deepseek-v4-flash-dspark.sh` (and
   `docker compose -f docker-compose.upstream.yml down` on BOTH nodes) before
   starting the other stack.
2. **Worker first** on every launch, exactly as the scripts/runbook do.
3. **Identical images on both nodes** before any two-node start
   (`docker images --digests` on both; ship with
   `docker save <img> | ssh $WORKER_HOST docker load`).
4. **Fresh judgment after every config change:** restart the server (prefix
   cache reset) before running the soak; the soak builds fresh client
   sessions itself. Never judge corruption from a session that already
   contains leaked markers.
5. **Long jobs** (image builds are multi-hour; 284B model load is ~5–10 min):
   run under `nohup ... &` or tmux, poll logs; do not sit in a blocking shell
   that can time out. Never kill a build because it is slow.
6. If a phase's gate fails, **stop escalating, capture diagnostics** (last
   200 log lines from both nodes, exact command, env), record them in the
   report, and move to the next phase only if it does not depend on the
   failed one. Do not thrash-retry more than twice per gate.
7. Do not delete the HF model cache, do not modify `.env.dspark` fabric
   values, do not push to any remote other than `origin` of this repo.
8. Record every measurement immediately in `bench-results/` (gitignored
   scratch) and assemble the final report at the end.

## Phase 0 — recon (gate: all checks pass)

```bash
cd <repo>; git pull --ff-only            # must include commit af15f22 or later
set -a; source .env.dspark; set +a
ssh $WORKER_HOST 'hostname && nvidia-smi -L && df -h / | tail -1'
nvidia-smi -L && df -h / | tail -1       # need ~80 GB free on head for builds
./status-deepseek-v4-flash-dspark.sh     # note what is currently running
ls $HF_CACHE/hub | grep -i dspark        # model cache present
systemctl is-active earlyoom && echo "WARN: earlyoom active (README says disable)"
mkdir -p bench-results
```

## Phase 1 — deploy the corruption fix on the fork stack

The fix ships in this repo (proposer guard + compose knobs). The bind-mounted
proposer is synced by the start script, but the **baked overlay copy needs an
image rebuild** (fast — COPY layers only, no compilation):

```bash
./stop-deepseek-v4-flash-dspark.sh
./build-dspark-vllm-runtime.sh           # builds head AND worker (rsyncs repo)
./start-deepseek-v4-flash-dspark.sh      # worker-first, waits for API
./smoke-deepseek-v4-flash-dspark.sh      # gate: passes
```

## Phase 2 — fork-stack corruption A/B (the repaired stack)

Run each row; restart the service between rows. Save each soak's full output
to `bench-results/fork-<row>.txt`.

| row | env change (in `.env.dspark`, then restart) | expect |
| --- | --- | --- |
| F1 baseline | none (fixed guard only) | measure; may still be dirty if other misalignment exists |
| F2 | `VLLM_DSPARK_SKIP_RAGGED_SPECULATION=1` | cleaner or clean |
| F3 | `GREEDY_VERIFICATION=1` (with F2 reverted) | expected clean (historical temp-0 profile) |
| F4 only if F1–F3 all dirty | `MTP_NUM_TOKENS=0` | clean, slow (~18 tok/s) |

Soak command (same for every row; also run once with `--concurrency 1` for a
single-stream throughput number):

```bash
python3 scripts/dspark-corruption-soak.py \
  --base-url http://127.0.0.1:8888/v1 --model deepseek-v4-flash-dspark \
  --concurrency 4 --turns 12 --temperature 0.6 --label fork-<row>
```

Gate: at least one row ≤ F3 is clean (exit 0). Record tok/s for every row.
Note: F3 forces temperature 0.0 server-side, so its soak runs effectively
greedy — that is the point; still pass `--temperature 0.6` for identical
client behavior across rows.

## Phase 3 — build the upstream lane

Follow `docs/UPSTREAM-VLLM-GB10-BRANCH.md` exactly; condensed:

```bash
git clone --branch deepseek-v4-gb10-dspark-tp \
  https://github.com/fulvius31/vllm.git ~/vllm-gb10 && cd ~/vllm-gb10
nohup docker build -f docker/Dockerfile --target vllm-openai \
  --build-arg torch_cuda_arch_list='12.0f' \
  --build-arg max_jobs=16 --build-arg nvcc_threads=2 \
  -t vllm-upstream-gb10:dspark-tp . > ~/build-upstream.log 2>&1 &
# poll: tail -5 ~/build-upstream.log  (multi-hour; keep working on Phase 2 rows meanwhile)
```

After the build, in order (each is a gate):

1. Unit tests in-image (first real execution of the branch's tests):
   `docker run --rm --gpus all --entrypoint bash -v ~/vllm-gb10/tests:/wd/tests
   vllm-upstream-gb10:dspark-tp -c "cd /wd && (python3 -m pip install -q
   pytest || uv pip install --system -q pytest) && python3 -m pytest
   tests/v1/spec_decode/test_dspark_local_argmax.py -v"` — all pass.
2. Tiny model, single node (platform gate — see runbook step 3).
3. Ship image: `docker save vllm-upstream-gb10:dspark-tp | ssh $WORKER_HOST docker load`.
4. Tiny model, two nodes, TP=2, `NCCL_DEBUG=INFO` — gate: logs show IB/RoCE
   transport, not sockets (runbook "First-time bring-up" step 5).

## Phase 4 — upstream real model

**Stop the fork stack on both nodes first.** Use the compose file from the
runbook (`docker-compose.upstream.yml`, both nodes), `MAX_MODEL_LEN=200000`.

| row | SPECULATIVE_CONFIG | purpose |
| --- | --- | --- |
| U1 | `{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}` | stock upstream baseline |
| U2 | U1 + `"replicate_markov_w1":true` | branch feature 1 (acceptance must be unchanged) |
| U3 | `{"method":"dspark","num_speculative_tokens":3,"replicate_markov_w1":true,"use_local_argmax_reduction":true}` | branch feature 2 (greedy drafting) |

For each row: worker-first start, `curl /v1/models`, smoke, then the soak at
`--concurrency 4` and `--concurrency 1`, saving to `bench-results/upstream-<row>.txt`.
Grep acceptance from logs: `docker compose -f docker-compose.upstream.yml logs
| grep -iE "accept|spec"` (record whatever metric lines appear).

Gate for adoption: **U1 or U2 soak exits 0 at concurrency 4 with speculation
on.** That is the test the fork failed before the fix.

## Phase 5 — decision and report

Decision rule, in order:
1. Discard any configuration whose concurrency-4 soak is dirty.
2. Among clean ones, prefer probabilistic sampling over greedy-forced.
3. Then highest concurrency-4 aggregate tok/s; tie-break on single-stream.
4. If upstream and fork tie on all of the above, prefer upstream
   (maintainability: rebase-able, no frozen binary base image).

Write `docs/EVALUATION-RESULTS.md` containing: date, image tags/digests,
every row's soak exit code + tok/s (c=1 and c=4) + acceptance, log excerpts
for any corruption, the decision per the rule above, and anything anomalous.
Then:

```bash
git add docs/EVALUATION-RESULTS.md && git commit -m "Add 2x Spark stack evaluation results" && git push origin main
```

Finally, **leave the winning configuration running** (worker-first), confirm
`./smoke-deepseek-v4-flash-dspark.sh` (or a curl smoke against the upstream
compose) passes, and state the endpoint + config in your final message.

## Failure triage quick-reference

- Fork build/start fails → `./logs-deepseek-v4-flash-dspark.sh`, check both
  nodes; commonest: image tag mismatch between nodes, stale env.
- Upstream build fails → check `~/build-upstream.log`; try
  `torch_cuda_arch_list='12.1'` if `12.0f` is rejected by the base CUDA;
  reduce `max_jobs` on OOM.
- Two-node start hangs → NCCL: rerun with `NCCL_DEBUG=INFO`; if you see
  `NET/Socket`, the RDMA env is wrong (`NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`,
  `/dev/infiniband` passthrough) — fix before benchmarking anything.
- Upstream startup ValueError about `use_local_argmax_reduction` +
  probabilistic → expected by design; that combination is invalid (U3 uses
  greedy drafting).
- Model load OOM at 200K → lower `GPU_MEMORY_UTILIZATION` to 0.78, retry once.
- Soak exit 1 (run error) → the server died or timed out; grab logs before
  restarting; a worker OOM kill with earlyoom active is the usual suspect.
