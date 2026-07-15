#!/usr/bin/env bash
# Decisive test: route DSV4 SM120 decode through the grafted flashinfer
# BatchSparseMLAPagedAttentionWrapper (the fork's actual multi-query path) and
# run the real model WITH speculation.
set -uo pipefail
REPO=/home/alessangiorgi/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
cd "$REPO"; set -a; source .env.dspark; set +a
W=alessangiorgi@10.0.0.2; CF=docker-compose.upstream.yml; PROJ=upstream
export UPSTREAM_IMAGE=vllm-upstream-gb10:dspark-tp-fisparse
export MAX_MODEL_LEN=200000
export VLLM_DSV4_SM120_WRAPPER_DECODE=1
SPEC='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'

echo "[$(date +%H:%M:%S)] stop fork winner + any upstream"
./stop-deepseek-v4-flash-dspark.sh >/dev/null 2>&1 || true
COMPOSE_DISABLE_ENV_FILE=1 SPECULATIVE_CONFIG='{}' docker compose -p "$PROJ" --env-file .env.dspark -f "$CF" down >/dev/null 2>&1 || true
ssh "$W" "cd '$REPO' && COMPOSE_DISABLE_ENV_FILE=1 SPECULATIVE_CONFIG='{}' docker compose -p '$PROJ' --env-file .env.dspark -f '$CF' down" >/dev/null 2>&1 || true
scp -q "$CF" .env.dspark "$W:$REPO/"

echo "[$(date +%H:%M:%S)] worker up (rank1, sm120 wrapper decode, spec=5)"
ssh "$W" "cd '$REPO' && COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=1 HEADLESS=1 VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' MAX_MODEL_LEN=200000 UPSTREAM_IMAGE='$UPSTREAM_IMAGE' VLLM_DSV4_SM120_WRAPPER_DECODE=1 SPECULATIVE_CONFIG='$SPEC' docker compose -p '$PROJ' --env-file .env.dspark -f '$CF' up -d" 2>&1 | tail -1
echo "[$(date +%H:%M:%S)] head up (rank0)"
COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 HEADLESS= VLLM_HOST_IP="$VLLM_HOST_IP" MAX_MODEL_LEN=200000 UPSTREAM_IMAGE="$UPSTREAM_IMAGE" VLLM_DSV4_SM120_WRAPPER_DECODE=1 SPECULATIVE_CONFIG="$SPEC" docker compose -p "$PROJ" --env-file .env.dspark -f "$CF" up -d 2>&1 | tail -1

for i in $(seq 1 70); do
  if docker ps -a --format '{{.Status}}' --filter name=upstream | grep -q Exited; then
    echo "HEAD EXITED $(date +%H:%M:%S):"
    docker logs upstream-vllm-upstream-1 2>&1 | grep -oE 'num_tokens > 64[^,]*|Check failed[^,]*|RuntimeError:[^,]*|AssertionError[^,]*|nvcc[^,]*|compile[^,]*error|gen_sparse_mla[^,]*|Error[^,]*' | tail -5
    exit 1
  fi
  if curl -fsS --max-time 4 http://127.0.0.1:8888/v1/models >/dev/null 2>&1; then
    echo "=== SM120-WRAPPER SPEC API UP $(date +%H:%M:%S) ==="
    docker logs upstream-vllm-upstream-1 2>&1 | grep -oiE 'sparse-MLA WRAPPER decode enabled' | head -1 | sed 's/^/  /'
    R=$(curl -s --max-time 120 http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"deepseek-v4-flash-dspark","messages":[{"role":"user","content":"Write a haiku about GPUs, then explain it in one sentence."}],"max_tokens":120,"temperature":0.6}')
    if echo "$R" | python3 -c "import json,sys;d=json.load(sys.stdin);print('  REAL SPEC DECODE OK:',repr(d['choices'][0]['message']['content'][:180]))" 2>/dev/null; then
      echo "=== !!! UPSTREAM SPECULATIVE DECODE WORKS ($(date +%H:%M:%S)) !!! ==="; exit 0
    else
      echo "  decode returned no choices. raw: $(echo "$R" | head -c 160)"; exit 2
    fi
  fi
  [ $i -eq 24 ] && echo "[$(date +%H:%M:%S)] mid: $(docker logs upstream-vllm-upstream-1 2>&1 | grep -oE 'gen_sparse_mla|Compiling|nvcc|WRAPPER decode|shards: *[0-9]+%|num_tokens > 64' | tail -1)"
  sleep 15
done
echo "still loading $(date +%H:%M:%S)"
