#!/usr/bin/env bash
# Decisive test: route the SM120 DSV4 decode through the b12x TileLang kernel
# (multi-query capable) and run the real model WITH speculation. Default
# attention backend (SM120 class, now b12x-patched) + VLLM_DSV4_B12X_COMPRESSED_MLA=1.
set -uo pipefail
REPO=/home/alessangiorgi/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
cd "$REPO"; set -a; source .env.dspark; set +a
W=alessangiorgi@10.0.0.2; CF=docker-compose.upstream.yml; PROJ=upstream
export UPSTREAM_IMAGE=vllm-upstream-gb10:dspark-tp-b12x
export MAX_MODEL_LEN=200000
export VLLM_DSV4_B12X_COMPRESSED_MLA=1
export TORCHINDUCTOR_STATIC_CUDA_LAUNCHER=0
export ENFORCE_EAGER=1
SPEC='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'

echo "[$(date +%H:%M:%S)] stop fork winner + any upstream (free GPUs)"
./stop-deepseek-v4-flash-dspark.sh >/dev/null 2>&1 || true
COMPOSE_DISABLE_ENV_FILE=1 SPECULATIVE_CONFIG='{}' docker compose -p "$PROJ" --env-file .env.dspark -f "$CF" down >/dev/null 2>&1 || true
ssh "$W" "cd '$REPO' && COMPOSE_DISABLE_ENV_FILE=1 SPECULATIVE_CONFIG='{}' docker compose -p '$PROJ' --env-file .env.dspark -f '$CF' down" >/dev/null 2>&1 || true
scp -q "$CF" .env.dspark "$W:$REPO/"

echo "[$(date +%H:%M:%S)] worker up (rank1, b12x decode, spec=5)"
ssh "$W" "cd '$REPO' && COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=1 HEADLESS=1 VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' MAX_MODEL_LEN=200000 UPSTREAM_IMAGE='$UPSTREAM_IMAGE' VLLM_DSV4_B12X_COMPRESSED_MLA=1 TORCHINDUCTOR_STATIC_CUDA_LAUNCHER=0 ENFORCE_EAGER=1 SPECULATIVE_CONFIG='$SPEC' docker compose -p '$PROJ' --env-file .env.dspark -f '$CF' up -d" 2>&1 | tail -1
echo "[$(date +%H:%M:%S)] head up (rank0)"
COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 HEADLESS= VLLM_HOST_IP="$VLLM_HOST_IP" MAX_MODEL_LEN=200000 UPSTREAM_IMAGE="$UPSTREAM_IMAGE" VLLM_DSV4_B12X_COMPRESSED_MLA=1 TORCHINDUCTOR_STATIC_CUDA_LAUNCHER=0 ENFORCE_EAGER=1 SPECULATIVE_CONFIG="$SPEC" docker compose -p "$PROJ" --env-file .env.dspark -f "$CF" up -d 2>&1 | tail -1

for i in $(seq 1 60); do
  if docker ps -a --format '{{.Status}}' --filter name=upstream | grep -q Exited; then
    echo "HEAD EXITED $(date +%H:%M:%S):"
    docker logs upstream-vllm-upstream-1 2>&1 | grep -oE 'num_tokens > 64[^,]*|Check failed[^,]*|RuntimeError:[^,]*|ValueError:[^,]*|AssertionError[^,]*|b12x[^,]*|TileLang[^,]*|Error[^,]*' | tail -4
    exit 1
  fi
  if curl -fsS --max-time 4 http://127.0.0.1:8888/v1/models >/dev/null 2>&1; then
    echo "=== B12X SPEC API UP $(date +%H:%M:%S) ==="
    docker logs upstream-vllm-upstream-1 2>&1 | grep -oE 'b12x compressed-MLA decode enabled[^"]*' | head -1 | sed 's/^/  /'
    echo "--- SPEC DECODE (haiku) ---"
    curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"deepseek-v4-flash-dspark","messages":[{"role":"user","content":"Write a haiku about GPUs, then explain it in one sentence."}],"max_tokens":120,"temperature":0.6}' \
      | python3 -c "import json,sys;print('  DECODE OK:',repr(json.load(sys.stdin)['choices'][0]['message']['content'][:170]))"
    echo "=== B12X SPEC DECODE WORKS ($(date +%H:%M:%S)) ==="
    exit 0
  fi
  sleep 15
done
echo "still loading $(date +%H:%M:%S)"
