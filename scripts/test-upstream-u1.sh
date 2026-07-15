#!/usr/bin/env bash
# Focused proof that the flashinfer-0.6.14 overlay lets upstream serve the REAL
# model: bring up upstream U1 (probabilistic, spec depth 5) with the fi0614
# image and exercise the decode kernel that crashed on flashinfer 0.6.13.
set -uo pipefail
REPO=/home/alessangiorgi/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
cd "$REPO"; set -a; source .env.dspark; set +a
W=alessangiorgi@10.0.0.2
CF=docker-compose.upstream.yml
PROJ=upstream
export UPSTREAM_IMAGE=vllm-upstream-gb10:dspark-tp-fi0615
export MAX_MODEL_LEN=200000
SPEC='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'

echo "[$(date +%H:%M:%S)] ensure clean (both nodes)"
COMPOSE_DISABLE_ENV_FILE=1 SPECULATIVE_CONFIG='{}' docker compose -p "$PROJ" --env-file .env.dspark -f "$CF" down >/dev/null 2>&1 || true
ssh "$W" "cd '$REPO' && COMPOSE_DISABLE_ENV_FILE=1 SPECULATIVE_CONFIG='{}' docker compose -p '$PROJ' --env-file .env.dspark -f '$CF' down" >/dev/null 2>&1 || true

echo "[$(date +%H:%M:%S)] sync compose+env to worker"
scp -q "$CF" .env.dspark "$W:$REPO/"

echo "[$(date +%H:%M:%S)] worker up (rank 1)"
ssh "$W" "cd '$REPO' && COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=1 HEADLESS=1 \
  VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' MAX_MODEL_LEN='$MAX_MODEL_LEN' \
  UPSTREAM_IMAGE='$UPSTREAM_IMAGE' SPECULATIVE_CONFIG='$SPEC' \
  docker compose -p '$PROJ' --env-file .env.dspark -f '$CF' up -d" || exit 2

echo "[$(date +%H:%M:%S)] head up (rank 0)"
COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 HEADLESS= \
  VLLM_HOST_IP="$VLLM_HOST_IP" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  UPSTREAM_IMAGE="$UPSTREAM_IMAGE" SPECULATIVE_CONFIG="$SPEC" \
  docker compose -p "$PROJ" --env-file .env.dspark -f "$CF" up -d || exit 2

echo "[$(date +%H:%M:%S)] waiting for API (fail fast on exit)"
for i in $(seq 1 60); do
  if docker ps -a --format '{{.Status}}' --filter name=upstream | grep -q Exited; then
    echo "HEAD EXITED at $(date +%H:%M:%S):"
    docker logs upstream-vllm-upstream-1 2>&1 | grep -oE 'TypeError:.*|ValueError:.*|RuntimeError:.*|swa_topk_lens.*|unexpected keyword.*' | head -3
    exit 3
  fi
  if ssh "$W" 'docker ps -a --format "{{.Status}}" --filter name=upstream' 2>/dev/null | grep -q Exited; then
    echo "WORKER EXITED at $(date +%H:%M:%S):"
    ssh "$W" 'docker logs upstream-vllm-upstream-1 2>&1 | grep -oE "TypeError:.*|ValueError:.*|RuntimeError:.*|swa_topk_lens.*|unexpected keyword.*" | head -3'
    exit 3
  fi
  if curl -fsS --max-time 5 http://127.0.0.1:8888/v1/models >/dev/null 2>&1; then
    echo "=== API UP at $(date +%H:%M:%S) ==="; break
  fi
  sleep 15
done

echo "--- /v1/models ---"
curl -s --max-time 5 http://127.0.0.1:8888/v1/models | python3 -c "import json,sys;d=json.load(sys.stdin);print('  served:',d['data'][0]['id'])" || { echo "API never came up"; exit 4; }

echo "--- REAL decode (the kernel that crashed on 0.6.13) ---"
curl -s --max-time 90 http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash-dspark","messages":[{"role":"user","content":"Write a haiku about GPUs, then explain it in one sentence."}],"max_tokens":120,"temperature":0.6}' \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print('  DECODE OK:',repr(d['choices'][0]['message']['content'][:180]))" || { echo "DECODE FAILED"; exit 5; }

echo "--- NCCL transport ---"
docker logs upstream-vllm-upstream-1 2>&1 | grep -oE 'NET/IB : Using \[0\]rocep1s0f0[^;]*' | head -1 | sed 's/^/  /'
echo "=== U1 SERVES THE REAL MODEL WITH flashinfer 0.6.14: SUCCESS ($(date +%H:%M:%S)) ==="
