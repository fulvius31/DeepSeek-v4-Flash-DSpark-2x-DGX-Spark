#!/usr/bin/env bash
# Phase 4 -- upstream lane, real model.
#
# Same discipline as the fork rows: worker-first start, restart before EVERY
# soak so each tok/s number is a cold-prefix-cache number and each corruption
# verdict is fresh.
set -uo pipefail

REPO=/home/alessangiorgi/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
cd "$REPO"
set -a; source .env.dspark; set +a

W="$WORKER_HOST"
OUT=bench-results
PROJ=upstream
CF=docker-compose.upstream.yml
mkdir -p "$OUT"

export MAX_MODEL_LEN=200000

# num_speculative_tokens=5: the branch enforces >= dspark_block_size, which the
# DeepSeek-V4-Flash-DSpark checkpoint declares as 5. Values below that are
# rejected at startup ("Smaller values produce incorrect output") -- 3 (the
# runbook example / the fork's MTP_NUM_TOKENS) is invalid here. So the upstream
# rows run at spec depth 5 vs the fork's 3; this asymmetry is documented in the
# evaluation report and is unavoidable (3 will not start on this branch).
U1='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
U2='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","replicate_markov_w1":true}'
U3='{"method":"dspark","num_speculative_tokens":5,"replicate_markov_w1":true,"use_local_argmax_reduction":true}'

spec_for() { case "$1" in U1) echo "$U1";; U2) echo "$U2";; U3) echo "$U3";; esac; }

# The fork's start script scps the fork compose + env to the worker; nothing
# ships the upstream compose. Do it here (worker-first launch needs it there).
sync_worker() {
  ssh "$W" "mkdir -p '$REPO'"
  scp -q "$CF" .env.dspark docker-compose.upstream-tiny.yml "$W:$REPO/"
}
sync_worker

down_stack() {
  COMPOSE_DISABLE_ENV_FILE=1 SPECULATIVE_CONFIG='{}' \
    docker compose -p "$PROJ" --env-file .env.dspark -f "$CF" down >/dev/null 2>&1 || true
  ssh "$W" "cd '$REPO' && COMPOSE_DISABLE_ENV_FILE=1 SPECULATIVE_CONFIG='{}' docker compose -p '$PROJ' --env-file .env.dspark -f '$CF' down" >/dev/null 2>&1 || true
}

up_stack() {
  local spec="$1"
  echo "[$(date -Is)] starting worker (rank 1)..."
  ssh "$W" "cd '$REPO' && COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=1 HEADLESS=1 \
      VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' MAX_MODEL_LEN='$MAX_MODEL_LEN' \
      SPECULATIVE_CONFIG='$spec' \
      docker compose -p '$PROJ' --env-file .env.dspark -f '$CF' up -d" || return 1

  echo "[$(date -Is)] starting head (rank 0)..."
  COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 HEADLESS= \
    VLLM_HOST_IP="$VLLM_HOST_IP" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    SPECULATIVE_CONFIG="$spec" \
    docker compose -p "$PROJ" --env-file .env.dspark -f "$CF" up -d || return 1

  echo "[$(date -Is)] waiting for API..."
  for _ in $(seq 1 100); do
    if curl -fsS --max-time 5 http://127.0.0.1:8888/v1/models >/dev/null 2>&1; then
      echo "[$(date -Is)] API up"
      return 0
    fi
    sleep 15
  done
  echo "[$(date -Is)] TIMED OUT waiting for API"
  return 1
}

soak() {
  local row="$1"
  local conc="$2"
  local out="$OUT/upstream-${row}-c${conc}.txt"
  python3 scripts/dspark-corruption-soak.py \
    --base-url http://127.0.0.1:8888/v1 --model deepseek-v4-flash-dspark \
    --concurrency "$conc" --turns 12 --temperature 0.6 \
    --label "upstream-${row}" > "$out" 2>&1
  local rc=$?
  echo "exit=$rc" >> "$out"
  echo "row=$row c=$conc exit=$rc  $(grep -E 'aggregate throughput' "$out" || true)"
}

capture_logs() {
  local row="$1"
  COMPOSE_DISABLE_ENV_FILE=1 SPECULATIVE_CONFIG='{}' \
    docker compose -p "$PROJ" --env-file .env.dspark -f "$CF" logs \
    > "$OUT/upstream-${row}-head.log" 2>&1 || true
  grep -iE 'accept|spec|draft' "$OUT/upstream-${row}-head.log" | tail -40 \
    > "$OUT/upstream-${row}-acceptance.txt" 2>&1 || true
  echo "--- acceptance/spec lines ($row) ---"
  tail -8 "$OUT/upstream-${row}-acceptance.txt" 2>/dev/null || true
}

for row in ${ROWS:-U1 U2 U3}; do
  spec="$(spec_for "$row")"
  echo
  echo "############ [$(date -Is)] ROW $row ############"
  echo "SPECULATIVE_CONFIG=$spec"

  down_stack
  if ! up_stack "$spec"; then
    echo "ROW $row: FAILED TO START -- capturing diagnostics"
    capture_logs "$row"
    ssh "$W" "cd '$REPO' && COMPOSE_DISABLE_ENV_FILE=1 SPECULATIVE_CONFIG='{}' docker compose -p '$PROJ' --env-file .env.dspark -f '$CF' logs --tail=200" \
      > "$OUT/upstream-${row}-worker.log" 2>&1 || true
    down_stack
    continue
  fi

  CONCURRENCY=6 ./smoke-deepseek-v4-flash-dspark.sh || echo "ROW $row: smoke FAILED"
  soak "$row" 4
  capture_logs "$row"

  # cold prefix cache for the single-stream number
  down_stack
  if up_stack "$spec"; then soak "$row" 1; else echo "ROW $row: c=1 restart failed"; fi
  down_stack
done

echo
echo "############ [$(date -Is)] UPSTREAM ROWS COMPLETE ############"
grep -H -E 'exit=|aggregate throughput' "$OUT"/upstream-*.txt 2>/dev/null || true
