#!/usr/bin/env bash
# Phase 2 -- fork-stack corruption A/B.
#
# Each row: set the env knob, restart the stack (worker-first, via the repo's
# start script), soak at concurrency 4 (the decisive corruption gate), restart
# again, soak at concurrency 1 (single-stream throughput).
#
# The stack restarts before EVERY soak, not just before every row: the soak's
# task list is fixed and the server runs with prefix caching, so a warm cache
# left by a previous soak would inflate the next one's tok/s. Restarting keeps
# every number a cold-cache number and every corruption verdict a fresh one
# (ground rule 4).
set -uo pipefail

REPO=/home/alessangiorgi/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
cd "$REPO"
OUT=bench-results
mkdir -p "$OUT"

ROWS="${ROWS:-F1 F2 F3}"

set_key() {
  local key="$1" val="$2"
  if grep -q "^${key}=" .env.dspark; then
    sed -i "s|^${key}=.*|${key}=${val}|" .env.dspark
  else
    printf '%s=%s\n' "$key" "$val" >> .env.dspark
  fi
}

# Reset every knob the ladder touches back to its default, so rows never leak
# into each other.
reset_knobs() {
  set_key VLLM_DSPARK_SKIP_RAGGED_SPECULATION 0
  set_key GREEDY_VERIFICATION 0
  set_key MTP_NUM_TOKENS 3
}

apply_row() {
  reset_knobs
  case "$1" in
    F1) : ;;                                              # fixed guard only
    F2) set_key VLLM_DSPARK_SKIP_RAGGED_SPECULATION 1 ;;
    F3) set_key GREEDY_VERIFICATION 1 ;;
    F4) set_key MTP_NUM_TOKENS 0 ;;
    *)  echo "unknown row $1" >&2; return 1 ;;
  esac
}

restart_stack() {
  ./stop-deepseek-v4-flash-dspark.sh >/dev/null 2>&1 || true
  ssh alessangiorgi@10.0.0.2 \
    "cd $REPO && docker compose -p deepseek-v4-flash --env-file .env.dspark -f docker-compose.dspark.yml down" \
    >/dev/null 2>&1 || true
  ./start-deepseek-v4-flash-dspark.sh
}

soak() {
  local row="$1"
  local conc="$2"
  local out="$OUT/fork-${row}-c${conc}.txt"
  echo "--- [$(date -Is)] soak row=$row concurrency=$conc ---"
  python3 scripts/dspark-corruption-soak.py \
    --base-url http://127.0.0.1:8888/v1 --model deepseek-v4-flash-dspark \
    --concurrency "$conc" --turns 12 --temperature 0.6 \
    --label "fork-${row}" > "$out" 2>&1
  local rc=$?
  echo "exit=$rc" >> "$out"
  echo "row=$row c=$conc exit=$rc  $(grep -E 'aggregate throughput' "$out" || true)"
  return 0
}

for row in $ROWS; do
  echo
  echo "############ [$(date -Is)] ROW $row ############"
  apply_row "$row" || continue
  grep -E '^(VLLM_DSPARK_SKIP_RAGGED_SPECULATION|GREEDY_VERIFICATION|MTP_NUM_TOKENS)=' .env.dspark

  echo "[$(date -Is)] restart for c=4 soak"
  if ! restart_stack; then
    echo "ROW $row: STACK FAILED TO START -- capturing diagnostics"
    ./logs-deepseek-v4-flash-dspark.sh > "$OUT/fork-${row}-startfail.txt" 2>&1 || true
    continue
  fi
  soak "$row" 4

  echo "[$(date -Is)] restart for c=1 soak (cold prefix cache)"
  restart_stack >/dev/null 2>&1 && soak "$row" 1 \
    || echo "ROW $row: c=1 restart failed"
done

echo
echo "############ [$(date -Is)] FORK ROWS COMPLETE ############"
grep -H -E 'exit=|aggregate throughput' "$OUT"/fork-*.txt 2>/dev/null || true
