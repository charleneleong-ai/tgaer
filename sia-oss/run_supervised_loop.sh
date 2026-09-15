#!/usr/bin/env bash
# Runs N more single-generation sia runs, each fresh (the framework has no
# --resume), each seeded by the current sia-oss/tasks/arc-agi3/reference/
# (which the supervisor promotes into on a genuine win), with the supervisor
# updating the persistent AGENTS.md ledger between every run.
set -euo pipefail
cd "$(dirname "$0")/.."

set -a; source .env; set +a
export ARC_TGAER_REPO="$(pwd)"
export ARC_SIA_REPEATS=1
export ARC_SIA_MAX_ACTIONS=600

START_RUN_ID="${1:?usage: run_supervised_loop.sh <start_run_id> <count>}"
COUNT="${2:?usage: run_supervised_loop.sh <start_run_id> <count>}"

for i in $(seq 0 $((COUNT - 1))); do
  run_id=$((START_RUN_ID + i))
  echo "=== LOOP: launching sia run_id=${run_id} ==="
  sia-oss/.venv/bin/sia run \
    --task_dir sia-oss/tasks/arc-agi3 \
    --meta-agent-profile sia-oss/profiles/openai-meta.json \
    --target-agent-profile sia-oss/profiles/arc-agi3-target.json \
    --sandbox none \
    --focus harness \
    --max_gen 1 \
    --run_id "${run_id}" \
    --no-web
  echo "=== LOOP: sia run_id=${run_id} finished, running supervisor ==="
  .venv/bin/python sia-oss/supervisor.py --run-id "${run_id}" --gen 1
  echo "=== LOOP: supervisor done for run_id=${run_id} ==="
done
echo "=== LOOP: all ${COUNT} generations complete ==="
