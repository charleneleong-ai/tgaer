#!/usr/bin/env bash
# Retry a code submission until the daily allowance resets.
#
# The allowance resets at ~00:00 UTC and the API answers a spent slot with a
# bare 400, so the only way to hand off "submit as soon as we are allowed" is to
# keep asking. Detached (PPID=1) so it outlives the shell and the session.
#
# Usage: submit_when_slot_opens.sh <kernel-version> "<message>"
set -uo pipefail

REPO="/Users/charleneleong/Dropbox/Mac/Documents/gen-ai/orak-hackathon/tgaer"
COMP="arc-prize-2026-arc-agi-3"
KERNEL="charyeezy/arc-agi-3"
VERSION="${1:?kernel version required}"
MESSAGE="${2:?submission message required}"
KAGGLE="$REPO/.venv/bin/kaggle"

cd "$REPO" || exit 1

for attempt in $(seq 1 60); do
    out=$("$KAGGLE" competitions submit -c "$COMP" -k "$KERNEL" \
        -v "$VERSION" -f submission.parquet -m "$MESSAGE" 2>&1)
    printf '%s attempt %s: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$attempt" \
        "$(printf '%s' "$out" | tail -1)"

    # The CLI prints the 400 rather than failing loudly, so confirm against the
    # submission list instead of trusting the exit status.
    if "$KAGGLE" competitions submissions "$COMP" 2>/dev/null |
        grep -q "$MESSAGE"; then
        printf '%s SUBMITTED v%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$VERSION"
        exit 0
    fi
    sleep 300
done

printf '%s GAVE UP after 60 attempts\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 1
