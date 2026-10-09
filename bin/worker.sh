#!/bin/bash
# worker.sh <name> <worktree-name|/abs/path> <ticket-file> [read-only]
# Dispatch the worker model (Codex). stdin is CLOSED: an open stdin makes codex wait forever.
# Never pass --add-dir. Run as a background tool call.
# shellcheck source=bin/env.sh
. "$(dirname "$0")/env.sh"
N=$1; T=$3; MODE=${4:-write}
case $2 in /*) W=$2 ;; *) W=$VFA_WT/$2 ;; esac
[ -d "$W" ] || { echo "worktree missing: $W" >&2; exit 2; }
[ -f "$T" ] || { echo "ticket missing: $T" >&2; exit 2; }
if [ "$MODE" = read-only ]; then SB=(-s read-only); else SB=(-s workspace-write); fi
codex exec -m "$WORKER_MODEL" -c model_reasoning_effort="$WORKER_EFFORT" "${SB[@]}" \
  -C "$W" -o "$VFA_RECEIPTS/$N.md" "$(cat "$T")" < /dev/null > "$VFA_RECEIPTS/$N.log" 2>&1
echo "exit=$?"; git -C "$W" status -s | head -40; tail -40 "$VFA_RECEIPTS/$N.md"
