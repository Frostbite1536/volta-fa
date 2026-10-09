#!/bin/bash
# worker-review.sh <name> <worktree-name|/abs/path> "<claims to verify + focus>"
# Cheap read-only review by the worker model. Kernel/numerics/P2P changes need an independent manager-spawned reviewer instead.
# shellcheck source=bin/env.sh
. "$(dirname "$0")/env.sh"
N=$1; FOCUS=$3; T=$VFA_RECEIPTS/$N.ticket.md
case $2 in /*) WT=$2 ;; *) WT=$VFA_WT/$2 ;; esac
cat > "$T" <<EOT
You are an independent code reviewer. READ ONLY: do not edit files, no git writes, no network, never read .env or tokens.
Review the uncommitted change in $WT: run \`git -C $WT status\` and \`git -C $WT diff\`, and read every untracked file. Read docs/INVARIANTS.md first.
Target is NVIDIA Volta sm_70 only: WMMA and PTX mma.m8n8k4 are legal; BF16/FP8/cp.async/ldmatrix/m16n8k* are not.
$FOCUS
Report findings ranked BLOCKING / MEDIUM / LOW with file:line and a concrete failure scenario. Say plainly if nothing blocks. At most 450 words.
EOT
"$(dirname "$0")/worker.sh" "$N" "$WT" "$T" read-only
