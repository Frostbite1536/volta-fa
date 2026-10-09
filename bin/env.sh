#!/bin/bash
# Sourced by every script. Override per machine.
VFA_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VFA_HOME
export VFA_RECEIPTS="$VFA_HOME/state/receipts"
export VFA_WT="$VFA_HOME/.worktrees"
export VFA_PROFILES="$VFA_HOME/profiles"
export VFA_PY="$VFA_HOME/.venv/bin/python"
# Verify before first use: `codex exec -m <id>` must be a model your CLI accepts.
export WORKER_MODEL="${WORKER_MODEL:-gpt-6.1-sol}"
export WORKER_EFFORT="${WORKER_EFFORT:-high}"
