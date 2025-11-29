#!/usr/bin/env bash
set -euo pipefail

# One-command bootstrap: setup env + start API.
# Usage: HF_TOKEN=your_hf_token bash models/synchuman/bootstrap.sh

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required (your Hugging Face token)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_ROOT="${SYNC_ROOT:-/workspace/SyncHuman}"

# Run setup (installs deps, downloads weights)
SYNC_ROOT="$SYNC_ROOT" HF_TOKEN="$HF_TOKEN" bash "$SCRIPT_DIR/setup_instance.sh"

# Start API
cd "$SYNC_ROOT"
source venv/bin/activate
export PYTHONPATH="$SYNC_ROOT"
export ATTN_BACKEND=flash_attn SPARSE_ATTN_BACKEND=flash_attn
nohup python api_server.py > /workspace/sync_api.log 2>&1 &
echo "API started (log: /workspace/sync_api.log)."

# Health check
sleep 3
if curl -fsS http://localhost:8000/health >/dev/null; then
  echo "Health check: ok"
else
  echo "Health check failed; see /workspace/sync_api.log" >&2
  exit 1
fi
