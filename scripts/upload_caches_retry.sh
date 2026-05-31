#!/usr/bin/env bash
# Auto-restarting wrapper around scripts/upload_caches_to_hf.py.
#
# upload_large_folder has exited silently a couple of times (transient LFS
# I/O errors / orphaned-on-session-resume). The upload is fully resumable
# (state cached under results/.cache/huggingface/upload/), so the robust
# fix is simply to re-run it until the HF repo holds the expected file
# count. Each restart continues from where the last attempt stopped.
#
# Run (detached, survives shell/session boundaries):
#   nohup bash scripts/upload_caches_retry.sh > /tmp/hf_upload.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

REPO="faizancodes/no-op-circuit-caches"
TARGET=8900          # ~8931 total files; treat >=8900 as complete
MAX_ATTEMPTS=40

count_hf() {
  .venv/bin/python - <<'PY' 2>/dev/null
import os
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import HfApi
api = HfApi(token=os.getenv("HF_TOKEN"))
try:
    print(len(list(api.list_repo_files(
        "faizancodes/no-op-circuit-caches", repo_type="dataset"))))
except Exception:
    print(-1)
PY
}

# Robust count: retry up to 3x and fall back to -1 on a transient empty
# return, so an HF API hiccup can't (a) crash the [ -ge ] test or (b) make
# the loop miss a genuine completion (the bug that printed "STOPPED" on the
# first full run even though all files were already uploaded).
count_hf_robust() {
  local n
  for _ in 1 2 3; do
    n=$(count_hf)
    [ -n "$n" ] && [ "$n" -ge 0 ] 2>/dev/null && { echo "$n"; return; }
    sleep 3
  done
  echo "-1"
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  n=$(count_hf_robust)
  echo "=== $(date '+%H:%M:%S') attempt $attempt — HF has $n / $TARGET files ==="
  if [ "$n" -ge "$TARGET" ]; then
    echo "=== COMPLETE: $n files on HF ==="
    exit 0
  fi
  .venv/bin/python scripts/upload_caches_to_hf.py
  echo "=== $(date '+%H:%M:%S') upload attempt $attempt returned (exit $?); pausing 10s ==="
  sleep 10
done

n=$(count_hf_robust)
echo "=== STOPPED after $MAX_ATTEMPTS attempts — HF has $n / $TARGET files ==="
[ "$n" -ge "$TARGET" ] && exit 0 || exit 1
