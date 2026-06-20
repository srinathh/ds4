#!/usr/bin/env bash
# ds4-server launcher for gx10 (GB10 / sm_121 / 128 GB unified memory)
#
# Clean base 2026-06-11 (docs/plans/ok-please-go-adaptive-yeti.md):
#   - Engine: upstream antirez/ds4, built with `make cuda-spark` (NOT
#     `make cuda CUDA_ARCH=sm_121` — the spark target is faster on GB10).
#   - Quant: q2-q4-imatrix (98 GB) via ./download_model.sh q2-q4-imatrix.
#     q4-imatrix needs >=256 GB RAM — never use it on this host.
#   - Disk KV cache ON (first-class in ds4's design; OpenClaw session
#     switches resume from disk instead of re-prefilling). Cache dir is on
#     NVMe, NOT /tmp: every ds4 restart goes through a host reboot and /tmp
#     may be wiped, which would defeat restart-resume.
#   - No --nothink: OpenClaw controls thinking per-request
#     (model=deepseek-chat + think:false). No --trace (enable ad-hoc when
#     debugging). No --quality yet (canonical base first; A/B later).
#   - MTP not attached (upstream: experimental, "at most a slight speedup",
#     greedy-only). The MTP GGUF is kept in $DS4_DIR/gguf for later A/B.
#
# Operational reminders (NOT in the upstream README):
#   - Native ds4 leaks unified memory on ANY stop (graceful or not).
#     Every stop/restart must be: stop -> sudo systemctl reboot -> relaunch.
#   - Single graph worker: concurrent requests serialize.

set -euo pipefail

DS4_DIR="${DS4_DIR:-$HOME/ds4}"
MODEL="${MODEL:-}"                      # default: $DS4_DIR/ds4flash.gguf symlink
KV_CACHE_DIR="${KV_CACHE_DIR:-$HOME/ds4-kv}"
KV_CACHE_MB="${KV_CACHE_MB:-65536}"     # 64 GB disk budget
CTX="${CTX:-204800}"                    # 200K (~5.2 GB KV) + 0.6B embedder co-resident
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"                    # production slot; ds4 is sole tenant on it

export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -z "$MODEL" ]]; then
  if [[ -e "$DS4_DIR/ds4flash.gguf" ]]; then
    MODEL="$DS4_DIR/ds4flash.gguf"      # symlink maintained by download_model.sh
  else
    MODEL=$(find "$DS4_DIR/gguf" -maxdepth 1 -name "*.gguf" ! -name "*MTP*" 2>/dev/null | sort | head -1)
  fi
  [[ -z "$MODEL" ]] && { echo "No GGUF found under $DS4_DIR — run ./download_model.sh q2-q4-imatrix, or set MODEL=" >&2; exit 1; }
fi

mkdir -p "$KV_CACHE_DIR"

echo "ds4-server: model=$(basename "$(readlink -f "$MODEL")") ctx=$CTX host=$HOST port=$PORT"
echo "ds4-server: kv-disk=$KV_CACHE_DIR (${KV_CACHE_MB} MB budget)"

exec "$DS4_DIR/ds4-server" --cuda \
  -m "$MODEL" \
  -c "$CTX" \
  --host "$HOST" \
  --port "$PORT" \
  --kv-disk-dir "$KV_CACHE_DIR" \
  --kv-disk-space-mb "$KV_CACHE_MB" \
  --warm-weights
