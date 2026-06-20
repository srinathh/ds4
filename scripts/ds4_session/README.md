# ds4_session — Per-Turn DS4 Request Extractor

Captures DS4 server logs and trace blocks after each OpenClaw turn, splits them into per-request files, and writes standardized metrics as CSV.

## How it works

After you complete an OpenClaw turn, run `ds4_session.py turn`. The script:

1. Pulls docker log lines from gx10 since the last anchor (via SSH)
2. Reads new blocks from the local trace file since the last byte offset
3. Groups log lines into per-request buckets (one per `prompt start` event)
4. Pairs each log bucket with its corresponding trace block (by position + timestamp validation)
5. Writes per-request log and trace files, a per-turn CSV, and a prose summary

## Prerequisites

- SSH access to gx10 (`192.168.0.250`) without a passphrase prompt
- Local `data/ds4-trace.txt` — run `/trace-update` before starting a session to sync it from gx10
- Python ≥ 3.10 with `uv` available

## Configuration

Edit `scripts/ds4_session/ds4_session.cfg` to set defaults for your environment:

```ini
[ds4_session]
host         = srinathh@192.168.0.250   # SSH host for docker logs
trace_file   = data/ds4-trace.txt       # local trace file (relative to repo root)
sessions_dir = data/sessions            # where sessions are written
```

All relative paths resolve against the repo root. CLI flags override config values.

## Usage

```bash
# 1. Sync trace from gx10 (do this once before the session)
#    /trace-update

# 2. Initialise — anchors to current position in log stream and trace file
uv run scripts/ds4_session/ds4_session.py init

# 3. After each OpenClaw turn, extract that turn's data
uv run scripts/ds4_session/ds4_session.py turn

# 4. Check current session state
uv run scripts/ds4_session/ds4_session.py status
```

### Offline / test mode (no SSH needed)

```bash
uv run scripts/ds4_session/ds4_session.py \
    --log-file data/sample_logs.txt \
    --trace data/sample_trace.txt \
    init --out test-offline

uv run scripts/ds4_session/ds4_session.py \
    --log-file data/sample_logs.txt \
    --trace data/sample_trace.txt \
    turn --session data/sessions/test-offline
```

### Flags (override config)

| Flag | Description |
|------|-------------|
| `--host HOST` | SSH host for docker logs |
| `--trace FILE` | Local trace file path |
| `--log-file FILE` | Offline mode: read logs from file instead of SSH |
| `--session PATH` | Explicit session directory (default: most recent) |
| `--out NAME` | Session name for `init` (default: `session-YYYYMMDD-HHMMSS`) |

## Output structure

```
data/sessions/session-YYYYMMDD-HHMMSS/
  state.json                        session anchors and turn counter
  session-requests.csv              cumulative metrics across all turns
  turn_001/
    summary.md                      prose narrative (no tables)
    turn_001-requests.csv           per-request metrics for this turn
    turn_001-request_001-log.txt    raw log lines for request 1
    turn_001-request_001-trace.txt  raw trace block for request 1
    turn_001-request_002-log.txt    (tool-call loop request 2, if any)
    ...
```

`session-requests.csv` is the primary analysis artifact — all turns appended to one file so you can open it in a spreadsheet without concatenating files.

## CSV columns

| Column | Source | Description |
|--------|--------|-------------|
| `turn` | state | Turn number (1-based) |
| `request` | sequence | Request number within the turn (1-based) |
| `timestamp` | log | `MMDD HH:MM:SS` from the `prompt start` line |
| `mode` | log | `TOOLS` or `NOTOOL` |
| `start_ctx` | log | Tokens already in KV before this request (X in `ctx=X..Y:N`) |
| `end_ctx` | log | KV position after the prompt (Y) |
| `fresh_prefill_tokens` | log | Tokens that required compute (N in `:N`) |
| `prefill_duration_s` | log | Prefill wall time in seconds |
| `prefill_rate_avg_tps` | log | Average prefill throughput (t/s) |
| `kv_continued_count` | log | Mid-prefill KV checkpoint saves (`reason=continued`) |
| `kv_continued_save_ms_total` | log | Total ms spent saving continued KV checkpoints |
| `kv_evict_tokens` | log | Tokens evicted to disk before loading a snapshot (`reason=evict`) |
| `kv_evict_save_ms` | log | Time to write the evicted KV snapshot |
| `disk_kv_loaded_tokens` | log | Tokens loaded from disk KV (`kv cache hit` line) |
| `disk_kv_load_ms` | log | Time to load from disk KV |
| `cache_miss_common` | log | Common-prefix boundary from `live kv cache miss` (-1 if live-continue) |
| `cache_miss_reason_log` | log | Miss reason string from log (`token-mismatch` etc.) |
| `generated_tokens` | log | Tokens generated (from finish line `gen=G`) |
| `decode_duration_s` | log | Decode wall time (from last decode chunk) |
| `decode_rate_avg_tps` | log | Average decode throughput (t/s) |
| `total_request_time_s` | log | Total request wall time from finish line |
| `finish_reason` | log | `stop`, `tool_calls`, or `error` |
| `tool_names` | log | Comma-joined tool names called (from `tool calls` line) |
| `cache_classification` | trace | `HIT` / `COLD` / `MISS` derived from `memory_miss_reason` |
| `live_prompt_common` | trace | Common live-memory prefix from trace cache decision block |
| `memory_miss_reason_trace` | trace | Full miss reason string from trace |
| `disk_cached_tokens_trace` | trace | Disk-cached token count from trace (useful when no log `kv cache hit`) |
| `cache_source` | trace | `none`, `disk-text`, `memory-token`, etc. |
| `trace_req_id` | trace | Global request number from the trace file |

## Notes

- **Disk KV**: `disk_kv_loaded_tokens` (from log) and `disk_cached_tokens_trace` (from trace) may differ. The trace value is always present; the log value only appears when a `kv cache hit` line fires mid-session.
- **MTP noise**: Lines prefixed `ds4:` (not `ds4-server:`) are excluded by the SSH grep filter — they are high-frequency MTP speculation events, not per-request metrics.
- **Tool-call loops**: A single OpenClaw turn often produces 2–4 requests (model calls a tool, sees result, calls another tool, etc.). All are captured in the same turn directory.
