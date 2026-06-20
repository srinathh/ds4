# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""ds4_session.py — per-turn DS4 request extractor.

Anchors in the docker log stream and local trace file, then after each
OpenClaw turn extracts per-request log + trace files and a CSV of metrics.

Commands:
  init   -- establish anchors (log epoch + trace byte offset), create session dir
  turn   -- extract next turn's data, write files, advance anchors
  status -- print current session state

Config: scripts/ds4_session.cfg (same dir as this file). CLI flags override.
Run:    uv run scripts/ds4_session.py init
        uv run scripts/ds4_session.py turn
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # scripts/ds4_session/ → scripts/ → repo root
CFG_PATH = SCRIPT_DIR / "ds4_session.cfg"

DEFAULTS = {
    "host": "srinathh@192.168.0.250",
    "trace_file": "data/ds4-trace.txt",
    "sessions_dir": "data/sessions",
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CFG_PATH.exists():
        cp = configparser.ConfigParser()
        cp.read(CFG_PATH)
        if cp.has_section("ds4_session"):
            for k, v in cp["ds4_session"].items():
                cfg[k] = v
    # resolve relative paths against repo root
    for key in ("trace_file", "sessions_dir"):
        p = Path(cfg[key])
        if not p.is_absolute():
            cfg[key] = str(REPO_ROOT / p)
    return cfg


# ---------------------------------------------------------------------------
# Log line regexes
# ---------------------------------------------------------------------------

PROMPT_START_RE = re.compile(
    r"(\d{4} \d{2}:\d{2}:\d{2}) ds4-server: chat ctx=(\d+)\.\.(\d+):(\d+) (\S+) prompt start"
)
PROMPT_DONE_RE = re.compile(
    r"ds4-server: chat ctx=(\d+)\.\.(\d+):(\d+) \S+ prompt done ([0-9.]+)s"
)
PREFILL_CHUNK_RE = re.compile(
    r"ds4-server: chat ctx=(\d+)\.\.(\d+):(\d+) \S+ prefill chunk \d+/\d+"
    r" \([0-9.]+%\) chunk=[0-9.]+ t/s avg=([0-9.]+) t/s ([0-9.]+)s"
)
DECODE_CHUNK_RE = re.compile(
    r"ds4-server: chat ctx=\d+\.\.\d+:\d+ gen=(\d+) (?:\S+ )+decoding"
    r" chunk=[0-9.]+ t/s avg=([0-9.]+) t/s ([0-9.]+)s"
)
FINISH_RE = re.compile(
    r"(\d{4} \d{2}:\d{2}:\d{2}) ds4-server: chat ctx=(\d+)\.\.(\d+):(\d+)"
    r" gen=(\d+) (?:\S+ )+finish=(\S+) ([0-9.]+)s"
)
KV_HIT_RE = re.compile(
    r"ds4-server: kv cache hit text tokens=(\d+).*load=([0-9.]+) ms"
)
KV_STORED_RE = re.compile(
    r"ds4-server: kv cache stored tokens=(\d+) trimmed=\d+ reason=(\S+)"
    r" key=\S+ size=([0-9.]+ \S+) save=([0-9.]+) ms"
)
KV_MISS_RE = re.compile(
    r"ds4-server: live kv cache miss live=\d+ prompt=\d+ common=(\d+) reason=(\S+)"
)
TOOL_CALLS_RE = re.compile(
    r"ds4-server: tool calls ctx=\d+\.\.\d+:\d+ n=\d+ .*names=\[([^\]]+)\]"
)

# Trace markers (same as trace_tools.py)
TRACE_START_RE = re.compile(r"^===== request (\d+) (.+?) =====$", re.M)
TRACE_END_RE = re.compile(r"^===== end request (\d+) =====$", re.M)

# ---------------------------------------------------------------------------
# Log grouping
# ---------------------------------------------------------------------------


@dataclass
class LogBucket:
    """All log lines belonging to one DS4 request."""
    prompt_start_ts: str = ""
    mode: str = ""
    start_ctx: int = 0
    end_ctx: int = 0
    fresh_prefill_tokens: int = 0
    lines: list[str] = field(default_factory=list)


def group_log_lines(log_text: str) -> list[LogBucket]:
    """Split log lines into per-request buckets at each 'prompt start' line.

    Lines that appear before the first prompt start (e.g. a live kv cache miss
    that is logged just before prefill begins) are prepended to the first bucket.
    """
    lines = log_text.splitlines()
    # find indices of all prompt-start lines
    starts: list[tuple[int, re.Match]] = []
    for i, line in enumerate(lines):
        m = PROMPT_START_RE.search(line)
        if m:
            starts.append((i, m))

    if not starts:
        return []

    buckets: list[LogBucket] = []
    for idx, (start_line_idx, m) in enumerate(starts):
        end_line_idx = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)

        # include any preamble lines (miss/evict) that sit just before this start
        preamble_start = starts[idx - 1][0] + 1 if idx > 0 else 0
        # only take preamble lines from between prev bucket's start and this one
        preamble_lines = [
            l for l in lines[preamble_start:start_line_idx]
            if KV_MISS_RE.search(l) or
               (KV_STORED_RE.search(l) and "reason=evict" in l)
        ]

        bucket_lines = preamble_lines + lines[start_line_idx:end_line_idx]

        b = LogBucket(
            prompt_start_ts=m.group(1),
            mode=m.group(5),
            start_ctx=int(m.group(2)),
            end_ctx=int(m.group(3)),
            fresh_prefill_tokens=int(m.group(4)),
            lines=bucket_lines,
        )
        buckets.append(b)

    return buckets


# ---------------------------------------------------------------------------
# Metric extraction from a log bucket
# ---------------------------------------------------------------------------


@dataclass
class RequestMetrics:
    turn: int = 0
    request: int = 0
    timestamp: str = ""
    mode: str = ""
    start_ctx: int = 0
    end_ctx: int = 0
    fresh_prefill_tokens: int = 0
    prefill_duration_s: float = 0.0
    prefill_rate_avg_tps: float = 0.0
    kv_continued_count: int = 0
    kv_continued_save_ms_total: float = 0.0
    kv_evict_tokens: int = 0
    kv_evict_save_ms: float = 0.0
    disk_kv_loaded_tokens: int = 0
    disk_kv_load_ms: float = 0.0
    cache_miss_common: int = -1
    cache_miss_reason_log: str = ""
    generated_tokens: int = 0
    decode_duration_s: float = 0.0
    decode_rate_avg_tps: float = 0.0
    total_request_time_s: float = 0.0
    finish_reason: str = ""
    tool_names: str = ""
    # trace-derived (filled in later)
    cache_classification: str = ""
    live_prompt_common: int = -1
    memory_miss_reason_trace: str = ""
    disk_cached_tokens_trace: int = 0
    cache_source: str = ""
    trace_req_id: int = -1


CSV_COLS = [
    "turn", "request", "timestamp", "mode",
    "start_ctx", "end_ctx", "fresh_prefill_tokens",
    "prefill_duration_s", "prefill_rate_avg_tps",
    "kv_continued_count", "kv_continued_save_ms_total",
    "kv_evict_tokens", "kv_evict_save_ms",
    "disk_kv_loaded_tokens", "disk_kv_load_ms",
    "cache_miss_common", "cache_miss_reason_log",
    "generated_tokens", "decode_duration_s", "decode_rate_avg_tps",
    "total_request_time_s", "finish_reason", "tool_names",
    "cache_classification", "live_prompt_common",
    "memory_miss_reason_trace", "disk_cached_tokens_trace",
    "cache_source", "trace_req_id",
]


def metrics_from_bucket(bucket: LogBucket) -> RequestMetrics:
    m = RequestMetrics(
        timestamp=bucket.prompt_start_ts,
        mode=bucket.mode,
        start_ctx=bucket.start_ctx,
        end_ctx=bucket.end_ctx,
        fresh_prefill_tokens=bucket.fresh_prefill_tokens,
    )
    for line in bucket.lines:
        if hit := KV_MISS_RE.search(line):
            m.cache_miss_common = int(hit.group(1))
            m.cache_miss_reason_log = hit.group(2)
        elif hit := KV_HIT_RE.search(line):
            m.disk_kv_loaded_tokens = int(hit.group(1))
            m.disk_kv_load_ms = float(hit.group(2))
        elif hit := KV_STORED_RE.search(line):
            reason = hit.group(2)
            if reason == "continued":
                m.kv_continued_count += 1
                m.kv_continued_save_ms_total += float(hit.group(4))
            elif reason == "evict":
                m.kv_evict_tokens = int(hit.group(1))
                m.kv_evict_save_ms = float(hit.group(4))
        elif hit := PREFILL_CHUNK_RE.search(line):
            m.prefill_rate_avg_tps = float(hit.group(4))
        elif hit := PROMPT_DONE_RE.search(line):
            m.prefill_duration_s = float(hit.group(4))
        elif hit := DECODE_CHUNK_RE.search(line):
            m.generated_tokens = int(hit.group(1))
            m.decode_rate_avg_tps = float(hit.group(2))
            m.decode_duration_s = float(hit.group(3))
        elif hit := FINISH_RE.search(line):
            m.generated_tokens = int(hit.group(5))
            m.finish_reason = hit.group(6)
            m.total_request_time_s = float(hit.group(7))
        elif hit := TOOL_CALLS_RE.search(line):
            names = hit.group(1).replace(" ", "")
            m.tool_names = names
    return m


# ---------------------------------------------------------------------------
# Trace block extraction
# ---------------------------------------------------------------------------


@dataclass
class TraceBlock:
    req_id: int
    req_ts: str
    text: str  # full block including ===== markers


def extract_trace_blocks(trace_text: str) -> list[TraceBlock]:
    starts = list(TRACE_START_RE.finditer(trace_text))
    ends = {int(m.group(1)): m for m in TRACE_END_RE.finditer(trace_text)}
    blocks: list[TraceBlock] = []
    for s in starts:
        req_id = int(s.group(1))
        req_ts = s.group(2)
        e = ends.get(req_id)
        if e:
            block_text = trace_text[s.start(): e.end()]
        else:
            # no end marker — take to next start or EOF
            next_starts = [t for t in starts if t.start() > s.start()]
            end_pos = next_starts[0].start() if next_starts else len(trace_text)
            block_text = trace_text[s.start():end_pos]
        blocks.append(TraceBlock(req_id=req_id, req_ts=req_ts, text=block_text))
    return blocks


def _trace_field(text: str, name: str) -> str:
    m = re.search(rf"^{re.escape(name)}:[ \t]*(.*)$", text, re.M)
    return m.group(1).strip() if m else ""


def _trace_int(text: str, name: str) -> int:
    v = _trace_field(text, name)
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def augment_from_trace(metrics: RequestMetrics, block: TraceBlock) -> None:
    """Fill trace-derived fields into an existing RequestMetrics."""
    metrics.trace_req_id = block.req_id
    reason = _trace_field(block.text, "memory_miss_reason")
    metrics.memory_miss_reason_trace = reason
    metrics.live_prompt_common = _trace_int(block.text, "live_prompt_common")
    metrics.disk_cached_tokens_trace = _trace_int(block.text, "disk_cached_tokens")
    metrics.cache_source = _trace_field(block.text, "cache_source")
    if reason == "live-prefix-match":
        metrics.cache_classification = "HIT"
    elif reason == "no-live-checkpoint":
        metrics.cache_classification = "COLD"
    elif reason:
        metrics.cache_classification = "MISS"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

STATE_VERSION = 1


def _session_state_path(session_dir: str) -> Path:
    return Path(session_dir) / "state.json"


def load_state(session_dir: str) -> dict:
    p = _session_state_path(session_dir)
    with open(p) as f:
        return json.load(f)


def save_state(session_dir: str, state: dict) -> None:
    _session_state_path(session_dir).write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )


def latest_session(sessions_dir: str) -> str | None:
    d = Path(sessions_dir)
    if not d.exists():
        return None
    candidates = sorted(d.glob("session-*"), reverse=True)
    return str(candidates[0]) if candidates else None


# ---------------------------------------------------------------------------
# SSH / log fetch
# ---------------------------------------------------------------------------


def fetch_log_lines(host: str, since_epoch: int) -> str:
    cmd = [
        "ssh", host,
        f"docker logs ds4 --since {since_epoch} 2>&1 | grep 'ds4-server:'"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # grep returns exit 1 when no lines match — that's fine
    if result.returncode not in (0, 1):
        print(f"!! SSH failed (exit {result.returncode}): {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def read_trace_slice(trace_file: str, byte_offset: int) -> tuple[str, int]:
    """Read trace from byte_offset to EOF. Returns (text, new_offset)."""
    with open(trace_file, "rb") as f:
        f.seek(byte_offset)
        data = f.read()
        new_offset = f.tell()
    return data.decode("utf-8", errors="replace"), new_offset


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _turn_dir(session_dir: str, turn_n: int) -> Path:
    d = Path(session_dir) / f"turn_{turn_n:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_request_files(
    turn_dir: Path,
    turn_n: int,
    req_n: int,
    bucket: LogBucket,
    trace_block: TraceBlock | None,
) -> None:
    prefix = f"turn_{turn_n:03d}-request_{req_n:03d}"
    log_path = turn_dir / f"{prefix}-log.txt"
    log_path.write_text("\n".join(bucket.lines) + "\n", encoding="utf-8")
    if trace_block:
        trace_path = turn_dir / f"{prefix}-trace.txt"
        trace_path.write_text(trace_block.text, encoding="utf-8")


def write_turn_summary(
    turn_dir: Path,
    turn_n: int,
    metrics_list: list[RequestMetrics],
) -> None:
    if not metrics_list:
        return

    first_ts = metrics_list[0].timestamp
    last_ts = metrics_list[-1].timestamp
    finish_chain = " → ".join(m.finish_reason or "?" for m in metrics_list)
    total_fresh = sum(m.fresh_prefill_tokens for m in metrics_list)
    total_disk = max(
        (m.disk_kv_loaded_tokens or m.disk_cached_tokens_trace for m in metrics_list),
        default=0,
    )
    total_gen = sum(m.generated_tokens for m in metrics_list)
    total_wall = sum(m.total_request_time_s for m in metrics_list)

    lines = [
        f"# Turn {turn_n:03d} — {first_ts} to {last_ts} ({total_wall:.1f}s wall)\n",
        f"{len(metrics_list)} request(s) ({finish_chain}).",
        f"Total fresh prefill: {total_fresh:,} tok."
        f"  Total disk KV reused: {total_disk:,} tok."
        f"  Total generated: {total_gen:,} tok.\n",
    ]

    for m in metrics_list:
        cache_label = (
            f"{m.cache_classification}/{m.cache_source}"
            if m.cache_classification else
            f"miss_pos={m.cache_miss_common}" if m.cache_miss_common >= 0 else "live-continue"
        )
        evict_str = (
            f"  Evict: {m.kv_evict_tokens:,} tok, {m.kv_evict_save_ms:.0f}ms."
            if m.kv_evict_tokens else ""
        )
        kv_ckpt = (
            f" ({m.kv_continued_count} KV checkpoints, {m.kv_continued_save_ms_total:.0f}ms saves)."
            if m.kv_continued_count else "."
        )
        disk_tok = m.disk_kv_loaded_tokens or m.disk_cached_tokens_trace
        disk_str = (
            f"  Disk KV: {disk_tok:,} tok"
            + (f" in {m.disk_kv_load_ms:.0f}ms." if m.disk_kv_loaded_tokens else " (trace).")
            if disk_tok else ""
        )
        lines += [
            f"Request {m.request} ({m.timestamp}): {cache_label},"
            f" ctx {m.start_ctx}..{m.end_ctx}, {m.fresh_prefill_tokens:,} fresh tok.",
            f"  Prefill {m.prefill_duration_s:.1f}s at {m.prefill_rate_avg_tps:.1f} t/s{kv_ckpt}"
            f"{evict_str}{disk_str}",
            f"  Decode: {m.generated_tokens} tok, {m.decode_duration_s:.1f}s,"
            f" {m.decode_rate_avg_tps:.1f} t/s. → {m.finish_reason}\n",
        ]

    summary_path = turn_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def append_to_session_csv(session_dir: str, rows: list[RequestMetrics]) -> None:
    csv_path = Path(session_dir) / "session-requests.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if write_header:
            w.writeheader()
        for m in rows:
            w.writerow({k: getattr(m, k) for k in CSV_COLS})


def write_turn_csv(turn_dir: Path, rows: list[RequestMetrics]) -> None:
    turn_n = rows[0].turn if rows else 0
    csv_path = turn_dir / f"turn_{turn_n:03d}-requests.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for m in rows:
            w.writerow({k: getattr(m, k) for k in CSV_COLS})


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace, cfg: dict) -> int:
    trace_file = args.trace or cfg["trace_file"]
    sessions_dir = cfg["sessions_dir"]
    host = args.host or cfg["host"]

    # establish log anchor
    if args.log_file:
        log_anchor_epoch = 0  # offline: we'll read the whole file
        print(f"Offline mode: log anchor = start of file ({args.log_file})")
    else:
        result = subprocess.run(
            ["ssh", host, "date +%s"], capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"!! SSH date failed: {result.stderr.strip()}", file=sys.stderr)
            return 1
        log_anchor_epoch = int(result.stdout.strip())
        print(f"Log anchor: epoch {log_anchor_epoch} on {host}")

    # establish trace anchor
    if not Path(trace_file).exists():
        print(f"!! Trace file not found: {trace_file}", file=sys.stderr)
        print("   Run: /trace-update  to pull it from gx10 first.", file=sys.stderr)
        return 1

    if args.log_file:
        # offline mode: start trace from beginning so turn reads the whole file
        trace_byte_offset = 0
        trace_next_req = 1
    else:
        with open(trace_file, "rb") as f:
            f.seek(0, 2)  # seek to end
            trace_byte_offset = f.tell()

        # find last request number in trace
        trace_next_req = 1
        with open(trace_file, "rb") as f:
            # read last 4096 bytes to find the tail request number
            tail_size = min(4096, trace_byte_offset)
            f.seek(max(0, trace_byte_offset - tail_size))
            tail = f.read().decode("utf-8", errors="replace")
        for m in TRACE_START_RE.finditer(tail):
            trace_next_req = int(m.group(1)) + 1

    # create session dir
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    session_name = args.out or f"session-{ts}"
    session_dir = str(Path(sessions_dir) / session_name)
    Path(session_dir).mkdir(parents=True, exist_ok=True)

    state = {
        "version": STATE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "trace_file": trace_file,
        "log_file": args.log_file or "",
        "log_anchor_epoch": log_anchor_epoch,
        "trace_byte_offset": trace_byte_offset,
        "trace_next_req": trace_next_req,
        "turn_count": 0,
    }
    save_state(session_dir, state)
    print(f"Session: {session_dir}")
    print(f"Trace offset: {trace_byte_offset} bytes (next req ≥ {trace_next_req})")
    print("Ready. Run `ds4_session.py turn` after your next OpenClaw turn.")
    return 0


def cmd_turn(args: argparse.Namespace, cfg: dict) -> int:
    session_dir = args.session or latest_session(cfg["sessions_dir"])
    if not session_dir:
        print("!! No session found. Run `init` first.", file=sys.stderr)
        return 1

    state = load_state(session_dir)
    turn_n = state["turn_count"] + 1
    host = state.get("host") or cfg["host"]
    trace_file = state.get("trace_file") or cfg["trace_file"]
    log_file_override = args.log_file or state.get("log_file") or ""

    print(f"Processing turn {turn_n:03d} …")

    # --- fetch log lines ---
    if log_file_override:
        log_text = Path(log_file_override).read_text(encoding="utf-8", errors="replace")
        # in offline mode: first turn gets everything; subsequent turns get nothing
        if state["turn_count"] > 0:
            print("!! Offline mode only supports turn 1 (whole file). Subsequent turns need live SSH.")
            log_text = ""
    else:
        log_text = fetch_log_lines(host, state["log_anchor_epoch"])

    # --- read new trace slice ---
    trace_text, new_trace_offset = read_trace_slice(trace_file, state["trace_byte_offset"])

    # --- group log lines into per-request buckets ---
    buckets = group_log_lines(log_text)
    if not buckets:
        print("!! No 'prompt start' lines found in log. Did the turn complete?")
        return 1

    # --- extract trace blocks ---
    trace_blocks = extract_trace_blocks(trace_text)

    # warn if counts diverge
    if len(buckets) != len(trace_blocks):
        print(
            f"!! Log has {len(buckets)} request(s) but trace has {len(trace_blocks)}."
            " Pairing by position (first min(N,M) pairs)."
        )

    n = min(len(buckets), len(trace_blocks)) if trace_blocks else len(buckets)

    # --- build metrics list ---
    metrics_list: list[RequestMetrics] = []
    turn_dir = _turn_dir(session_dir, turn_n)

    for i in range(max(len(buckets), len(trace_blocks) if trace_blocks else 0)):
        req_n = i + 1
        bucket = buckets[i] if i < len(buckets) else None
        tblock = trace_blocks[i] if trace_blocks and i < len(trace_blocks) else None

        if bucket is None:
            print(f"  req {req_n}: trace block present but no log lines — skipping")
            continue

        m = metrics_from_bucket(bucket)
        m.turn = turn_n
        m.request = req_n
        if tblock:
            augment_from_trace(m, tblock)

        # validate timestamp correlation
        if tblock:
            # log: "0613 16:45:23" → "06-13 16:45:23"
            log_ts_norm = m.timestamp[0:2] + "-" + m.timestamp[2:]  # "06-13 16:45:23"
            # trace: "2026-06-13 16:45:23" → take MM-DD HH:MM:SS part
            trace_ts_norm = tblock.req_ts[5:]  # "06-13 16:45:23"
            if log_ts_norm != trace_ts_norm:
                print(f"  !! req {req_n}: timestamp mismatch log={m.timestamp} trace={tblock.req_ts}")

        write_request_files(turn_dir, turn_n, req_n, bucket, tblock)
        metrics_list.append(m)

        cache_label = m.cache_classification or ("live" if m.cache_miss_common < 0 else f"miss@{m.cache_miss_common}")
        print(
            f"  req {req_n}: ctx {m.start_ctx}..{m.end_ctx}"
            f" prefill={m.fresh_prefill_tokens} gen={m.generated_tokens}"
            f" [{cache_label}] finish={m.finish_reason} {m.total_request_time_s:.1f}s"
        )

    if not metrics_list:
        print("!! No metrics extracted.")
        return 1

    write_turn_csv(turn_dir, metrics_list)
    write_turn_summary(turn_dir, turn_n, metrics_list)
    append_to_session_csv(session_dir, metrics_list)

    # --- advance anchors ---
    state["log_anchor_epoch"] = int(time.time())
    state["trace_byte_offset"] = new_trace_offset
    if trace_blocks:
        state["trace_next_req"] = trace_blocks[-1].req_id + 1
    state["turn_count"] = turn_n
    save_state(session_dir, state)

    print(f"  → {turn_dir}")
    print(f"  → {session_dir}/session-requests.csv (cumulative)")
    return 0


def cmd_status(args: argparse.Namespace, cfg: dict) -> int:
    session_dir = args.session or latest_session(cfg["sessions_dir"])
    if not session_dir:
        print("No session found.")
        return 1
    state = load_state(session_dir)
    print(f"Session : {session_dir}")
    print(f"Turns   : {state['turn_count']}")
    print(f"Log anchor epoch : {state['log_anchor_epoch']}")
    print(f"Trace offset     : {state['trace_byte_offset']} bytes")
    print(f"Trace next req   : {state['trace_next_req']}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    cfg = load_config()

    ap = argparse.ArgumentParser(description="DS4 per-turn request extractor")
    ap.add_argument("--host", help="SSH host for docker logs (overrides config)")
    ap.add_argument("--trace", help="local trace file (overrides config)")
    ap.add_argument("--log-file", dest="log_file", help="offline: read logs from file")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="create session, establish anchors")
    pi.add_argument("--out", help="session name (default: session-YYYYMMDD-HHMMSS)")
    pi.set_defaults(fn=cmd_init)

    pt = sub.add_parser("turn", help="extract next turn's data")
    pt.add_argument("--session", help="session dir (default: most recent)")
    pt.set_defaults(fn=cmd_turn)

    ps = sub.add_parser("status", help="show current session state")
    ps.add_argument("--session", help="session dir (default: most recent)")
    ps.set_defaults(fn=cmd_status)

    args = ap.parse_args()
    return args.fn(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
