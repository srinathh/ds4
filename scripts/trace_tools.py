# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""trace_tools.py — per-request extraction from DS4 --trace slices.

Context: investigating KV/prefix cache mismatches between OpenClaw and the DS4
server (see docs/plans/we-are-going-to-luminous-hennessy.md). The DS4 trace
format is authoritative — for every request it writes a `--- cache decision ---`
block (live_prompt_common, memory_miss_reason, cache_source, disk_cached_tokens)
and, on a real mismatch, a `first_mismatch_token` token window showing
`live <id> "text" | prompt <id> "text"` with whitespace escaped. We parse that
verbatim rather than re-deriving it. Trace format: ds4_server.c:9091-9297.

Context hygiene (the whole point): this tool writes the bulky per-request
artifacts (full rendered prompt, raw JSON) to files for a human to eyeball, and
prints ONLY bounded diagnostics to stdout (a one-line summary per request, plus
the token window and a 400-char rendered window for each MISS). The calling
agent reads stdout only — never the rendered/raw artifacts.

Subcommands:
  extract  --turn N --slice F --out D   parse one turn's trace slice
  taxonomy --out D                       roll up index/*.csv into taxonomy.md

Run:  uv run scripts/trace_tools.py extract --turn 2 \
          --slice data/cache-probe/raw-slices/turn-02.slice.txt --out data/cache-probe
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

# --- exact trace markers (ds4_server.c) --------------------------------------
REQ_START_RE = re.compile(r"^===== request (\d+) (.+?) =====$", re.M)
M_CACHE = "\n--- cache decision ---\n"
M_RAWJSON = "\n--- raw request json ---\n"
M_RENDERED = "\n--- rendered prompt ---\n"
M_GENERATED = "\n--- generated text ---\n"
M_FIRSTMISS = "\nfirst_mismatch_token:"

# 400-char window = CTX_CHARS before + CTX_CHARS after the divergence char.
CTX_CHARS = 200


def _field(text: str, name: str) -> str | None:
    """Return the value of a `name: value` line, first occurrence."""
    m = re.search(rf"^{re.escape(name)}:[ \t]*(.*)$", text, re.M)
    return m.group(1).strip() if m else None


def _int(text: str, name: str, default: int = 0) -> int:
    v = _field(text, name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


def _first_marker(block: str, markers, start: int = 0) -> int:
    """Index of the earliest of `markers` at/after start, or len(block)."""
    best = len(block)
    for mk in markers:
        i = block.find(mk, start)
        if i != -1 and i < best:
            best = i
    return best


def _unescape(s: str) -> str:
    """Inverse of trace_write_escaped_bytes (ds4_server.c:9066)."""
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            if n == "n":
                out.append("\n"); i += 2; continue
            if n == "r":
                out.append("\r"); i += 2; continue
            if n == "t":
                out.append("\t"); i += 2; continue
            if n in ('"', "\\"):
                out.append(n); i += 2; continue
            if n == "x" and i + 3 < len(s):
                try:
                    out.append(chr(int(s[i + 2 : i + 4], 16))); i += 4; continue
                except ValueError:
                    pass
        out.append(c)
        i += 1
    return "".join(out)


@dataclass
class TokenRow:
    pos: int
    mark: str
    live_id: int
    live_text: str
    prompt_id: int
    prompt_text: str


@dataclass
class Request:
    turn: int
    seq: int  # req-MM within the turn (1-based)
    trace_id: str = ""
    req_ts: str = ""  # wall-clock from the `===== request <id> <ts> =====` line
    kind: str = ""
    model: str = ""
    tools: int = 0
    think_mode: str = ""
    prompt_tokens: int = 0
    effective_prompt_tokens: int = 0
    cached_tokens: int = 0
    # cache-decision block
    live_tokens_before: int = 0
    live_prompt_common: int = 0
    memory_miss_reason: str = ""
    cache_source: str = "none"
    disk_cached_tokens: int = 0
    disk_cache_file: str = ""
    first_mismatch_token: int | None = None
    token_window: str = ""
    rows: list[TokenRow] = field(default_factory=list)
    rendered: str = ""
    rawjson: str = ""

    @property
    def classification(self) -> str:
        r = self.memory_miss_reason
        if r == "live-prefix-match":
            return "HIT"
        if r == "no-live-checkpoint":
            return "COLD"
        return "MISS"  # token-mismatch / shorter-than-live / etc.


# --- token-row parsing -------------------------------------------------------
ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\S+)\s+live\s+(.*?)\s+\|\s+prompt\s+(.*)$"
)


def _parse_token(tok: str) -> tuple[int, str]:
    """`198 "\\n"` -> (198, '\n');  `- <none>` -> (-1, '')."""
    tok = tok.strip()
    if tok.startswith("- <none>"):
        return -1, ""
    m = re.match(r'^(-?\d+)\s+"(.*)"$', tok)
    if not m:
        return -1, ""
    return int(m.group(1)), _unescape(m.group(2))


def _parse_window(window_block: str) -> list[TokenRow]:
    rows: list[TokenRow] = []
    for line in window_block.splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        pos = int(m.group(1))
        mark = m.group(2)
        lid, ltext = _parse_token(m.group(3))
        pid, ptext = _parse_token(m.group(4))
        rows.append(TokenRow(pos, mark, lid, ltext, pid, ptext))
    return rows


# --- request block parsing ---------------------------------------------------
def parse_request(turn: int, seq: int, block: str) -> Request:
    req = Request(turn=turn, seq=seq)
    m = REQ_START_RE.match(block)
    if m:
        req.trace_id = m.group(1)
        req.req_ts = m.group(2)

    cache_at = block.find(M_CACHE)
    head = block[: cache_at if cache_at != -1 else len(block)]
    req.kind = _field(head, "kind") or ""
    req.model = _field(head, "model") or ""
    req.tools = _int(head, "tools")
    req.think_mode = _field(head, "think_mode") or ""
    req.prompt_tokens = _int(head, "prompt_tokens")
    req.effective_prompt_tokens = _int(head, "effective_prompt_tokens")
    req.cached_tokens = _int(head, "cached_tokens")

    # cache-decision block: from M_CACHE to the first following section marker.
    if cache_at != -1:
        cache_end = _first_marker(
            block, [M_RAWJSON, M_RENDERED, M_GENERATED], cache_at + len(M_CACHE)
        )
        cdec = block[cache_at:cache_end]
        req.live_tokens_before = _int(cdec, "live_tokens_before")
        req.live_prompt_common = _int(cdec, "live_prompt_common")
        req.memory_miss_reason = _field(cdec, "memory_miss_reason") or ""
        req.cache_source = _field(cdec, "cache_source") or "none"
        req.disk_cached_tokens = _int(cdec, "disk_cached_tokens")
        req.disk_cache_file = _field(cdec, "disk_cache_file") or ""
        fm = _field(cdec, "first_mismatch_token")
        if fm is not None:
            try:
                req.first_mismatch_token = int(fm)
            except ValueError:
                pass
            req.token_window = _field(cdec, "token_window") or ""
            req.rows = _parse_window(cdec)

    # raw json: M_RAWJSON .. (M_RENDERED | M_GENERATED)
    rj = block.find(M_RAWJSON)
    if rj != -1:
        start = rj + len(M_RAWJSON)
        end = _first_marker(block, [M_RENDERED, M_GENERATED], start)
        req.rawjson = block[start:end].rstrip("\n")

    # rendered prompt: M_RENDERED .. M_GENERATED
    rp = block.find(M_RENDERED)
    if rp != -1:
        start = rp + len(M_RENDERED)
        end = block.find(M_GENERATED, start)
        if end == -1:
            end = len(block)
        req.rendered = block[start:end]
        if req.rendered.endswith("\n"):
            req.rendered = req.rendered[:-1]

    return req


def parse_slice(turn: int, text: str) -> list[Request]:
    starts = list(REQ_START_RE.finditer(text))
    reqs: list[Request] = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        block = text[m.start() : end]
        reqs.append(parse_request(turn, i + 1, block))
    return reqs


# --- 400-char window location ------------------------------------------------
def locate_window(req: Request) -> tuple[int, str, int] | None:
    """Anchor the divergence in the rendered prompt and slice +/-CTX_CHARS.

    Primary anchor: concatenate the prompt-side token texts straddling the
    mismatch (a few matching tokens before `common`, plus the diverging tokens
    at/after `common`) and find that exact substring in the rendered prompt.
    Token pieces concatenate back to the source text, so this is byte-exact.
    Returns (anchor_char_index, window_text, marker_offset) or None, where
    marker_offset is the divergence position *within* window_text (it is not a
    fixed CTX_CHARS when the window clamps at the start of the prompt).
    """
    if not req.rendered or req.first_mismatch_token is None or not req.rows:
        return None
    common = req.first_mismatch_token
    pre = "".join(
        r.prompt_text for r in req.rows if r.prompt_id >= 0 and common - 4 <= r.pos < common
    )
    post = "".join(
        r.prompt_text for r in req.rows if r.prompt_id >= 0 and r.pos >= common
    )
    needle = pre + post
    anchor = None
    if needle:
        idx = req.rendered.find(needle)
        if idx != -1:
            anchor = idx + len(pre)
    if anchor is None and post:  # fall back to the diverging side alone
        idx = req.rendered.rfind(post)  # rfind: the divergence is late in prompt
        if idx != -1:
            anchor = idx
    if anchor is None:
        return None
    lo = max(0, anchor - CTX_CHARS)
    hi = min(len(req.rendered), anchor + CTX_CHARS)
    return anchor, req.rendered[lo:hi], anchor - lo


# --- artifact writing + stdout ----------------------------------------------
def _name(req: Request) -> str:
    return f"turn-{req.turn:02d}.req-{req.seq:02d}"


def write_artifacts(req: Request, out_dir: str) -> dict:
    base = os.path.join(out_dir, _name(req))
    if req.rendered:
        with open(base + ".rendered.txt", "w", encoding="utf-8") as f:
            f.write(req.rendered)
    if req.rawjson:
        with open(base + ".raw.json", "w", encoding="utf-8") as f:
            f.write(req.rawjson)

    loc = locate_window(req) if req.classification == "MISS" else None
    window_found = loc is not None
    if req.classification == "MISS":
        with open(base + ".window.txt", "w", encoding="utf-8") as f:
            f.write(f"# {_name(req)}  trace_id={req.trace_id}\n")
            f.write(
                f"# reason={req.memory_miss_reason} common={req.live_prompt_common} "
                f"prompt_tokens={req.prompt_tokens} cache_source={req.cache_source} "
                f"disk_cached={req.disk_cached_tokens}\n\n"
            )
            f.write(f"--- token mismatch window {req.token_window} ---\n")
            for r in req.rows:
                f.write(
                    f"{r.pos:>8} {r.mark:<11} live {r.live_id} {r.live_text!r}"
                    f"  |  prompt {r.prompt_id} {r.prompt_text!r}\n"
                )
            f.write("\n--- 400-char rendered window (centered on divergence) ---\n")
            if loc:
                anchor, win, off = loc
                f.write(f"[anchor char={anchor}]\n")
                f.write(win[:off] + "  <<<MISMATCH>>>  " + win[off:])
                f.write("\n")
            else:
                f.write("(could not locate divergence in rendered prompt)\n")

    return {"window_found": window_found, "loc": loc}


def _print_summary(req: Request, info: dict) -> None:
    cls = req.classification
    head = (
        f"{_name(req)}  [{req.req_ts}]  {cls:<4} kind={req.kind} tools={req.tools} "
        f"prompt={req.prompt_tokens} eff={req.effective_prompt_tokens} "
        f"cached={req.cached_tokens} common={req.live_prompt_common} "
        f"reason={req.memory_miss_reason} source={req.cache_source} "
        f"disk={req.disk_cached_tokens}"
    )
    print(head)
    if cls != "MISS":
        return
    # token window (bounded ~17 rows) — whitespace made visible via repr()
    print(f"    token window {req.token_window}:")
    for r in req.rows:
        flag = " <<<" if r.mark == "!=" else ""
        print(
            f"      {r.pos:>8} {r.mark:<11} live {r.live_text!r:<14} "
            f"prompt {r.prompt_text!r}{flag}"
        )
    loc = info["loc"]
    if loc:
        anchor, win, off = loc
        print("    400-char rendered window (| marks divergence):")
        body = win[:off] + " |<<<>>>| " + win[off:]
        print("      " + repr(body))
    else:
        print("    (divergence not locatable in rendered prompt; see token window)")


CSV_COLS = [
    "turn", "req", "trace_id", "req_ts", "kind", "tools", "think_mode",
    "prompt_tokens", "effective_prompt_tokens", "cached_tokens",
    "live_prompt_common", "memory_miss_reason", "cache_source",
    "disk_cached_tokens", "classification", "window_found",
]


def _csv_row(req: Request, info: dict) -> dict:
    return {
        "turn": req.turn, "req": req.seq, "trace_id": req.trace_id,
        "req_ts": req.req_ts,
        "kind": req.kind, "tools": req.tools, "think_mode": req.think_mode,
        "prompt_tokens": req.prompt_tokens,
        "effective_prompt_tokens": req.effective_prompt_tokens,
        "cached_tokens": req.cached_tokens,
        "live_prompt_common": req.live_prompt_common,
        "memory_miss_reason": req.memory_miss_reason,
        "cache_source": req.cache_source,
        "disk_cached_tokens": req.disk_cached_tokens,
        "classification": req.classification,
        "window_found": int(info["window_found"]),
    }


# --- subcommands -------------------------------------------------------------
def cmd_extract(args: argparse.Namespace) -> int:
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "index"), exist_ok=True)

    with open(args.slice, "rb") as f:
        text = f.read().decode("utf-8", errors="replace")

    reqs = parse_slice(args.turn, text)
    if not reqs:
        print(f"!! no '===== request' blocks found in {args.slice}", file=sys.stderr)
        return 2

    rows = []
    print(f"== turn {args.turn}: {len(reqs)} request(s) ==")
    for req in reqs:
        # Turn 1 is a cold prefill baseline — save artifacts, skip MISS analysis.
        if args.turn == 1:
            req.memory_miss_reason = req.memory_miss_reason or "no-live-checkpoint"
        info = write_artifacts(req, out_dir)
        _print_summary(req, info)
        rows.append(_csv_row(req, info))

    idx_path = os.path.join(out_dir, "index", f"turn-{args.turn:02d}.csv")
    with open(idx_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)
    n_miss = sum(1 for r in rows if r["classification"] == "MISS")
    print(f"== wrote {len(rows)} rows -> {idx_path}  ({n_miss} MISS) ==")
    return 0


def cmd_taxonomy(args: argparse.Namespace) -> int:
    out_dir = args.out
    paths = sorted(glob.glob(os.path.join(out_dir, "index", "turn-*.csv")))
    if not paths:
        print("!! no index/turn-*.csv files yet", file=sys.stderr)
        return 2
    all_rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            all_rows.extend(csv.DictReader(f))

    by_reason = Counter(r["memory_miss_reason"] or "(none)" for r in all_rows)
    by_class = Counter(r["classification"] for r in all_rows)
    examples: dict[str, str] = {}
    for r in all_rows:
        reason = r["memory_miss_reason"] or "(none)"
        examples.setdefault(reason, f"turn-{int(r['turn']):02d}.req-{int(r['req']):02d}")

    lines = ["# DS4 cache-mismatch taxonomy", "",
             f"Requests analyzed: {len(all_rows)}  "
             f"(turns: {len({r['turn'] for r in all_rows})})", "",
             "## By classification", ""]
    for cls, n in by_class.most_common():
        lines.append(f"- {cls}: {n}")
    lines += ["", "## By memory_miss_reason", "",
              "| reason | count | example |", "|---|---|---|"]
    for reason, n in by_reason.most_common():
        lines.append(f"| {reason} | {n} | {examples.get(reason, '')} |")
    out = "\n".join(lines) + "\n"

    tax_path = os.path.join(out_dir, "taxonomy.md")
    with open(tax_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(out)
    print(f"== wrote {tax_path} ==")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="DS4 trace per-request extractor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="parse one turn's trace slice")
    pe.add_argument("--turn", type=int, required=True)
    pe.add_argument("--slice", required=True)
    pe.add_argument("--out", default="data/cache-probe")
    pe.set_defaults(fn=cmd_extract)

    pt = sub.add_parser("taxonomy", help="roll up index/*.csv")
    pt.add_argument("--out", default="data/cache-probe")
    pt.set_defaults(fn=cmd_taxonomy)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
