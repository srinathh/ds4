#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai"]
# ///
"""Simple multi-turn chat CLI for a ds4-server OpenAI-compatible endpoint.

Sends thinking: {"type": "disabled"} (via extra_body) and model="deepseek-chat"
so the server keeps thinking off for every turn.

Usage:
    python3 scripts/chat_ds4.py [--base-url URL]

Defaults to http://192.168.0.250:8000/v1.
Press Ctrl-C or Ctrl-D to exit.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = "You are a helpful assistant."
MODEL = "deepseek-chat"
THINKING_OFF = {"type": "disabled"}

def write_turn(logfile, turn: int, ts_req: str, ts_resp: str, messages_sent: list, response: str, error: str = None):
    entry = {
        "turn": turn,
        "ts_request": ts_req,
        "ts_response": ts_resp,
        "model": MODEL,
        "thinking": THINKING_OFF,
        "messages_sent": messages_sent,
        "response": response,
    }
    if error:
        entry["error"] = error
    logfile.write(json.dumps(entry) + "\n")
    logfile.flush()

def main():
    parser = argparse.ArgumentParser(description="Chat with ds4-server")
    parser.add_argument("--base-url", default="http://192.168.0.250:8000/v1")
    parser.add_argument("--log", default="ds4-chat.log", type=str, help="log file path (default: ds4-chat.log)")
    args = parser.parse_args()

    log_path = Path(args.log)
    logfile = log_path.open("a")

    client = OpenAI(api_key="ds4-local", base_url=args.base_url)
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    turn = 0

    print(f"Connected to {args.base_url}  (model={MODEL}, thinking=disabled)")
    print(f"Logging to {log_path.resolve()}")
    print("Ctrl-C or Ctrl-D to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})
        turn += 1
        ts_req = datetime.now(timezone.utc).isoformat()

        print("Assistant: ", end="", flush=True)
        collected = []
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=history,
                stream=True,
                extra_body={"thinking": THINKING_OFF},
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    collected.append(delta)
        except Exception as exc:
            ts_resp = datetime.now(timezone.utc).isoformat()
            write_turn(logfile, turn, ts_req, ts_resp, list(history), "", error=str(exc))
            print(f"\n[error: {exc}]")
            history.pop()
            continue

        response_text = "".join(collected)
        ts_resp = datetime.now(timezone.utc).isoformat()
        print()
        write_turn(logfile, turn, ts_req, ts_resp, list(history), response_text)
        history.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()
