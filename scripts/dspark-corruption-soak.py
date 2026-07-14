#!/usr/bin/env python3
"""Concurrent multi-turn corruption soak for DSpark + DeepSeek-V4-Flash.

Reproduces and detects the "speculation + concurrency" corruption documented
in the repo README: it drives several independent multi-turn agent-style
sessions at once against an OpenAI-compatible vLLM endpoint, and flags the
known corruption signatures per turn. It is the fresh-session verifier the
README's validation gotcha calls for -- each worker uses its own conversation
history, so a healthy server is never blamed for a client replaying leaked
markers.

Use it to compare configurations/stacks (fork vs. upstream): same flags, same
endpoint shape, same detectors. stdlib only.

    python3 scripts/dspark-corruption-soak.py \
        --base-url http://127.0.0.1:8888/v1 \
        --model deepseek-v4-flash-dspark \
        --concurrency 4 --turns 12 --temperature 0.6

Exit code 0 = no corruption detected, 2 = corruption detected, 1 = run error.
A clean run at concurrency 1 plus a dirty run at concurrency >=2 is the
fingerprint of the speculation/concurrency bug.
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request

# --- Corruption detectors -------------------------------------------------
# Each returns a short reason string if the text looks corrupted, else "".

# Literal control/special markers that must never appear as plain text.
_LEAKED_MARKERS = [
    "<|begin_of_sentence|>",
    "<|end_of_sentence|>",
    "<|User|>",
    "<|Assistant|>",
    "<｜begin▁of▁sentence｜>",  # fullwidth variant seen in DeepSeek tokenizers
    "<｜end▁of▁sentence｜>",
]
# Tool/markup that should have been parsed into tool_calls, not emitted raw.
_MARKUP_LEAK = re.compile(
    r"<\|tool▁calls?▁begin\|>|<\｜tool▁call▁begin\｜>|<tool_call>|"
    r"<\|tool_call\|>|```json\s*\{\s*\"name\"",
    re.IGNORECASE,
)
# CJK, Hangul, Greek, Cyrillic -- none expected in English coding answers.
_UNEXPECTED_SCRIPT = re.compile(
    r"[一-鿿぀-ヿ가-힯Ͱ-ϿЀ-ӿ]"
)


def detect_leaked_marker(text: str) -> str:
    for m in _LEAKED_MARKERS:
        if m in text:
            return f"leaked control marker {m!r}"
    return ""


def detect_markup_leak(text: str) -> str:
    hit = _MARKUP_LEAK.search(text)
    return f"raw tool markup {hit.group(0)!r}" if hit else ""


def detect_unexpected_script(text: str, threshold: int = 3) -> str:
    hits = _UNEXPECTED_SCRIPT.findall(text)
    if len(hits) >= threshold:
        sample = "".join(hits[:8])
        return f"{len(hits)} unexpected-script chars (e.g. {sample!r})"
    return ""


def detect_char_repeat(text: str, run: int = 40) -> str:
    # A single char/short cycle repeated many times = degenerate decode.
    m = re.search(r"(.{1,4}?)\1{" + str(run) + r",}", text)
    return f"repeated fragment {m.group(1)!r}" if m else ""


def _longest_common_substring(a: str, b: str) -> str:
    """Longest common substring (rolling window; inputs here are small)."""
    if not a or not b:
        return ""
    prev = [0] * (len(b) + 1)
    best_len = best_end = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best_len:
                    best_len, best_end = cur[j], i
        prev = cur
    return a[best_end - best_len : best_end]


def detect_replay(
    prev_turns: list[tuple[str, str]],
    task: str,
    text: str,
    min_block: int = 120,
    min_fraction: float = 0.6,
) -> str:
    """A long, distinctive verbatim block copied from an answer to a DIFFERENT
    question -- the stale-prefix / context-jump signature.

    Flags either an absolutely long shared block (``min_block`` chars, well
    beyond stock phrasing) or a shared block that makes up most of the answer
    (``min_fraction``), which catches wholesale replay of a short answer.

    Two false-positive sources are excluded deliberately:
      * repeating the same question legitimately yields the same answer, so
        same-task pairs are skipped;
      * models reuse boilerplate ("Here is a concise answer...", common code),
        so a merely shared sentence never counts.
    """
    norm = " ".join(text.split())
    if len(norm) < 40:
        return ""
    for i, (prev_task, prev_answer) in enumerate(prev_turns):
        if prev_task == task:
            continue
        pnorm = " ".join(prev_answer.split())
        if len(pnorm) < 40:
            continue
        shared = len(_longest_common_substring(norm, pnorm))
        if shared >= min_block or shared >= min_fraction * len(norm):
            return (
                f"verbatim replay of turn #{i + 1} (different question): "
                f"{shared} identical chars ({shared / len(norm):.0%} of answer)"
            )
    return ""


def scan(text: str, prev_turns: list[tuple[str, str]], task: str) -> list[str]:
    reasons = []
    for det in (
        detect_leaked_marker,
        detect_markup_leak,
        detect_unexpected_script,
        detect_char_repeat,
    ):
        r = det(text)
        if r:
            reasons.append(r)
    r = detect_replay(prev_turns, task, text)
    if r:
        reasons.append(r)
    return reasons


# --- Prompt material (varied lengths => varied query segments) ------------

_TASKS = [
    "Write a Python function that merges two sorted lists. Explain briefly.",
    "Refactor this to use a dict comprehension: "
    + "d={}\nfor k,v in pairs:\n    d[k]=v*2",
    "Summarize the tradeoffs between mutexes and channels in one paragraph.",
    "Give me a regex for an ISO-8601 date and one test string.",
    "Write a short haiku about garbage collection, then a 2-line explanation.",
    "Explain what a bloom filter is and when you would not use one.",
    "Translate this pseudocode to Rust: for i in 0..n { sum += a[i] }",
    "What is tail-call optimization? Answer in exactly three sentences.",
    # A long turn to force chunked prefill interleaving with others' decodes:
    "Here is a log excerpt:\n"
    + ("2026-07-14T10:00:00 INFO request served in 42ms\n" * 60)
    + "List three things you can infer about this service.",
]


def _post_chat(
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[str, int]:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    msg = data["choices"][0]["message"]
    text = msg.get("content") or ""
    n = int(data.get("usage", {}).get("completion_tokens", 0))
    return text, n


def worker(
    wid: int,
    args: argparse.Namespace,
    results: queue.Queue,
) -> None:
    # Fresh session: a system turn + independent multi-turn history.
    messages = [
        {
            "role": "system",
            "content": "You are a terse, helpful coding assistant. "
            "Answer in English.",
        }
    ]
    prev_turns: list[tuple[str, str]] = []  # (task, answer)
    findings = []
    total_tokens = 0
    t0 = time.monotonic()
    for turn in range(args.turns):
        task = _TASKS[(wid + turn) % len(_TASKS)]
        messages.append({"role": "user", "content": task})
        try:
            text, ntok = _post_chat(
                args.base_url,
                args.model,
                messages,
                args.max_tokens,
                args.temperature,
                args.timeout,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            results.put(("error", wid, turn, f"request failed: {e}"))
            return
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            results.put(("error", wid, turn, f"bad response: {e}"))
            return
        total_tokens += ntok
        reasons = scan(text, prev_turns, task)
        if reasons:
            findings.append((turn, reasons, text[:200]))
        messages.append({"role": "assistant", "content": text})
        prev_turns.append((task, text))
    elapsed = time.monotonic() - t0
    results.put(("done", wid, total_tokens, elapsed, findings))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="deepseek-v4-flash-dspark")
    ap.add_argument(
        "--concurrency", type=int, default=4, help="parallel sessions"
    )
    ap.add_argument(
        "--turns", type=int, default=12, help="turns per session"
    )
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument(
        "--label", default="", help="tag for this run in the summary"
    )
    args = ap.parse_args()

    tag = f" [{args.label}]" if args.label else ""
    print(
        f"soak{tag}: {args.concurrency} sessions x {args.turns} turns, "
        f"temperature={args.temperature}, model={args.model}",
        flush=True,
    )

    results: queue.Queue = queue.Queue()
    threads = [
        threading.Thread(target=worker, args=(wid, args, results))
        for wid in range(args.concurrency)
    ]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - start

    total_tokens = 0
    all_findings = []
    errors = []
    while not results.empty():
        item = results.get()
        if item[0] == "error":
            errors.append(item[1:])
        else:
            _, wid, ntok, _elapsed, findings = item
            total_tokens += ntok
            for turn, reasons, sample in findings:
                all_findings.append((wid, turn, reasons, sample))

    print(f"\n--- summary{tag} ---")
    print(f"wall: {wall:.1f}s  aggregate decode: {total_tokens} tokens")
    if wall > 0:
        print(f"aggregate throughput: {total_tokens / wall:.1f} tok/s")
    if errors:
        print(f"\nRUN ERRORS ({len(errors)}):")
        for wid, turn, msg in errors[:10]:
            print(f"  session {wid} turn {turn}: {msg}")
        if not all_findings:
            return 1

    if all_findings:
        print(f"\nCORRUPTION DETECTED: {len(all_findings)} bad turn(s)")
        for wid, turn, reasons, sample in all_findings[:20]:
            print(f"  session {wid} turn {turn}: {'; '.join(reasons)}")
            print(f"    sample: {sample!r}")
        return 2

    print("\nno corruption detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
