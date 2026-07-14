#!/usr/bin/env python3
"""Is the served sampling actually probabilistic, or forced greedy?

Phase 5 decision rule 2 says "prefer probabilistic sampling over greedy-forced".
Two configs look superficially similar but are not:

  * fork F3 (GREEDY_VERIFICATION=1) passes
    --override-generation-config {"temperature":0.0,"top_p":1.0}, which
    overrides the *served* sampling for every client. Greedy-forced.
  * upstream U3 (use_local_argmax_reduction) makes only the *draft* proposals
    greedy. Rejection sampling still corrects to the target distribution, so a
    client's temperature should still be honoured.

Empirical test: send the same prompt N times at a high temperature. If every
completion is byte-identical, sampling is dead (greedy-forced). If they vary,
the client's temperature is live.

Exit 0 = sampling live (probabilistic), 3 = greedy-forced (all identical).
"""
import argparse
import json
import sys
import urllib.request

PROMPT = "Invent a name for a coffee shop and give a one-line slogan."


def one(base_url: str, model: str, temperature: float) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 40,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        d = json.loads(resp.read())
    return (d["choices"][0]["message"].get("content") or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="deepseek-v4-flash-dspark")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    outs = [one(a.base_url, a.model, a.temperature) for _ in range(a.samples)]
    uniq = sorted(set(outs))
    print(f"label: {a.label}  temperature={a.temperature}  samples={a.samples}")
    for i, o in enumerate(outs):
        print(f"  [{i}] {o[:80]!r}")
    print(f"distinct completions: {len(uniq)}/{a.samples}")
    if len(uniq) == 1:
        print("VERDICT: GREEDY-FORCED (client temperature ignored)")
        return 3
    print("VERDICT: sampling live (client temperature honoured)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
