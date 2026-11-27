#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize rr_replay out.jsonl files: TTFT p50/p90 overall and by turn-depth bin, recompute tokens."""

import json
import statistics as st
import sys


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


for path in sys.argv[1:]:
    R = [json.loads(l) for l in open(path)]
    ok = [r for r in R if r["ttft_s"] is not None and not r["error"]]
    errs = len(R) - len(ok)
    tt = [r["ttft_s"] for r in ok]
    first = [r for r in ok if r["turn"] == 0]
    later = [r for r in ok if r["turn"] > 0]
    newtok = [r["prompt_tokens"] - (r["cached_tokens"] or 0) for r in later]
    wall = (
        (max(r["t_start"] + r["total_s"] for r in R) - min(r["t_start"] for r in R))
        if R
        else 0
    )
    ctoks = sum(r["completion_tokens"] or 0 for r in ok)
    ptoks = sum(r["prompt_tokens"] or 0 for r in ok)
    print(
        f"== {path}: n={len(R)} err={errs} wall={wall:.0f}s out_tok/s={ctoks / wall:.0f}"
    )
    print(
        f"   TTFT all p50 {pct(tt, 50) * 1e3:.0f} p90 {pct(tt, 90) * 1e3:.0f} ms | turn>0 p50 {pct([r['ttft_s'] for r in later], 50) * 1e3:.0f} p90 {pct([r['ttft_s'] for r in later], 90) * 1e3:.0f} ms"
        f" | uncached tok/turn>0 p50 {pct(newtok, 50):.0f} mean {st.mean(newtok) if newtok else 0:.0f}"
    )
    bins = [(0, 0), (1, 4), (5, 9), (10, 14), (15, 99)]
    for lo, hi in bins:
        b = [r for r in ok if lo <= r["turn"] <= hi]
        if not b:
            continue
        nt = [r["prompt_tokens"] - (r["cached_tokens"] or 0) for r in b]
        print(
            f"   turns {lo:2d}-{hi:2d}: n={len(b):3d} ctx~{st.mean(r['prompt_tokens'] for r in b) / 1e3:5.1f}k  TTFT p50 {pct([r['ttft_s'] for r in b], 50) * 1e3:7.0f} p90 {pct([r['ttft_s'] for r in b], 90) * 1e3:7.0f} ms  uncached p50 {pct(nt, 50):6.0f}"
        )
