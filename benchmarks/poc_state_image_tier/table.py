#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""One row per run: TTFT p50/p90 over turns > 0 and by depth, uncached tokens, tok/s,
CPU tier bytes and restore bandwidth (from serve.log / metrics_after.txt).
Usage: table.py RESULTS_DIR run1 run2 ..."""

import json
import os
import re
import sys


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def metric(text, name, label=None):
    for line in text.splitlines():
        if line.startswith(name + "{") and (label is None or label in line):
            return float(line.rsplit(" ", 1)[1])
    return None


root = sys.argv[1]
bins = [(1, 4), (5, 9), (10, 14), (15, 99)]
hdr = (
    f"{'run':28s} {'n':>4s} {'p50':>6s} {'p90':>6s} | "
    + " ".join(f"{f'p50 t{a}-{b}':>10s}" for a, b in bins)
    + f" | {'uncach':>6s} {'tok/s':>6s} {'wall':>5s} | cpu_GiB  restore"
)
print(hdr)
for run in sys.argv[2:]:
    d = os.path.join(root, run)
    try:
        R = [json.loads(l) for l in open(os.path.join(d, "out.jsonl"))]
    except FileNotFoundError:
        print(f"{run:28s} (missing)")
        continue
    ok = [r for r in R if r["ttft_s"] is not None and not r["error"]]
    later = [r for r in ok if r["turn"] > 0]
    tt = [r["ttft_s"] * 1e3 for r in later]
    unc = [r["prompt_tokens"] - (r["cached_tokens"] or 0) for r in later]
    wall = max(r["t_start"] + r["total_s"] for r in R) - min(r["t_start"] for r in R)
    tps = sum(r["completion_tokens"] or 0 for r in ok) / wall
    cols = []
    for a, b in bins:
        x = [r["ttft_s"] * 1e3 for r in ok if a <= r["turn"] <= b]
        cols.append(f"{pct(x, 50):10.0f}")
    serve = (
        open(os.path.join(d, "serve.log"), errors="ignore").read()
        if os.path.exists(os.path.join(d, "serve.log"))
        else ""
    )
    met = (
        open(os.path.join(d, "metrics_after.txt")).read()
        if os.path.exists(os.path.join(d, "metrics_after.txt"))
        else ""
    )
    cpu = rest = ""
    m = re.findall(r"StateImageTier: used (\d+)/(\d+) slots \(([\d.]+) GiB\)", serve)
    if m:
        cpu = f"{float(m[-1][2]):5.1f} (max {max(float(x[2]) for x in m):.1f})"
    w = re.findall(r"loaded ([\d.]+) GB in ([\d.]+) s \(([\d.]+) GB/s\)", serve)
    if w:
        rest = f"{float(w[-1][2]):.1f} GB/s"
    lb = metric(met, "vllm:kv_offload_load_bytes_total")
    lt = metric(met, "vllm:kv_offload_load_time_total")
    if lb and lt:
        rest = f"{lb / lt / 1e9:.1f} GB/s"
        sb = metric(met, "vllm:kv_offload_store_bytes_total") or 0
        cpu = f"stored {sb / 2**30:.1f}"
    err = len(R) - len(ok)
    print(
        f"{run:28s} {len(R):4d} {pct(tt, 50):6.0f} {pct(tt, 90):6.0f} | {' '.join(cols)} | {pct(unc, 50):6.0f} {tps:6.0f} {wall:5.0f} | {cpu} {rest}"
        + (f" ERR={err}" if err else "")
    )
