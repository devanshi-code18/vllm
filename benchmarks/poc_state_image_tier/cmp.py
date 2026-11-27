#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare two rr_replay out.jsonl runs turn by turn: identical text, first divergence,
cached-token difference, and (with --logprobs runs) max |dlogprob| before divergence."""

import json
import sys


def load(p):
    return {(r["agent"], r["turn"]): r for r in map(json.loads, open(p))}


A, B = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(A) & set(B))
same = 0
diffs = []
maxd = 0.0
ndl = 0
for k in keys:
    a, b = A[k], B[k]
    if a["text"] == b["text"]:
        same += 1
    else:
        i = next(
            (j for j, (x, y) in enumerate(zip(a["text"], b["text"])) if x != y),
            min(len(a["text"]), len(b["text"])),
        )
        diffs.append((k, i, a["cached_tokens"], b["cached_tokens"]))
    la, lb = a.get("logprobs"), b.get("logprobs")
    if la and lb:
        for (ta, pa), (tb, pb) in zip(la, lb):
            if ta != tb:
                break
            if pa is not None and pb is not None:
                maxd = max(maxd, abs(pa - pb))
                ndl += 1
print(
    f"{sys.argv[1]} vs {sys.argv[2]}: {same}/{len(keys)} turns identical text; "
    f"cached_tokens equal on {sum(A[k]['cached_tokens'] == B[k]['cached_tokens'] for k in keys)}/{len(keys)}"
    + (f"; max|dlogprob| {maxd:.3g} over {ndl} matched tokens" if ndl else "")
)
for k, i, ca, cb in diffs[:12]:
    print(f"  diff a{k[0]} t{k[1]}: first char {i}, cached {ca} vs {cb}")
