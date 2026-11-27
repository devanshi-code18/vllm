# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import torch


def bw(n_gb=2, reps=5):
    n = int(n_gb * (1 << 30))
    h = torch.empty(n, dtype=torch.int8, pin_memory=True)
    d = torch.empty(n, dtype=torch.int8, device="cuda")
    h.fill_(1)
    for direction in ("H2D", "D2H"):
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            (
                d.copy_(h, non_blocking=True)
                if direction == "H2D"
                else h.copy_(d, non_blocking=True)
            )
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        print(f"{direction} {n / min(ts) / 1e9:.1f} GB/s (best of {reps}, {n_gb} GiB)")
    # 15 MiB chunks (one Nemotron block), 128 of them, like a restore
    c = 15 << 20
    torch.cuda.synchronize()
    t = time.perf_counter()
    for i in range(128):
        d[i * c : (i + 1) * c].copy_(h[i * c : (i + 1) * c], non_blocking=True)
    torch.cuda.synchronize()
    print(f"H2D 128 x 15 MiB {128 * c / (time.perf_counter() - t) / 1e9:.1f} GB/s")


bw()
