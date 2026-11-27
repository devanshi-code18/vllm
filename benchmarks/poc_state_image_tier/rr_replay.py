#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Multi-agent re-creation of capy-flash's agentic replay (bench/capy_flash/replay.py).

Each agent replays one of the 3 sessions in sessions.json (agent a -> session a % 3) with a
per-agent tag in its first user message, so agents share only the system prompt (8.6k tok).
Turn t of an agent = system + task + turns 0..t (fixed assistant text + tool output), streamed,
temperature 0, MAXNEW output tokens (ignore_eos so every turn decodes the same length).

Modes:
  rr   round-robin, one request at a time: a0 t0, a1 t0, ..., aN t0, a0 t1, ...  Between two turns
       of one agent the other N-1 agents run, so its cache must survive N-1 turns of other traffic.
  conc N agents at once, each sending its next turn as soon as the previous one finished.

Usage: rr_replay.py URL MODEL sessions.json out.jsonl --agents N [--mode rr|conc] [--maxnew 256]
       [--max-turns T] [--max-prompt-tokens P]
"""

import argparse
import json
import threading
import time
import urllib.request

TOK = None  # set by --completions: render the chat template client side


def one(url, model, msgs, maxnew, tag, logprobs=False):
    req_body = {
        "model": model,
        "max_tokens": maxnew,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if TOK is not None:
        # Raw completions: every token streams as it is sampled (no chat/tool
        # parser buffering), so the first chunk is the first token.
        req_body["prompt"] = TOK.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        req_body["skip_special_tokens"] = False
        if logprobs:
            req_body["logprobs"] = 1
        endpoint = "/v1/completions"
    else:
        req_body["messages"] = msgs
        if logprobs:
            req_body["logprobs"] = True
        endpoint = "/v1/chat/completions"
    body = json.dumps(req_body).encode()
    t0 = time.time()
    first = last = None
    usage = None
    text = []
    lps = []
    err = None
    try:
        req = urllib.request.Request(
            url + endpoint, body, {"Content-Type": "application/json", "X-Agent": tag}
        )
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                d = json.loads(data)
                if d.get("error"):
                    err = "stream error: " + json.dumps(d["error"])[:280]
                    break
                if d.get("usage"):
                    usage = d["usage"]
                for c in d.get("choices", []):
                    # vLLM sends the first chunk (role) with the first token, so any
                    # choice chunk marks first-token time, also when the turn opens
                    # with a tool call instead of text.
                    now = time.time()
                    first = first or now
                    clp = c.get("logprobs") or {}
                    for lp in clp.get("content") or []:
                        lps.append([lp.get("token"), lp.get("logprob")])
                    for t_, l_ in zip(
                        clp.get("tokens") or [], clp.get("token_logprobs") or []
                    ):
                        lps.append([t_, l_])
                    delta = c.get("delta", {})
                    if "text" in c and TOK is not None:
                        delta = {"content": c["text"]}
                    piece = (
                        delta.get("content")
                        or delta.get("reasoning_content")
                        or delta.get("reasoning")
                    )
                    if delta.get("tool_calls"):
                        piece = (piece or "") + json.dumps(delta["tool_calls"])
                    if piece:
                        last = now
                        text.append(piece)
    except Exception as e:  # noqa: BLE001
        err = repr(e)[:300]
    t1 = time.time()
    u = usage or {}
    ct = u.get("completion_tokens")
    return {
        "t_start": t0,
        "prompt_tokens": u.get("prompt_tokens"),
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": ct,
        "ttft_s": (first - t0) if first else None,
        "total_s": t1 - t0,
        "decode_tps": (ct - 1) / (last - first)
        if first and last and last > first and ct and ct > 1
        else None,
        "text": "".join(text),
        "logprobs": lps or None,
        "error": err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("sessions")
    ap.add_argument("out")
    ap.add_argument("--agents", type=int, default=3)
    ap.add_argument("--mode", choices=["rr", "conc"], default="rr")
    ap.add_argument("--maxnew", type=int, default=256)
    ap.add_argument("--max-turns", type=int, default=10**9)
    ap.add_argument("--first-agent", type=int, default=0)
    ap.add_argument("--logprobs", action="store_true")
    ap.add_argument(
        "--completions",
        default=None,
        help="HF tokenizer id: render chat template client side, use /v1/completions",
    )
    ap.add_argument(
        "--think",
        type=float,
        default=0.0,
        help="conc: mean tool time (s) between turns, exponential, seeded per agent",
    )
    args = ap.parse_args()

    if args.completions:
        global TOK
        from transformers import AutoTokenizer

        TOK = AutoTokenizer.from_pretrained(args.completions)
    S = json.load(open(args.sessions))
    sessions = S["sessions"]
    agents = []
    for a in range(args.first_agent, args.first_agent + args.agents):
        s = sessions[a % len(sessions)]
        msgs = [
            {"role": "system", "content": S["system"]},
            {"role": "user", "content": f"[agent {a}] " + s["task"]},
        ]
        agents.append(
            {
                "a": a,
                "si": a % len(sessions),
                "turns": s["turns"][: args.max_turns],
                "msgs": msgs,
            }
        )

    out = open(args.out, "a")
    lock = threading.Lock()

    def run_turn(ag, ti):
        t = ag["turns"][ti]
        ag["msgs"] = ag["msgs"] + [
            {"role": "assistant", "content": t["assistant"]},
            {"role": "user", "content": t["tool"]},
        ]
        rec = one(
            args.url,
            args.model,
            ag["msgs"],
            args.maxnew,
            f"a{ag['a']}t{ti}",
            args.logprobs,
        )
        rec.update(
            {
                "agent": ag["a"],
                "session": ag["si"],
                "turn": ti,
                "mode": args.mode,
                "n_agents": args.agents,
            }
        )
        with lock:
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(
                f"a{ag['a']:02d} s{ag['si']} t{ti:02d} prompt {rec['prompt_tokens']} cached {rec['cached_tokens']} "
                f"ttft {rec['ttft_s'] and round(rec['ttft_s'], 3)} total {round(rec['total_s'], 2)} {rec['error'] or ''}",
                flush=True,
            )

    t0 = time.time()
    if args.mode == "rr":
        max_t = max(len(ag["turns"]) for ag in agents)
        for ti in range(max_t):
            for ag in agents:
                if ti < len(ag["turns"]):
                    run_turn(ag, ti)
    else:
        import random

        def worker(ag):
            rng = random.Random(1000 + ag["a"])
            time.sleep(rng.uniform(0, args.think))  # stagger starts
            for ti in range(len(ag["turns"])):
                if ti and args.think:
                    time.sleep(rng.expovariate(1 / args.think))
                run_turn(ag, ti)

        th = [threading.Thread(target=worker, args=(ag,)) for ag in agents]
        for x in th:
            x.start()
        for x in th:
            x.join()
    print(f"WALL {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
