# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run one `ParityConfig` on a prompt suite, in a fresh process.

    python -m tests.correctness.parity_runner --config cfg.json \
        --suite suite.json --out run.json [--ref ref_run.json] [--gsm8k 200]

`run_cached` wraps this in a subprocess so every config gets a clean engine
(TP workers, CUDA graphs, env vars such as `VLLM_BATCH_INVARIANT`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

try:
    from tests.correctness.parity import (
        ParityConfig,
        PromptRecord,
        RunResult,
        seq_hash,
    )
except ImportError:  # run by path against another vLLM install
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parity import ParityConfig, PromptRecord, RunResult, seq_hash  # type: ignore

TOPK = 8

SHORT_PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number using memoization.",
    "List five countries in South America and their capitals.",
    (
        "Translate to French: The weather is nice today and I would like to go "
        "for a walk by the river."
    ),
    "Explain in two sentences why the sky is blue.",
    "Write a haiku about autumn leaves.",
    "Return a JSON object with keys name, age and city for a fictional person.",
    "What is 17 * 23? Show your work.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]

SUITES = {
    # name: (chat prompts, zero-shot gsm8k, long prompts, shots per long
    #        prompt, shared-prefix, copy groups x copies, max_tokens)
    "ci": (4, 8, 2, 25, 4, (1, 4), 64),
    "full": (8, 24, 6, 25, 8, (2, 4), 128),
    # For 4k-context models: long prompts stay under ~3.5k tokens.
    "full4k": (8, 24, 6, 16, 8, (2, 4), 128),
}


def _gsm8k_rows() -> tuple[list[dict], list[dict]]:
    try:
        from tests.evals.gsm8k.gsm8k_eval import load_gsm8k_data
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from tests.evals.gsm8k.gsm8k_eval import load_gsm8k_data
    return load_gsm8k_data()


def _qa(row: dict) -> str:
    return f"Question: {row['question']}\nAnswer: {row['answer']}\n\n"


def build_suite(tokenizer_name: str, size: str = "full") -> dict[str, Any]:
    """Fixed prompts, tokenized once so every config sees the same ids.

    Kinds: `chat` (chat template), `qa` (zero-shot GSM8K), `long` (16-25
    shots, longer than common sliding windows), `shared` (one 12-shot prefix +
    different questions: prefix-cache hits), `copy` (identical prompts in
    one batch: same-step prefix hits).
    """
    from transformers import AutoTokenizer

    n_chat, n_qa, n_long, shots, n_shared, (n_groups, n_copies), max_tokens = SUITES[
        size
    ]
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    train, test = _gsm8k_rows()
    items: list[tuple[str, str, str]] = []
    for i, p in enumerate(SHORT_PROMPTS[:n_chat]):
        msgs = [{"role": "user", "content": p}]
        try:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except Exception:
            text = p
        items.append((f"chat{i}", "chat", text))
    for i in range(n_qa):
        items.append((f"qa{i}", "qa", f"Question: {test[i]['question']}\nAnswer:"))
    for i in range(n_long):
        ctx = "".join(_qa(train[(i * 25 + j) % len(train)]) for j in range(shots))
        q = test[100 + i]["question"]
        items.append((f"long{i}", "long", ctx + f"Question: {q}\nAnswer:"))
    prefix = "".join(_qa(train[500 + j]) for j in range(12))
    for i in range(n_shared):
        q = test[200 + i]["question"]
        items.append((f"shared{i}", "shared", prefix + f"Question: {q}\nAnswer:"))
    for g in range(n_groups):
        ctx = "".join(_qa(train[800 + g * 8 + j]) for j in range(8))
        text = ctx + f"Question: {test[300 + g]['question']}\nAnswer:"
        for c in range(n_copies):
            items.append((f"copy{g}.{c}", "copy", text))
    prompts = []
    for pid, kind, text in items:
        ids = tok(text, add_special_tokens=kind != "chat")["input_ids"]
        prompts.append(
            dict(prompt_id=pid, kind=kind, token_ids=ids, max_tokens=max_tokens)
        )
    return dict(tokenizer=tokenizer_name, size=size, prompts=prompts)


def _topk(d: dict | None, k: int) -> tuple[list[int], list[float]]:
    if not d:
        return [], []
    items = sorted(
        ((lp.rank, tid, lp.logprob) for tid, lp in d.items() if lp.rank <= k),
        key=lambda x: x[0],
    )
    return [t for _, t, _ in items], [float(x) for _, _, x in items]


def _decode_record(p: dict, out: Any, k: int) -> PromptRecord:
    c = out.outputs[0]
    ids = list(c.token_ids)
    lp, tk_ids, tk_lp = [], [], []
    for t, d in zip(ids, c.logprobs or []):
        lp.append(float(d[t].logprob))
        i, v = _topk(d, k)
        tk_ids.append(i)
        tk_lp.append(v)
    return PromptRecord(
        prompt_id=p["prompt_id"],
        kind=p["kind"],
        gen_ids=ids,
        gen_lp=lp,
        gen_topk_ids=tk_ids,
        gen_topk_lp=tk_lp,
    )


def run_config(
    cfg: ParityConfig,
    suite: dict[str, Any],
    ref: RunResult | None = None,
    gsm8k_n: int = 0,
    topk: int = TOPK,
) -> RunResult:
    """Run in this process. Set `cfg.run_env()` before vLLM is imported."""
    import torch

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.platforms import current_platform

    t0 = time.time()
    llm = LLM(**cfg.engine_kwargs())
    t_init = time.time() - t0
    ps = suite["prompts"]
    inputs = [TokensPrompt(prompt_token_ids=p["token_ids"]) for p in ps]
    sps = [
        SamplingParams(temperature=0.0, max_tokens=p["max_tokens"], logprobs=topk)
        for p in ps
    ]

    def decode_pass() -> list[PromptRecord]:
        outs = llm.generate(inputs, sps, use_tqdm=False)
        return [_decode_record(p, o, topk) for p, o in zip(ps, outs)]

    t1 = time.time()
    recs = decode_pass()
    recs2 = decode_pass() if cfg.enable_prefix_caching else None
    t_gen = time.time() - t1

    # Teacher-forced: score prompt + reference continuation. Requests with
    # prompt_logprobs never read the prefix cache, so this is the prefill
    # path of the config at its batch size and chunking.
    src = ref.prompts if ref is not None else recs
    gen_by_id = {r.prompt_id: r.gen_ids for r in src}
    seqs = [p["token_ids"] + gen_by_id[p["prompt_id"]] for p in ps]
    t2 = time.time()
    tf_outs = llm.generate(
        [TokensPrompt(prompt_token_ids=s) for s in seqs],
        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=topk),
        use_tqdm=False,
    )
    for r, s, o in zip(recs, seqs, tf_outs):
        r.tf_src = seq_hash(s)
        pl = o.prompt_logprobs or []
        for j in range(1, len(s)):
            d = pl[j]
            r.tf_lp.append(float(d[s[j]].logprob))
            ids, lps = _topk(d, topk)
            r.tf_topk_ids.append(ids)
            r.tf_margin.append(lps[0] - lps[1] if len(lps) > 1 else 0.0)
    t_tf = time.time() - t2

    preds = labels = None
    t_gsm = 0.0
    if gsm8k_n:
        try:
            from tests.evals.gsm8k.gsm8k_eval import (
                _build_gsm8k_prompts,
                get_answer_value,
            )
        except ImportError:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
            from tests.evals.gsm8k.gsm8k_eval import (
                _build_gsm8k_prompts,
                get_answer_value,
            )
        prompts, labels = _build_gsm8k_prompts(num_questions=gsm8k_n, num_shots=5)
        t3 = time.time()
        outs = llm.generate(
            prompts,
            SamplingParams(
                temperature=0.0,
                max_tokens=256,
                stop=["Question", "</s>", "<|im_end|>", "\n\n\n"],
            ),
            use_tqdm=False,
        )
        preds = [get_answer_value(o.outputs[0].text) for o in outs]
        t_gsm = time.time() - t3

    meta = dict(
        vllm_version=vllm.__version__,
        torch=torch.__version__,
        gpu=current_platform.get_device_name(),
        n_gpus_visible=torch.accelerator.device_count(),
        env={k: v for k, v in os.environ.items() if k.startswith("VLLM_")},
        seconds=dict(init=t_init, gen=t_gen, tf=t_tf, gsm8k=t_gsm),
        suite=suite.get("size"),
    )
    return RunResult(
        config=cfg.to_json(),
        prompts=recs,
        prompts_pass2=recs2,
        gsm8k_preds=preds,
        gsm8k_labels=labels,
        meta=meta,
    )


def result_path(cfg: ParityConfig, out_dir: str) -> str:
    safe = cfg.name.replace("/", "__")
    return os.path.join(os.path.abspath(out_dir), f"{safe}.{cfg.key()}.json")


def run_cached(
    cfg: ParityConfig,
    suite_path: str,
    out_dir: str,
    ref_path: str | None = None,
    gsm8k_n: int = 0,
    gpus: str | None = None,
    python: str | None = None,
    timeout: float = 3600,
) -> tuple[str, RunResult | None]:
    """Run `cfg` in a subprocess unless its result file exists.

    Returns the result path and the result (None if the run failed; the
    log next to the result path says why).
    """
    out = result_path(cfg, out_dir)
    if os.path.exists(out):
        return out, RunResult.load(out)
    os.makedirs(out_dir, exist_ok=True)
    cfg_path = out.replace(".json", ".cfg.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg.to_json(), f)
    env = dict(os.environ)
    env.update(cfg.run_env())
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    cmd = [
        python or sys.executable,
        os.path.abspath(__file__),
        "--config",
        cfg_path,
        "--suite",
        suite_path,
        "--out",
        out,
        "--gsm8k",
        str(gsm8k_n),
    ]
    if ref_path:
        cmd += ["--ref", ref_path]
    log = out.replace(".json", ".log")
    with open(log, "w") as lf:
        try:
            rc = subprocess.run(
                cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout
            ).returncode
        except subprocess.TimeoutExpired:
            rc = -1
    if rc != 0 or not os.path.exists(out):
        return out, None
    return out, RunResult.load(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--gsm8k", type=int, default=0)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = ParityConfig.from_json(json.load(f))
    os.environ.update(cfg.run_env())
    with open(args.suite) as f:
        suite = json.load(f)
    ref = RunResult.load(args.ref) if args.ref else None
    res = run_config(cfg, suite, ref=ref, gsm8k_n=args.gsm8k)
    res.save(args.out)


if __name__ == "__main__":
    main()
