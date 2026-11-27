# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity matrices: which configs to run, how to judge them, and the report.

    python tests/correctness/parity_matrix.py --matrix full --out DIR \
        --gpus 0,1 [--only REGEX] [--gsm8k 200]

Every config in a matrix scores the same reference sequences (the global
reference's prompt + greedy continuation), so any two runs that share a
tokenizer can be compared teacher-forced. Roles:

- `ref`: the family reference (bf16 / TP1 / FLASH_ATTN / batch 1 for the
  bf16 family; same serving setup with the family's weights otherwise).
- `noise`: same numerics as `ref`, only batch size, chunk size or run
  differs. Pairs among `ref` + `noise` that agree with the family consensus
  (see `NoiseModel.fit_robust`) calibrate the family's thresholds.
- `test`: judged against the family reference with the family noise model.
- `rerun`: a test or noise config in a fresh process; its greedy tokens must
  reproduce at least as well as the same batch size without prefix caching.
- `bi_ref` / `bi_test`: `VLLM_BATCH_INVARIANT=1`; must be bitwise equal.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import regex as re

try:
    from tests.correctness.parity import (
        NoiseModel,
        ParityConfig,
        RunResult,
        compare,
        decode_vs_prefill,
        exact_diff,
    )
    from tests.correctness.parity_runner import (
        SUITES,
        build_suite,
        result_path,
        run_cached,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parity import (  # type: ignore
        NoiseModel,
        ParityConfig,
        RunResult,
        compare,
        decode_vs_prefill,
        exact_diff,
    )
    from parity_runner import (  # type: ignore
        SUITES,
        build_suite,
        result_path,
        run_cached,
    )

# Absolute quality budget for a quantized family's reference against the bf16
# reference. Not noise-calibrated: quantization is supposed to change bits.
QUALITY_BUDGET = {"tf_top1_disagree": 0.25, "gsm8k_drop": 0.08, "tf_dnll": 0.30}


@dataclass
class Entry:
    cfg: ParityConfig
    role: str  # ref | noise | test | bi_ref | bi_test
    axis: str = ""  # what differs from the reference, for the report
    bi_ref: str = ""  # name of the bi_ref this bi_test compares against
    rerun_of: str = ""  # role "rerun": a fresh process of this config


def _family(
    fam: str,
    model: str,
    *,
    quantization: str | None = None,
    backend: str = "FLASH_ATTN",
    noise: tuple[str, ...] = ("run2", "bs8", "bs64", "mnbt256", "mnbt1024"),
    tests: tuple[str, ...] = (),
    reruns: tuple[str, ...] = (),
    max_model_len: int = 8192,
    extra: tuple = (),
    env: tuple = (),
) -> list[Entry]:
    base = ParityConfig(
        name=f"{fam}/ref",
        model=model,
        quantization=quantization,
        attention_backend=backend,
        family=fam,
        max_model_len=max_model_len,
        extra=extra,
        env=env,
    )
    out = [Entry(base, "ref", "reference")]
    variants: dict[str, dict[str, Any]] = {
        "run2": dict(env=env + (("PARITY_RUN", "2"),)),
        "bs8": dict(max_num_seqs=8),
        "bs64": dict(max_num_seqs=64),
        "mnbt256": dict(max_num_seqs=64, max_num_batched_tokens=256),
        "mnbt1024": dict(max_num_seqs=64, max_num_batched_tokens=1024),
        "pc_bs64": dict(max_num_seqs=64, enable_prefix_caching=True),
        "pc_bs1": dict(enable_prefix_caching=True),
        "FLASHINFER": dict(attention_backend="FLASHINFER"),
        "TRITON_ATTN": dict(attention_backend="TRITON_ATTN"),
        "FLEX_ATTENTION": dict(attention_backend="FLEX_ATTENTION"),
        "FA2": dict(flash_attn_version=2),
        "FLASHINFER_bs64": dict(attention_backend="FLASHINFER", max_num_seqs=64),
        "TRITON_ATTN_bs64": dict(attention_backend="TRITON_ATTN", max_num_seqs=64),
        "FA2_mnbt256": dict(
            flash_attn_version=2, max_num_seqs=64, max_num_batched_tokens=256
        ),
        "tp2": dict(tensor_parallel_size=2),
        "tp2_bs64": dict(tensor_parallel_size=2, max_num_seqs=64),
        "ngram": dict(
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 4,
                "prompt_lookup_max": 4,
            }
        ),
        "ngram_bs64": dict(
            max_num_seqs=64,
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 4,
                "prompt_lookup_max": 4,
            },
        ),
        "eagle3": dict(
            speculative_config={
                "method": "eagle3",
                "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
                "num_speculative_tokens": 3,
            }
        ),
    }
    for v in noise:
        out.append(Entry(base.replace(name=f"{fam}/{v}", **variants[v]), "noise", v))
    for v in tests:
        out.append(Entry(base.replace(name=f"{fam}/{v}", **variants[v]), "test", v))
    # Run-to-run determinism: the same config in a fresh process.
    for v in reruns:
        kw = dict(variants[v])
        kw["env"] = kw.get("env", env) + (("PARITY_RUN", "2"),)
        c = base.replace(name=f"{fam}/{v}_rerun", **kw)
        out.append(Entry(c, "rerun", f"{v} rerun", rerun_of=f"{fam}/{v}"))
    return out


def _bi_block(
    fam: str, model: str, backend: str, tp2: bool, extra_tests: tuple = ()
) -> list[Entry]:
    ref = ParityConfig(
        name=f"{fam}/bi_{backend}/bs1",
        model=model,
        attention_backend=backend,
        batch_invariant=True,
        family=fam,
    )
    tests = {
        "bs64": dict(max_num_seqs=64),
        "mnbt256": dict(max_num_seqs=64, max_num_batched_tokens=256),
        "pc_bs64": dict(max_num_seqs=64, enable_prefix_caching=True),
        "ngram": dict(
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 4,
                "prompt_lookup_max": 4,
            }
        ),
    }
    out = [Entry(ref, "bi_ref", "bs1")]
    for v in ("bs64", "mnbt256", "pc_bs64") + extra_tests:
        c = ref.replace(name=f"{fam}/bi_{backend}/{v}", **tests[v])
        out.append(Entry(c, "bi_test", v, bi_ref=ref.name))
    if tp2:
        r2 = ref.replace(name=f"{fam}/bi_{backend}/tp2_bs1", tensor_parallel_size=2)
        out.append(Entry(r2, "bi_ref", "tp2_bs1"))
        for v in ("bs64", "pc_bs64"):
            c = r2.replace(name=f"{fam}/bi_{backend}/tp2_{v}", **tests[v])
            out.append(Entry(c, "bi_test", f"tp2_{v}", bi_ref=r2.name))
        # Across TP sizes (same run as tp2_bs1): reported, not required exact.
        out.append(Entry(r2, "bi_test", "tp2 vs tp1", bi_ref=ref.name))
    return out


QWEN = "Qwen/Qwen3-8B"


def matrix(name: str, n_gpus: int, sm: int) -> list[Entry]:
    """Named matrices. `sm` is the compute capability (90 = H100, 100 = B200)."""
    tp2 = n_gpus >= 2
    if name == "ci":
        m = "Qwen/Qwen3-0.6B"
        e = _family(
            "ci_bf16",
            m,
            noise=("run2", "bs8", "bs64", "mnbt256"),
            tests=(
                "pc_bs64",
                "FLASHINFER",
                "TRITON_ATTN",
                "FLEX_ATTENTION",
                "ngram",
            )
            + (("tp2",) if tp2 else ()),
            reruns=("pc_bs64",),
        )
        e += _family("ci_fp8", m, quantization="fp8", noise=("bs64",), tests=())
        for b in ("FLASH_ATTN", "TRITON_ATTN"):
            e += _bi_block("ci_bf16", m, b, tp2=tp2)
        return e
    if name == "full":
        backends = ("FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION")
        if sm == 90:
            backends += ("FA2",)
        e = _family(
            "bf16",
            QWEN,
            tests=backends
            + (
                "FLASHINFER_bs64",
                "TRITON_ATTN_bs64",
                "pc_bs1",
                "pc_bs64",
                "ngram",
                "ngram_bs64",
                "eagle3",
            )
            + (("tp2", "tp2_bs64") if tp2 else ()),
            reruns=("bs64", "pc_bs64"),
        )
        qt = ("tp2", "pc_bs64") if tp2 else ("pc_bs64",)
        e += _family("fp8_online", QWEN, quantization="fp8", noise=("bs64",), tests=qt)
        e += _family("fp8_offline", "Qwen/Qwen3-8B-FP8", noise=("bs64",), tests=qt)
        e += _family("awq_int4", "Qwen/Qwen3-8B-AWQ", noise=("bs64",), tests=qt)
        if sm == 90:
            e += _family(
                "w4a16_ct",
                "RedHatAI/Qwen3-8B-quantized.w4a16",
                noise=("bs64",),
                tests=qt,
            )
        if sm >= 100:
            e += _family(
                "nvfp4", "nvidia/Qwen3-8B-NVFP4", noise=("bs64",), tests=("pc_bs64",)
            )
        for b in ("FLASH_ATTN", "TRITON_ATTN", "FLEX_ATTENTION"):
            e += _bi_block("bf16", QWEN, b, tp2=tp2, extra_tests=("ngram",))
        return e
    if name == "swa":
        # Sliding-window models: gemma-3-1b (head dim 256, window 512) and
        # Phi-3-mini-4k (head dim 96, window 2047).
        sw_tests = ("FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION", "pc_bs64") + (
            ("FA2", "FA2_mnbt256") if sm == 90 else ()
        )
        phi = _family(
            "phi3_mini",
            "microsoft/Phi-3-mini-4k-instruct",
            tests=sw_tests,
            max_model_len=4096,
        )
        return phi + _family(
            "gemma3_1b",
            "unsloth/gemma-3-1b-it",
            tests=(
                "FLASHINFER",
                "TRITON_ATTN",
                "FLEX_ATTENTION",
                "pc_bs64",
            )
            + (("FA2", "FA2_mnbt256") if sm == 90 else ()),
        )
    if name == "torchao":
        # A torchao AWQ checkpoint (tests/quantization/test_torchao.py's
        # `_save_awq_checkpoint` recipe) against its bf16 base model.
        ckpt = os.environ["VLLM_PARITY_TORCHAO_CKPT"]
        base = os.getenv("VLLM_PARITY_TORCHAO_BASE", "Qwen/Qwen3-1.7B")
        return _family("bf16_base", base, noise=("bs64",), tests=()) + _family(
            "torchao_awq",
            ckpt,
            noise=("bs64",),
            tests=(),
            extra=(("tokenizer", base),),
        )
    if name == "inkling":
        # Hybrid model with a short-conv state cached by prefix caching.
        return _family(
            "inkling2_mini",
            "thinkingmachines/Inkling-2-Mini",
            backend=None,  # type: ignore[arg-type]
            noise=("run2", "bs8", "bs64"),
            tests=("pc_bs64",),
            reruns=("bs64", "pc_bs64"),
        )
    raise ValueError(name)


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def run_matrix(
    entries: list[Entry],
    out_dir: str,
    gpus: list[str],
    gsm8k_n: int,
    global_ref: str | None = None,
    python: str | None = None,
    suite_size: str = "full",
) -> dict[str, RunResult]:
    """Run every entry (cached on disk). The global reference runs first and
    its sequences are scored by every other run with the same tokenizer."""
    suites: dict[str, str] = {}

    def suite_for(model: str, max_model_len: int) -> str:
        size = suite_size
        if max_model_len <= 4096 and f"{size}4k" in SUITES:
            size = f"{size}4k"
        if model not in suites:
            p = os.path.join(
                out_dir, f"suite.{size}." + model.replace("/", "_") + ".json"
            )
            if not os.path.exists(p):
                os.makedirs(out_dir, exist_ok=True)
                with open(p, "w") as f:
                    json.dump(build_suite(model, size), f)
            suites[model] = p
        return suites[model]

    tok_of: dict[str, str] = {}
    refs = [e for e in entries if e.role == "ref"]
    # The first family reference of each tokenizer is its global reference.
    groups: dict[str, str] = {}
    results: dict[str, RunResult] = {}
    paths: dict[str, str] = {}
    lock = threading.Lock()

    def tok_key(e: Entry) -> str:
        # Checkpoints of one base model share its tokenizer and reference.
        tok = dict(e.cfg.extra).get("tokenizer")
        if tok:
            return tok
        return QWEN if "Qwen3-8B" in e.cfg.model else e.cfg.model

    for e in entries:
        tok_of[e.cfg.name] = tok_key(e)
    for e in refs:
        if tok_of[e.cfg.name] not in groups:
            groups[tok_of[e.cfg.name]] = e.cfg.name
            paths[e.cfg.name] = result_path(e.cfg, out_dir)

    def refseq_path(tk: str) -> str:
        return os.path.join(out_dir, "refseq." + tk.replace("/", "_") + ".json")

    def run(e: Entry, gpu: str) -> None:
        # Every run scores frozen reference sequences: the global reference's
        # first continuation. Freezing matters because default mode is not
        # deterministic run to run, so a re-run reference can continue
        # differently.
        tk = tok_of[e.cfg.name]
        ref_path = global_ref
        if ref_path is None and os.path.exists(refseq_path(tk)):
            ref_path = refseq_path(tk)
        path, res = run_cached(
            e.cfg,
            suite_for(tk, e.cfg.max_model_len),
            out_dir,
            ref_path=ref_path,
            gsm8k_n=gsm8k_n if e.role in ("ref", "noise", "test") else 0,
            gpus=gpu,
            python=python,
        )
        with lock:
            paths[e.cfg.name] = path
            if res is not None:
                results[e.cfg.name] = res
                if groups.get(tk) == e.cfg.name and not os.path.exists(refseq_path(tk)):
                    shutil.copyfile(path, refseq_path(tk))
            print(
                f"[parity] {'ok ' if res else 'ERR'} {e.cfg.name} ({path})",
                flush=True,
            )

    # Phase 1: global references.
    first = [e for e in entries if e.cfg.name in groups.values()]
    _pool(first, gpus, run)
    rest = [e for e in entries if e.cfg.name not in groups.values()]
    _pool([e for e in rest if e.cfg.tensor_parallel_size == 1], gpus, run)
    multi = [e for e in rest if e.cfg.tensor_parallel_size > 1]
    for e in multi:
        if e.cfg.tensor_parallel_size <= len(gpus):
            run(e, ",".join(gpus[: e.cfg.tensor_parallel_size]))
    return results


def _pool(entries: list[Entry], gpus: list[str], fn) -> None:
    free = list(gpus)
    cv = threading.Condition()

    def task(e: Entry) -> None:
        with cv:
            while not free:
                cv.wait()
            g = free.pop(0)
        try:
            fn(e, g)
        finally:
            with cv:
                free.append(g)
                cv.notify()

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        list(ex.map(task, entries))


# --------------------------------------------------------------------------
# Judging + report
# --------------------------------------------------------------------------

KEY_STATS = (
    "tf_mean",
    "tf_p99",
    "tf_max",
    "dec_p99",
    "top1_disagree",
    "dec_match_rate",
    "flip_margin",
    "gsm8k",
)


def evaluate(
    entries: list[Entry],
    results: dict[str, RunResult],
    safety: float = 3.0,
    bf16_family: str | None = None,
    refseq: RunResult | None = None,
) -> dict[str, Any]:
    by_fam: dict[str, list[Entry]] = {}
    for e in entries:
        by_fam.setdefault(e.cfg.family, []).append(e)
    report: dict[str, Any] = dict(families={}, rows=[], bi=[], quality=[])
    noise_models: dict[str, NoiseModel] = {}
    for fam, es in by_fam.items():
        ref = next((e for e in es if e.role == "ref"), None)
        if ref is None or ref.cfg.name not in results:
            continue
        cal = [ref] + [e for e in es if e.role == "noise"]
        runs = {e.axis: results[e.cfg.name] for e in cal if e.cfg.name in results}
        # Other attention backends at batch 1: independent implementations of
        # the same math, used to judge whether calibration runs are sane.
        peers = {
            e.axis: results[e.cfg.name]
            for e in es
            if e.role == "test"
            and e.cfg.name in results
            and e.cfg.max_num_seqs == 1
            and e.cfg.tensor_parallel_size == 1
            and e.cfg.max_num_batched_tokens is None
            and not e.cfg.speculative_config
            and not e.cfg.enable_prefix_caching
        }
        info: dict[str, Any] = dict(reference=ref.cfg.name)
        if len(runs) >= 2:
            nm, rejected, pairs = NoiseModel.fit_robust(
                runs, safety=safety, peers=peers
            )
            noise_models[fam] = nm
            info["rejected_from_calibration"] = rejected
            info["consensus"] = nm.consensus
            info["reference_outlier"] = nm.reference_outlier
            info["noise_pairs"] = {
                k: {s: v[s] for s in v if s != "per_prompt"} for k, v in pairs.items()
            }
            for e in es:
                if e.role == "noise" and e.axis in rejected:
                    e.role = "test"  # judged (and flagged) like any test config
        report["families"][fam] = info
    fallback = noise_models.get(bf16_family or "")
    for fam, es in by_fam.items():
        nm = noise_models.get(fam) or fallback
        method = nm.describe() if nm is not None else ""
        if nm is not None and fallback is not None and nm is not fallback:
            # A quantized family is calibrated on fewer runs; its noise is at
            # least the bf16 family's (same model, same serving stack).
            nm = replace(
                nm,
                thresholds={
                    k: max(v, fallback.thresholds.get(k, 0.0))
                    for k, v in nm.thresholds.items()
                },
            )
            method += f"; each threshold is at least the {bf16_family} one"
        if nm is not None:
            report["families"].setdefault(fam, {})["noise_model"] = dict(
                nm.to_json(), method=method
            )
        ref = next((e for e in es if e.role == "ref"), None)
        if ref is None or ref.cfg.name not in results:
            continue
        R = results[ref.cfg.name]
        anchor = ref.cfg.name
        if nm is not None and nm.reference_outlier and nm.consensus:
            # The reference is the odd one out: judge against the consensus.
            peer = nm.consensus[0].removeprefix("peer:")
            hit = next(
                (x for x in es if x.axis == peer and x.cfg.name in results), None
            )
            if hit is not None:
                R, anchor = results[hit.cfg.name], hit.cfg.name
        for e in es:
            if e.role not in ("test", "noise") or e.cfg.name not in results:
                if e.role in ("test", "noise"):
                    report["rows"].append(
                        dict(name=e.cfg.name, family=fam, axis=e.axis, error=True)
                    )
                continue
            X = results[e.cfg.name]
            passes = [("", False)]
            if e.cfg.enable_prefix_caching and X.prompts_pass2 is not None:
                passes.append((" (warm)", True))
            for suffix, p2 in passes:
                st = compare(R, X, use_pass2=p2)
                verdict = nm.judge(st) if nm and e.role == "test" else None
                report["rows"].append(
                    dict(
                        name=e.cfg.name + suffix,
                        family=fam,
                        axis=e.axis + suffix,
                        role=e.role,
                        stats={k: v for k, v in st.items() if k != "per_prompt"},
                        per_prompt=st["per_prompt"],
                        verdict=verdict,
                        judged_against=anchor,
                    )
                )
    # Quality of each family reference against the bf16 reference.
    if bf16_family and bf16_family in report["families"]:
        g = results.get(report["families"][bf16_family]["reference"])
        for fam, info in report["families"].items():
            if fam == bf16_family or "reference" not in info:
                continue
            X = results.get(info["reference"])
            if g is None or X is None:
                continue
            if X.prompts[0].tf_lp and len(X.prompts[0].tf_lp) == len(
                g.prompts[0].tf_lp
            ):
                st = compare(g, X)
                viol = {
                    k: st[k] / v for k, v in QUALITY_BUDGET.items() if st.get(k, 0) > v
                }
                report["quality"].append(
                    dict(
                        family=fam,
                        stats={k: v for k, v in st.items() if k != "per_prompt"},
                        over_budget=viol,
                    )
                )
    # Run-to-run determinism of a config, next to the reference's own rerun.
    report["determinism"] = []
    for e in entries:
        if e.role != "rerun":
            continue
        A, B = results.get(e.rerun_of), results.get(e.cfg.name)
        if A is None or B is None:
            report["determinism"].append(dict(name=e.rerun_of, error=True))
            continue
        d = exact_diff(A, B)
        if A.prompts_pass2 is not None and B.prompts_pass2 is not None:
            d2 = exact_diff(A, B, use_pass2=True, ref_pass2=True)
            d["warm_tokens_differ"] = d2["tokens_differ"]
            d["warm_logprobs_differ"] = d2["decode_logprobs_differ"]
        d["n_tokens_differ"] = max(d["tokens_differ"], d.get("warm_tokens_differ", 0))
        fam = e.cfg.family
        ref_name = report["families"].get(fam, {}).get("reference")
        run2 = next(
            (x for x in entries if x.cfg.family == fam and x.axis == "run2"), None
        )
        if ref_name and run2 and ref_name in results and run2.cfg.name in results:
            d["reference_deterministic"] = exact_diff(
                results[ref_name], results[run2.cfg.name]
            )["bitwise_equal"]
        report["determinism"].append(dict(name=e.rerun_of, **d))
    # A config's greedy tokens must reproduce at least as well as the same
    # batch size with prefix caching off (the control), which must reproduce
    # when the batch-1 reference does.
    by_name = {d["name"]: d for d in report["determinism"] if not d.get("error")}
    for e in entries:
        d = by_name.get(e.rerun_of)
        if e.role != "rerun" or d is None:
            continue
        base = next(x for x in entries if x.cfg.name == e.rerun_of).cfg
        ctrl = next(
            (
                by_name.get(x.rerun_of)
                for x in entries
                if x.role == "rerun"
                and x.cfg.family == base.family
                and x.rerun_of != e.rerun_of
                and not x.cfg.enable_prefix_caching
                and x.cfg.max_num_seqs == base.max_num_seqs
            ),
            None,
        )
        if ctrl is not None and base.enable_prefix_caching:
            d["control"] = ctrl["name"]
            d["reproducible"] = d["n_tokens_differ"] <= ctrl["n_tokens_differ"]
        else:
            d["reproducible"] = d["n_tokens_differ"] == 0 or not d.get(
                "reference_deterministic"
            )
    # Batch-invariant blocks: bitwise, and decode == prefill within each run.
    gref = refseq
    if gref is None and bf16_family and bf16_family in report["families"]:
        gref = results.get(report["families"][bf16_family]["reference"])
    report["bi_decode_vs_prefill"] = {}
    for e in entries:
        if e.role in ("bi_ref", "bi_test") and gref and e.cfg.name in results:
            report["bi_decode_vs_prefill"][e.cfg.name] = decode_vs_prefill(
                results[e.cfg.name], gref
            )
    for e in entries:
        if e.role != "bi_test":
            continue
        R = results.get(e.bi_ref)
        X = results.get(e.cfg.name)
        if R is None or X is None:
            report["bi"].append(dict(name=e.cfg.name, axis=e.axis, error=True))
            continue
        name = e.cfg.name + (" vs tp1" if "vs" in e.axis else "")
        report["bi"].append(dict(name=name, axis=e.axis, **exact_diff(R, X)))
        if X.prompts_pass2 is not None:
            report["bi"].append(
                dict(
                    name=e.cfg.name + " (warm)",
                    axis=e.axis + " (warm)",
                    **exact_diff(R, X, use_pass2=True),
                )
            )
    return report


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def render_markdown(report: dict[str, Any], header: str = "") -> str:
    lines = [header] if header else []
    for fam, info in report["families"].items():
        nm = info.get("noise_model")
        if not nm:
            continue
        lines.append(f"\n### {fam}: noise model\n")
        lines.append(nm["method"] + "\n")
        lines.append("| stat | noise max | threshold |")
        lines.append("|---|---|---|")
        for s, t in nm["thresholds"].items():
            lines.append(f"| {s} | {_fmt(nm['noise_max'].get(s))} | {_fmt(t)} |")
        if info.get("reference_outlier"):
            lines.append(
                f"\n**REFERENCE OUTLIER**: {info['reference']} disagrees with the "
                f"consensus {info['consensus']}"
            )
    lines.append("\n### parity matrix\n")
    lines.append("| config | role | " + " | ".join(KEY_STATS) + " | verdict |")
    lines.append("|---|---|" + "---|" * len(KEY_STATS) + "---|")
    for r in report["rows"]:
        if r.get("error"):
            lines.append(f"| {r['name']} | error |" + " |" * (len(KEY_STATS) + 1))
            continue
        st = r["stats"]
        v = r["verdict"]
        vs = (
            "-"
            if v is None
            else (
                "FLAG " + ", ".join(f"{k} {x:.1f}x" for k, x in v["violations"].items())
                if v["flagged"]
                else "pass"
            )
        )
        lines.append(
            f"| {r['name']} | {r['role']} | "
            + " | ".join(_fmt(st.get(k)) for k in KEY_STATS)
            + f" | {vs} |"
        )
    if report["quality"]:
        lines.append("\n### quantized references vs bf16 reference\n")
        lines.append(
            "| family | tf_top1_disagree | tf_dnll | gsm8k ref / q | over budget |"
        )
        lines.append("|---|---|---|---|---|")
        for q in report["quality"]:
            st = q["stats"]
            lines.append(
                f"| {q['family']} | {_fmt(st['tf_top1_disagree'])} | "
                f"{_fmt(st['tf_dnll'])} | {_fmt(st.get('gsm8k_ref'))} / "
                f"{_fmt(st.get('gsm8k'))} | {q['over_budget'] or 'no'} |"
            )
    if report.get("determinism"):
        lines.append("\n### run-to-run determinism (same config, fresh process)\n")
        lines.append(
            "| config | tokens differ (cold / warm) | decode logprobs differ "
            "(cold / warm) | prefill differ | reference deterministic | verdict |"
        )
        lines.append("|---|---|---|---|---|---|")
        for d in report["determinism"]:
            if d.get("error"):
                lines.append(f"| {d['name']} | error | | | | |")
                continue
            n = d["n_prompts"]
            lines.append(
                f"| {d['name']} | {d['tokens_differ']} / "
                f"{d.get('warm_tokens_differ', '-')} of {n} | "
                f"{d['decode_logprobs_differ']} / "
                f"{d.get('warm_logprobs_differ', '-')} of {n} | "
                f"{d['tf_logprobs_differ']} of {n} | "
                f"{d.get('reference_deterministic', '-')} | "
                f"{'pass' if d.get('reproducible') else 'FLAG'} |"
            )
    if report.get("bi_decode_vs_prefill"):
        lines.append("\n### VLLM_BATCH_INVARIANT=1: decode vs prefill logprobs\n")
        lines.append("| run | tokens compared | differ | max abs |")
        lines.append("|---|---|---|---|")
        for n, d in report["bi_decode_vs_prefill"].items():
            lines.append(
                f"| {n} | {d['n_compared']} | {d['n_differ']} | {_fmt(d['max_abs'])} |"
            )
    if report["bi"]:
        lines.append("\n### VLLM_BATCH_INVARIANT=1 (bitwise expected)\n")
        lines.append("| config | tokens differ | logprobs differ | bitwise |")
        lines.append("|---|---|---|---|")
        for b in report["bi"]:
            if b.get("error"):
                lines.append(f"| {b['name']} | error | | |")
                continue
            lines.append(
                f"| {b['name']} | {b['tokens_differ']}/{b['n_prompts']} | "
                f"{max(b['decode_logprobs_differ'], b['tf_logprobs_differ'])}"
                f"/{b['n_prompts']} | {'yes' if b['bitwise_equal'] else 'NO'} |"
            )
    return "\n".join(lines) + "\n"


def explain(
    ref: RunResult, other: RunResult, suite: dict, n: int = 5, use_pass2: bool = False
) -> str:
    """Hand-verification view: the prompts that drive a verdict, with the
    reference vs config text at the greedy divergence and the teacher-forced
    positions with the largest |dlogprob|."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(suite["tokenizer"])
    st = compare(ref, other, use_pass2=use_pass2)
    plen = {p["prompt_id"]: len(p["token_ids"]) for p in suite["prompts"]}
    recs = other.prompts_pass2 if use_pass2 else other.prompts
    assert recs is not None
    oth = {p.prompt_id: p for p in recs}
    out = []
    rows = sorted(
        st["per_prompt"],
        key=lambda r: (-(r["flip_margin"] or 0), -r["tf_max"]),
    )[:n]
    for r in rows:
        rp = next(p for p in ref.prompts if p.prompt_id == r["prompt_id"])
        op = oth[r["prompt_id"]]
        div = r["first_div"]
        out.append(
            f"## {r['prompt_id']} ({r['kind']}, prompt {plen[r['prompt_id']]} tok): "
            f"first_div={div} margin={_fmt(r['flip_margin'])} "
            f"tf_max={_fmt(r['tf_max'])} at tf pos {r['tf_argmax']}"
        )
        if div is not None:
            lo = max(0, div - 12)
            out.append("  ref: " + repr(tok.decode(rp.gen_ids[lo : div + 12])))
            out.append("  cfg: " + repr(tok.decode(op.gen_ids[lo : div + 12])))
        d = [abs(a - b) for a, b in zip(rp.tf_lp, op.tf_lp)]
        worst = sorted(range(len(d)), key=lambda i: -d[i])[:5]
        out.append(
            "  worst tf positions (pos: |d|, ref top1 margin): "
            + ", ".join(
                f"{i + 1}: {d[i]:.3f} ({rp.tf_margin[i]:.2f})"
                if rp.tf_margin
                else f"{i + 1}: {d[i]:.3f}"
                for i in worst
            )
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--gsm8k", type=int, default=200)
    ap.add_argument("--only", default=None, help="regex on config names")
    ap.add_argument("--global-ref", default=None, help="score this run's sequences")
    ap.add_argument("--bf16-family", default="bf16")
    ap.add_argument("--python", default=None, help="interpreter for runs")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--safety", type=float, default=3.0)
    ap.add_argument("--suite", default="full", choices=("ci", "full"))
    ap.add_argument("--explain", default=None, help="config name to explain")
    ap.add_argument("--explain-ref", default=None, help="compare against this")
    args = ap.parse_args()
    from vllm.platforms import current_platform

    gpus = args.gpus.split(",")
    cap = current_platform.get_device_capability()
    entries = matrix(args.matrix, len(gpus), cap.to_int() if cap else 90)
    if args.only:
        entries = [e for e in entries if re.search(args.only, e.cfg.name)]
    if args.explain:
        byname = {e.cfg.name: e for e in matrix(args.matrix, len(gpus), 90)}
        byname.update({e.cfg.name: e for e in entries})
        x = byname[args.explain.split(" (")[0]]
        rname = args.explain_ref or next(
            e.cfg.name
            for e in byname.values()
            if e.role in ("ref", "bi_ref")
            and e.cfg.family == x.cfg.family
            and (e.role == "ref") == (x.role != "bi_test")
        )
        if x.role == "bi_test" and not args.explain_ref:
            rname = x.bi_ref
        R = RunResult.load(result_path(byname[rname].cfg, args.out))
        X = RunResult.load(result_path(x.cfg, args.out))
        sp = [f for f in os.listdir(args.out) if f.startswith(f"suite.{args.suite}.")]
        with open(os.path.join(args.out, sp[0])) as f:
            suite = json.load(f)
        print(f"# {x.cfg.name} vs {rname}")
        print(explain(R, X, suite, use_pass2="(warm)" in args.explain))
        return
    if args.report_only:
        results = {}
        for e in entries:
            p = result_path(e.cfg, args.out)
            if os.path.exists(p):
                results[e.cfg.name] = RunResult.load(p)
    else:
        results = run_matrix(
            entries,
            args.out,
            gpus,
            args.gsm8k,
            args.global_ref,
            args.python,
            suite_size=args.suite,
        )
    refseq = None
    rs = [f for f in sorted(os.listdir(args.out)) if f.startswith("refseq.")]
    if rs:
        refseq = RunResult.load(os.path.join(args.out, rs[0]))
    rep = evaluate(
        entries,
        results,
        safety=args.safety,
        bf16_family=args.bf16_family,
        refseq=refseq,
    )
    with open(os.path.join(args.out, f"report.{args.matrix}.json"), "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(args.out, f"report.{args.matrix}.md"), "w") as f:
        f.write(render_markdown(rep, f"## {args.matrix}"))
    print(render_markdown(rep, f"## {args.matrix}"))


if __name__ == "__main__":
    main()
