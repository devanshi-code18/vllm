# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-config parity: compare serving configs against a reference config.

A *run* executes one `ParityConfig` on a fixed prompt suite and records, per
prompt, the greedy continuation with its top-k decode logprobs, the top-k
teacher-forced logprobs of the reference sequence (prompt + reference
continuation, scored through `prompt_logprobs`), and optionally GSM8K answers.

`compare` turns two runs into a flat dict of statistics. `NoiseModel` fits a
threshold per statistic from pairs of runs that differ only in things that
should not matter numerically (batch size, chunk size, run to run), after
screening out runs that disagree with the consensus, and `judge` flags a
config whose statistics fall outside the noise band. Under
`VLLM_BATCH_INVARIANT=1`, `exact_diff` requires bitwise equality instead.

This module does not import vLLM at import time, so the statistics and the
noise model can be unit-tested on CPU.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Statistics where larger means "further from the reference"; each gets a
# threshold from the noise model. The per-token ones pool two paths that
# measure the same quantity, |dlogprob| of one token given one context:
# teacher-forced scoring of the reference sequences (prefill path) and the
# greedy prefix both runs share (decode path). On a given stack, batch size
# often leaves prefill bitwise unchanged while decode moves, so prefill-only
# noise would give a zero threshold.
LOWER_IS_BETTER = (
    "tok_mean",  # mean |dlogprob|, max over the two paths
    "tok_p99",
    "top1_disagree",  # TF argmax disagreement / decode flips per token
    "topk_miss",  # 1 - mean top-k overlap
    "nll_shift",  # |mean signed dlogprob of the scored tokens|
    "flip_margin",  # largest ref top1-top2 margin where the argmax differs
    "copy_flip_margin",  # same, between identical prompts inside one run
    "gsm8k_drop",  # ref accuracy - config accuracy (one-sided)
)

# Floors keep a threshold usable when every noise pair is bitwise equal.
THRESHOLD_FLOOR = {
    "tok_mean": 1e-3,
    "tok_p99": 1e-2,
    "top1_disagree": 1e-3,
    "topk_miss": 1e-3,
    "nll_shift": 1e-3,
    "flip_margin": 5e-2,
    "copy_flip_margin": 5e-2,
    "gsm8k_drop": 0.015,
}


@dataclass(frozen=True)
class ParityConfig:
    """One serving configuration. Everything here is passed to `LLM`."""

    name: str
    model: str
    quantization: str | None = None
    attention_backend: str | None = None
    tensor_parallel_size: int = 1
    max_num_seqs: int = 1
    max_num_batched_tokens: int | None = None
    enable_prefix_caching: bool = False
    speculative_config: dict | None = None
    batch_invariant: bool = False
    kv_cache_dtype: str = "auto"
    max_model_len: int = 8192
    env: tuple[tuple[str, str], ...] = ()
    extra: tuple[tuple[str, Any], ...] = ()
    # Numerics family: configs in one family should match within noise.
    family: str = ""
    flash_attn_version: int | None = None

    def key(self) -> str:
        d = dataclasses.asdict(self)
        d.pop("name")
        if d["flash_attn_version"] is None:
            d.pop("flash_attn_version")
        blob = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def engine_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = dict(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens or self.max_model_len,
            max_model_len=self.max_model_len,
            enable_prefix_caching=self.enable_prefix_caching,
            kv_cache_dtype=self.kv_cache_dtype,
            seed=0,
        )
        if self.quantization:
            kw["quantization"] = self.quantization
        if self.attention_backend:
            kw["attention_config"] = {"backend": self.attention_backend}
            if self.flash_attn_version is not None:
                kw["attention_config"]["flash_attn_version"] = self.flash_attn_version
        if self.speculative_config:
            kw["speculative_config"] = dict(self.speculative_config)
        kw.update(dict(self.extra))
        return kw

    def run_env(self) -> dict[str, str]:
        env = dict(self.env)
        env["VLLM_BATCH_INVARIANT"] = "1" if self.batch_invariant else "0"
        return env

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ParityConfig:
        d = dict(d)
        d["env"] = tuple(tuple(x) for x in d.get("env", ()))
        d["extra"] = tuple(tuple(x) for x in d.get("extra", ()))
        return cls(**d)

    def replace(self, **kw: Any) -> ParityConfig:
        return dataclasses.replace(self, **kw)


# --------------------------------------------------------------------------
# Run records
# --------------------------------------------------------------------------


@dataclass
class PromptRecord:
    """Per-prompt outputs of one run. Top-k lists are rank-ordered."""

    prompt_id: str
    kind: str
    gen_ids: list[int]
    # Per decode step: logprob of the generated token, then top-k ids and
    # logprobs at that step.
    gen_lp: list[float]
    gen_topk_ids: list[list[int]]
    gen_topk_lp: list[list[float]]
    # Teacher-forced over the scored sequence (positions 1..n-1).
    tf_lp: list[float] = field(default_factory=list)
    tf_topk_ids: list[list[int]] = field(default_factory=list)
    tf_topk_lp: list[list[float]] = field(default_factory=list)
    # Top1 - top2 logprob at each teacher-forced position.
    tf_margin: list[float] = field(default_factory=list)
    # Hash of the scored token ids: two runs are comparable teacher-forced
    # only if they scored the same sequence.
    tf_src: str = ""


@dataclass
class RunResult:
    config: dict[str, Any]
    prompts: list[PromptRecord]
    # Second pass over the same prompts (prefix caching warm), if any.
    prompts_pass2: list[PromptRecord] | None = None
    gsm8k_preds: list[int] | None = None
    gsm8k_labels: list[int] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def gsm8k_acc(self) -> float | None:
        if not self.gsm8k_preds:
            return None
        hits = sum(p == q for p, q in zip(self.gsm8k_preds, self.gsm8k_labels or []))
        return hits / len(self.gsm8k_preds)

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> RunResult:
        d = dict(d)
        d["prompts"] = [PromptRecord(**p) for p in d["prompts"]]
        if d.get("prompts_pass2") is not None:
            d["prompts_pass2"] = [PromptRecord(**p) for p in d["prompts_pass2"]]
        return cls(**d)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f)

    @classmethod
    def load(cls, path: str) -> RunResult:
        with open(path) as f:
            return cls.from_json(json.load(f))


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if x.size else 0.0


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """Index of the first differing token, or None if one is a prefix of the
    other and they have equal length."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def copy_consistency(prompts: list[PromptRecord]) -> dict[str, Any]:
    """Identical prompts (`copyG.C`) within one run, against the group's first
    copy: how many differ, and the first copy's top1-top2 margin where the
    greedy tokens part. Copies scheduled together hit each other's prefix
    blocks when prefix caching is on, so this sees same-step cache races."""
    groups: dict[str, list[PromptRecord]] = {}
    for p in prompts:
        if p.kind == "copy":
            groups.setdefault(p.prompt_id.split(".")[0], []).append(p)
    n, bad, margins = 0, 0, []
    for g in groups.values():
        head = g[0]
        for c in g[1:]:
            n += 1
            div = first_divergence(head.gen_ids, c.gen_ids)
            if div is None:
                continue
            bad += 1
            if div < len(head.gen_topk_lp) and len(head.gen_topk_lp[div]) >= 2:
                margins.append(head.gen_topk_lp[div][0] - head.gen_topk_lp[div][1])
    return dict(
        copy_pairs=n,
        copy_mismatch=bad / n if n else 0.0,
        copy_flip_margin=max(margins) if margins else 0.0,
    )


def compare(
    ref: RunResult,
    other: RunResult,
    use_pass2: bool = False,
) -> dict[str, Any]:
    """Statistics of `other` against `ref` (see `LOWER_IS_BETTER`).

    Teacher-forced statistics need `other` to have scored the reference's
    sequences (the runner does this). Decode statistics use each side's own
    greedy continuation up to the first divergence.
    """
    ref_prompts = ref.prompts
    oth_prompts = other.prompts_pass2 if use_pass2 else other.prompts
    if oth_prompts is None:
        raise ValueError("run has no second pass")
    by_id = {p.prompt_id: p for p in oth_prompts}
    # Teacher-forced scoring happens once per run (it never reads the prefix
    # cache), so a second pass reuses the first pass's scores.
    tf_by_id = {p.prompt_id: p for p in other.prompts}

    tf_d, tf_top1, tf_overlap, nll_r, nll_o = [], [], [], [], []
    dec_d: list[float] = []
    dec_signed: list[float] = []
    dec_overlap: list[float] = []
    per_prompt = []
    flip_margins: list[float] = []
    tf_flip_margins: list[float] = []
    n_match = 0
    for rp in ref_prompts:
        op = by_id[rp.prompt_id]
        tp = tf_by_id[rp.prompt_id]
        # Teacher-forced: both sides must have scored the same sequence.
        if rp.tf_src and tp.tf_src and rp.tf_src != tp.tf_src:
            raise ValueError(
                f"{rp.prompt_id}: runs scored different sequences; score both "
                "against the same frozen reference sequences"
            )
        m = min(len(rp.tf_lp), len(tp.tf_lp))
        if m:
            a = np.asarray(rp.tf_lp[:m], dtype=np.float64)
            b = np.asarray(tp.tf_lp[:m], dtype=np.float64)
            tf_d.append(np.abs(a - b))
            nll_r.append(-a)
            nll_o.append(-b)
            k = max(1, min(len(rp.tf_topk_ids[0]), len(tp.tf_topk_ids[0])))
            margin_src = rp.tf_margin or tp.tf_margin
            for i in range(m):
                ri, oi = rp.tf_topk_ids[i][:k], tp.tf_topk_ids[i][:k]
                tf_top1.append(ri[0] == oi[0])
                tf_overlap.append(len(set(ri) & set(oi)) / k)
                if ri[0] != oi[0] and i < len(margin_src):
                    tf_flip_margins.append(margin_src[i])
        # Decode: shared greedy prefix.
        div = first_divergence(rp.gen_ids, op.gen_ids)
        shared = len(rp.gen_ids) if div is None else div
        if shared:
            diff = np.asarray(op.gen_lp[:shared]) - np.asarray(rp.gen_lp[:shared])
            dec_d.append(np.abs(diff))
            dec_signed.append(diff)
            for i in range(min(shared, len(rp.gen_topk_ids), len(op.gen_topk_ids))):
                ri, oi = rp.gen_topk_ids[i], op.gen_topk_ids[i]
                k = max(1, min(len(ri), len(oi)))
                dec_overlap.append(len(set(ri[:k]) & set(oi[:k])) / k)
        margin = None
        if div is None:
            n_match += 1
        elif div < len(rp.gen_topk_lp) and len(rp.gen_topk_lp[div]) >= 2:
            top = rp.gen_topk_lp[div]
            margin = float(top[0] - top[1])
            flip_margins.append(margin)
        per_prompt.append(
            dict(
                prompt_id=rp.prompt_id,
                kind=rp.kind,
                first_div=div,
                gen_len=len(rp.gen_ids),
                flip_margin=margin,
                tf_max=float(tf_d[-1].max()) if m else 0.0,
                tf_argmax=int(tf_d[-1].argmax()) if m else -1,
            )
        )

    tf = np.concatenate(tf_d) if tf_d else np.zeros(0)
    dec = np.concatenate(dec_d) if dec_d else np.zeros(0)
    divs = [r["first_div"] for r in per_prompt if r["first_div"] is not None]
    dsg = np.concatenate(dec_signed) if dec_signed else np.zeros(0)
    stats: dict[str, Any] = dict(
        n_prompts=len(ref_prompts),
        n_tf_positions=int(tf.size),
        tf_mean=float(tf.mean()) if tf.size else 0.0,
        tf_p50=_pct(tf, 50),
        tf_p90=_pct(tf, 90),
        tf_p99=_pct(tf, 99),
        tf_p999=_pct(tf, 99.9),
        tf_max=float(tf.max()) if tf.size else 0.0,
        tf_top1_disagree=1.0 - float(np.mean(tf_top1)) if tf_top1 else 0.0,
        tf_topk_miss=1.0 - float(np.mean(tf_overlap)) if tf_overlap else 0.0,
        tf_nll_ref=float(np.concatenate(nll_r).mean()) if nll_r else 0.0,
        tf_nll=float(np.concatenate(nll_o).mean()) if nll_o else 0.0,
        dec_match_rate=n_match / max(1, len(ref_prompts)),
        dec_first_div_median=float(np.median(divs)) if divs else None,
        dec_n_tokens=int(dec.size),
        dec_mean=float(dec.mean()) if dec.size else 0.0,
        dec_p99=_pct(dec, 99),
        dec_p999=_pct(dec, 99.9),
        dec_max=float(dec.max()) if dec.size else 0.0,
        dec_shift=abs(float(dsg.mean())) if dsg.size else 0.0,
        dec_topk_miss=1.0 - float(np.mean(dec_overlap)) if dec_overlap else 0.0,
        dec_n_flips=len(flip_margins),
        dec_flip_rate=len(flip_margins) / max(1, int(dec.size) + len(flip_margins)),
        dec_flip_margin=max(flip_margins) if flip_margins else 0.0,
        tf_flip_margin=max(tf_flip_margins) if tf_flip_margins else 0.0,
    )
    stats["tf_dnll"] = abs(stats["tf_nll"] - stats["tf_nll_ref"])
    stats["tok_mean"] = max(stats["tf_mean"], stats["dec_mean"])
    stats["tok_p99"] = max(stats["tf_p99"], stats["dec_p99"])
    stats["tok_p999"] = max(stats["tf_p999"], stats["dec_p999"])
    stats["top1_disagree"] = max(stats["tf_top1_disagree"], stats["dec_flip_rate"])
    stats["topk_miss"] = max(stats["tf_topk_miss"], stats["dec_topk_miss"])
    stats["nll_shift"] = max(stats["tf_dnll"], stats["dec_shift"])
    stats["flip_margin"] = max(stats["dec_flip_margin"], stats["tf_flip_margin"])
    stats.update(copy_consistency(oth_prompts))
    ra, oa = ref.gsm8k_acc, other.gsm8k_acc
    if ra is not None and oa is not None:
        stats["gsm8k_ref"] = ra
        stats["gsm8k"] = oa
        stats["gsm8k_drop"] = max(0.0, ra - oa)
    stats["per_prompt"] = per_prompt
    return stats


def seq_hash(ids: list[int]) -> str:
    return hashlib.sha1(json.dumps(ids).encode()).hexdigest()[:16]


def exact_diff(
    ref: RunResult, other: RunResult, use_pass2: bool = False, ref_pass2: bool = False
) -> dict:
    """Bitwise comparison of greedy tokens and decode/teacher-forced logprobs.

    `use_pass2` / `ref_pass2` pick the warm (second) decode pass of `other` /
    `ref`. Returns counts of prompts that differ; all zero means bitwise equal.
    """
    oth = other.prompts_pass2 if use_pass2 else other.prompts
    refp = ref.prompts_pass2 if ref_pass2 else ref.prompts
    assert oth is not None and refp is not None
    by_id = {p.prompt_id: p for p in oth}
    tf_by_id = {p.prompt_id: p for p in other.prompts}
    tf_ref = {p.prompt_id: p for p in ref.prompts}
    tok, dec, tf, bad = 0, 0, 0, []
    for rp in refp:
        op, tp = by_id[rp.prompt_id], tf_by_id[rp.prompt_id]
        rt = tf_ref[rp.prompt_id]
        t = rp.gen_ids != op.gen_ids
        d = rp.gen_lp != op.gen_lp or rp.gen_topk_lp != op.gen_topk_lp
        f = rt.tf_lp != tp.tf_lp or rt.tf_topk_ids != tp.tf_topk_ids
        tok += t
        dec += d
        tf += f
        if t or d or f:
            bad.append(rp.prompt_id)
    return dict(
        n_prompts=len(refp),
        tokens_differ=tok,
        decode_logprobs_differ=dec,
        tf_logprobs_differ=tf,
        bitwise_equal=not bad,
        differing_prompts=bad,
    )


def decode_vs_prefill(run: RunResult, ref: RunResult) -> dict[str, Any]:
    """Within one run: decode logprobs vs teacher-forced (prefill) logprobs of
    the same tokens, over the greedy prefix `run` shares with `ref` (whose
    sequences `run` scored). Under batch invariance these must be bitwise
    equal: it is what lets a trainer recompute the sampler's logprobs.
    """
    by_id = {p.prompt_id: p for p in run.prompts}
    n, bad, worst = 0, 0, 0.0
    for rp in ref.prompts:
        op = by_id[rp.prompt_id]
        div = first_divergence(rp.gen_ids, op.gen_ids)
        shared = len(rp.gen_ids) if div is None else div
        off = len(op.tf_lp) - len(rp.gen_ids)
        for j in range(shared):
            a, b = op.gen_lp[j], op.tf_lp[off + j]
            n += 1
            if a != b:
                bad += 1
                worst = max(worst, abs(a - b))
    return dict(n_compared=n, n_differ=bad, max_abs=worst)


# --------------------------------------------------------------------------
# Noise model
# --------------------------------------------------------------------------


@dataclass
class NoiseModel:
    """Threshold per statistic: `safety * max(noise pair values)`, floored.

    With few noise pairs the maximum is the q=100 percentile of the pair
    values; `quantile` can lower it when many pairs are available.
    """

    thresholds: dict[str, float]
    noise_max: dict[str, float]
    n_pairs: int
    safety: float
    quantile: float
    pair_names: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    degenerate: bool = False
    reference_outlier: bool = False
    consensus: list[str] = field(default_factory=list)
    consistency_bound: float = 0.0

    @classmethod
    def fit(
        cls,
        pairs: dict[str, dict[str, Any]],
        safety: float = 3.0,
        quantile: float = 100.0,
    ) -> NoiseModel:
        if not pairs:
            raise ValueError("noise model needs at least one pair")
        noise_max, thr = {}, {}
        for s in LOWER_IS_BETTER:
            vals = [p[s] for p in pairs.values() if s in p]
            if not vals:
                thr[s] = THRESHOLD_FLOOR[s]
                continue
            q = float(np.percentile(vals, quantile))
            noise_max[s] = q
            thr[s] = max(safety * q, THRESHOLD_FLOOR[s])
        return cls(
            thresholds=thr,
            noise_max=noise_max,
            n_pairs=len(pairs),
            safety=safety,
            quantile=quantile,
            pair_names=sorted(pairs),
        )

    @classmethod
    def fit_robust(
        cls,
        runs: dict[str, RunResult],
        safety: float = 3.0,
        quantile: float = 100.0,
        peers: dict[str, RunResult] | None = None,
        consistency_factor: float = 9.0,
        reference: str = "reference",
    ) -> tuple[NoiseModel, list[str], dict[str, dict[str, Any]]]:
        """Fit on calibration `runs` that agree with each other.

        Two runs are *consistent* when both their prefill (teacher-forced)
        and decode p99 |dlogprob| stay within `consistency_factor` times the
        lower quartile of the decode-path p99 over all pairs: batch size moves
        decode numerics on every stack, so that is a stack-local bf16 scale
        that a broken prefill kernel cannot hide in. Bitwise-identical runs
        count once. The largest set of mutually consistent runs among the
        calibration runs and `peers` (other backends at batch 1) is the
        consensus; calibration runs outside it are rejected and returned, so
        a bug on the batched path cannot widen its own threshold.

        If the reference is not in the consensus, or fewer than two distinct
        calibration runs are, the model is `degenerate` (the reference stack
        disagrees with itself or with every other backend) and is fit on the
        consensus pairs instead.
        Returns the model, the rejected run names, and the pair statistics.
        """
        peers = peers or {}
        pool = {**{f"peer:{k}": v for k, v in peers.items()}, **runs}
        names = sorted(pool)
        pair: dict[tuple[str, str], dict[str, Any]] = {}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                pair[(a, b)] = compare(pool[a], pool[b])

        def st(a: str, b: str) -> dict[str, Any]:
            return pair[(a, b)] if (a, b) in pair else pair[(b, a)]

        def same(a: str, b: str) -> bool:
            s_ = st(a, b)
            return s_["tf_max"] == 0 and s_["dec_max"] == 0 and s_["dec_n_flips"] == 0

        # Lower quartile, not median: a bug that also moves decode inflates
        # the pairs it touches, and it may touch most of them.
        dec = [st(a, b)["dec_p99"] for (a, b) in pair if not same(a, b)]
        scale = float(np.percentile(dec, 25)) if dec else 0.0
        bound = consistency_factor * scale

        def ok(a: str, b: str) -> bool:
            s_ = st(a, b)
            return bound > 0 and s_["tf_p99"] <= bound and s_["dec_p99"] <= bound

        # Collapse bitwise-identical runs, then find the largest clique.
        groups: list[list[str]] = []
        for n in names:
            for g in groups:
                if same(g[0], n):
                    g.append(n)
                    break
            else:
                groups.append([n])
        best: list[list[str]] = []
        for mask in range(1, 1 << len(groups)):
            sel = [g for i, g in enumerate(groups) if mask >> i & 1]
            if len(sel) < len(best):
                continue
            if all(ok(a[0], b[0]) for i, a in enumerate(sel) for b in sel[i + 1 :]):
                has_ref = any(reference in g for g in sel)
                best_ref = any(reference in g for g in best)
                if len(sel) > len(best) or (has_ref and not best_ref):
                    best = sel
        consensus = sorted(n for g in best for n in g)
        rejected = [n for n in sorted(runs) if n not in consensus]
        cal = [n for n in sorted(runs) if n in consensus]
        kept = {
            f"{a} vs {b}": st(a, b) for i, a in enumerate(cal) for b in cal[i + 1 :]
        }
        distinct = len({tuple(g) for g in best if any(n in runs for n in g)})
        reference_outlier = reference not in consensus
        degenerate = reference_outlier or distinct < 2
        if degenerate:
            kept = {
                f"{a} vs {b}": st(a, b)
                for i, a in enumerate(consensus)
                for b in consensus[i + 1 :]
            }
        model = cls.fit(kept or {"none": {}}, safety=safety, quantile=quantile)
        model.rejected = rejected
        model.degenerate = degenerate
        model.reference_outlier = reference_outlier
        model.consensus = consensus
        model.consistency_bound = bound
        all_pairs = {f"{a} vs {b}": v for (a, b), v in pair.items()}
        return model, rejected, all_pairs

    def judge(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Return `{"flagged": bool, "violations": {stat: ratio}}`."""
        viol = {}
        for s, t in self.thresholds.items():
            if s in stats and stats[s] > t:
                viol[s] = stats[s] / t
        return dict(flagged=bool(viol), violations=viol)

    def describe(self) -> str:
        rej = (
            f"; rejected as inconsistent: {', '.join(self.rejected)}"
            if self.rejected
            else ""
        )
        if self.reference_outlier:
            rej += (
                "; REFERENCE OUTLIER: the reference disagrees with the consensus "
                f"({', '.join(self.consensus) or 'none'}), which sets the noise"
            )
        elif self.degenerate:
            rej += (
                "; fewer than two distinct calibration runs survive, so the "
                f"consensus ({', '.join(self.consensus)}) sets the noise"
            )
        return (
            f"threshold(stat) = max({self.safety:g} x p{self.quantile:g} of the "
            f"stat over {self.n_pairs} noise pairs, floor); "
            f"noise pairs: {', '.join(self.pair_names)}{rej}"
        )

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
