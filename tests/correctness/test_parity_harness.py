# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the parity statistics and the noise model (no GPU, no model).

Synthetic runs stand in for engine outputs: a reference, several noise runs
(tiny logprob jitter, greedy flips only at near-ties), and a broken run whose
logprobs are off by a real amount. The noise model must pass held-out noise
and flag the broken run.
"""

import numpy as np
import pytest

from tests.correctness.parity import (
    NoiseModel,
    ParityConfig,
    PromptRecord,
    RunResult,
    compare,
    exact_diff,
    first_divergence,
)

V, K, N_PROMPTS, GEN, TF = 50, 4, 12, 24, 64


def _logits(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(0, 3, size=(n, V))


def _logprobs(x: np.ndarray) -> np.ndarray:
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def _run(base: list[dict], jitter: float, seed: int, bias: float = 0.0) -> RunResult:
    """Score the same token sequences as `base` with perturbed logits."""
    rng = np.random.default_rng(seed)
    prompts = []
    for i, b in enumerate(base):
        tf = _logprobs(b["tf_logits"] + rng.normal(0, jitter, b["tf_logits"].shape))
        tf[:, 0] += bias  # a systematic error, e.g. a wrong kernel
        tf = _logprobs(tf)
        gl = _logprobs(b["gen_logits"] + rng.normal(0, jitter, b["gen_logits"].shape))
        gl[:, 0] += bias
        gl = _logprobs(gl)
        gen_ids = gl.argmax(-1).tolist()
        order = np.argsort(-gl, axis=-1)[:, :K]
        torder = np.argsort(-tf, axis=-1)[:, :K]
        seq = b["tf_tokens"]
        prompts.append(
            PromptRecord(
                prompt_id=f"p{i}",
                kind="qa",
                gen_ids=gen_ids,
                gen_lp=[float(gl[j, t]) for j, t in enumerate(gen_ids)],
                gen_topk_ids=order.tolist(),
                gen_topk_lp=[
                    [float(gl[j, t]) for t in row] for j, row in enumerate(order)
                ],
                tf_lp=[float(tf[j, t]) for j, t in enumerate(seq)],
                tf_topk_ids=torder.tolist(),
                tf_margin=[
                    float(tf[j, r[0]] - tf[j, r[1]]) for j, r in enumerate(torder)
                ],
            )
        )
    return RunResult(config={}, prompts=prompts)


@pytest.fixture(scope="module")
def base():
    rng = np.random.default_rng(0)
    out = []
    for _ in range(N_PROMPTS):
        tl = _logits(rng, TF)
        out.append(
            dict(
                tf_logits=tl,
                tf_tokens=rng.integers(0, V, TF).tolist(),
                gen_logits=_logits(rng, GEN),
            )
        )
    return out


def test_first_divergence():
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 5, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2


def test_identical_runs_are_zero(base):
    r = _run(base, 0.0, 1)
    st = compare(r, r)
    assert st["tf_max"] == 0.0 and st["dec_match_rate"] == 1.0
    assert st["top1_disagree"] == 0.0 and st["dec_n_flips"] == 0
    assert exact_diff(r, r)["bitwise_equal"]


def test_exact_diff_sees_one_ulp(base):
    r = _run(base, 0.0, 1)
    s = _run(base, 0.0, 1)
    s.prompts[3].tf_lp[5] = float(np.nextafter(s.prompts[3].tf_lp[5], 0))
    d = exact_diff(r, s)
    assert not d["bitwise_equal"] and d["differing_prompts"] == ["p3"]


def test_noise_model_passes_noise_and_flags_bug(base):
    ref = _run(base, 0.0, 1)
    noise = {f"n{i}": compare(ref, _run(base, 0.01, 10 + i)) for i in range(4)}
    nm = NoiseModel.fit(noise, safety=3.0)
    held_out = compare(ref, _run(base, 0.01, 99))
    assert not nm.judge(held_out)["flagged"]
    bug = compare(ref, _run(base, 0.01, 7, bias=2.0))
    v = nm.judge(bug)
    assert v["flagged"] and "tok_p99" in v["violations"]
    assert "3 x p100" in nm.describe()


def test_confident_flip_is_flagged_but_near_tie_is_not(base):
    ref = _run(base, 0.0, 1)
    noise = {f"n{i}": compare(ref, _run(base, 0.01, 20 + i)) for i in range(4)}
    nm = NoiseModel.fit(noise, safety=3.0)
    # Swap the greedy token at the most confident decode step of prompt 0.
    oth = _run(base, 0.01, 30)
    p = oth.prompts[0]
    margins = [lp[0] - lp[1] for lp in ref.prompts[0].gen_topk_lp]
    j = int(np.argmax(margins))
    p.gen_ids[j] = ref.prompts[0].gen_topk_ids[j][1]
    st = compare(ref, oth)
    assert st["dec_flip_margin"] == pytest.approx(margins[j])
    assert "flip_margin" in nm.judge(st)["violations"]


def test_teacher_forced_confident_flip(base):
    ref = _run(base, 0.0, 1)
    noise = {f"n{i}": compare(ref, _run(base, 0.01, 40 + i)) for i in range(4)}
    nm = NoiseModel.fit(noise, safety=3.0)
    oth = _run(base, 0.01, 50)
    p, r = oth.prompts[2], ref.prompts[2]
    j = int(np.argmax(r.tf_margin))
    p.tf_topk_ids[j] = [r.tf_topk_ids[j][1], r.tf_topk_ids[j][0]] + r.tf_topk_ids[j][2:]
    st = compare(ref, oth)
    assert st["tf_flip_margin"] == pytest.approx(r.tf_margin[j])
    assert "flip_margin" in nm.judge(st)["violations"]


def test_robust_fit_rejects_a_broken_calibration_run(base):
    runs = {"reference": _run(base, 0.0, 1)}
    runs.update({f"n{i}": _run(base, 0.01, 60 + i) for i in range(3)})
    runs["chunked"] = _run(base, 0.01, 70, bias=2.0)
    nm, rejected, _ = NoiseModel.fit_robust(runs, safety=3.0)
    assert rejected == ["chunked"] and "rejected as inconsistent" in nm.describe()
    assert not nm.degenerate
    clean = {k: v for k, v in runs.items() if k != "chunked"}
    nm2, rejected2, _ = NoiseModel.fit_robust(clean, safety=3.0)
    assert rejected2 == [] and nm2.thresholds == nm.thresholds


def test_robust_fit_degenerate_reference_stack(base):
    # Every batched calibration run is broken; only the reference and its
    # rerun agree. Peers (other backends) agree with the reference.
    ref = _run(base, 0.0, 1)
    runs = {"reference": ref, "run2": ref}
    runs.update({f"bs{i}": _run(base, 0.01, 80 + i, bias=2.0) for i in range(3)})
    peers = {f"backend{i}": _run(base, 0.01, 90 + i) for i in range(3)}
    nm, rejected, _ = NoiseModel.fit_robust(runs, safety=3.0, peers=peers)
    assert sorted(rejected) == ["bs0", "bs1", "bs2"]
    assert nm.degenerate and not nm.reference_outlier
    held_out = compare(ref, _run(base, 0.01, 99))
    assert not nm.judge(held_out)["flagged"]
    assert nm.judge(compare(ref, runs["bs0"]))["flagged"]


def test_robust_fit_flags_a_wrong_reference(base):
    # The reference kernel is wrong everywhere, differently per batch size;
    # two other backends agree with each other.
    ref = _run(base, 0.01, 1, bias=2.0)
    runs = {"reference": ref, "run2": ref}
    runs.update(
        {f"bs{i}": _run(base, 0.01, 110 + i, bias=b) for i, b in enumerate((-3, 6, -6))}
    )
    peers = {f"backend{i}": _run(base, 0.01, 120 + i) for i in range(2)}
    nm, rejected, _ = NoiseModel.fit_robust(runs, safety=3.0, peers=peers)
    assert nm.reference_outlier and "REFERENCE OUTLIER" in nm.describe()
    assert nm.consensus == ["peer:backend0", "peer:backend1"]


def test_copy_consistency_sees_a_confident_copy_flip(base):
    run = _run(base, 0.0, 1)
    for i, p in enumerate(run.prompts[:3]):
        p.kind, p.prompt_id = "copy", f"copy0.{i}"
        p.gen_ids, p.gen_lp = list(run.prompts[0].gen_ids), list(run.prompts[0].gen_lp)
        p.gen_topk_lp = [list(x) for x in run.prompts[0].gen_topk_lp]
        p.gen_topk_ids = [list(x) for x in run.prompts[0].gen_topk_ids]
    assert compare(run, run)["copy_mismatch"] == 0.0
    head = run.prompts[0]
    j = int(np.argmax([lp[0] - lp[1] for lp in head.gen_topk_lp]))
    run.prompts[2].gen_ids[j] = head.gen_topk_ids[j][1]
    st = compare(run, run)
    assert st["copy_mismatch"] == 0.5
    assert st["copy_flip_margin"] == head.gen_topk_lp[j][0] - head.gen_topk_lp[j][1]


def test_config_key_ignores_name_and_round_trips():
    c = ParityConfig(name="x", model="m", max_num_seqs=8, env=(("A", "1"),))
    assert c.key() == c.replace(name="y").key()
    assert c.key() != c.replace(max_num_seqs=1).key()
    assert ParityConfig.from_json(c.to_json()) == c
    assert c.run_env()["VLLM_BATCH_INVARIANT"] == "0"
