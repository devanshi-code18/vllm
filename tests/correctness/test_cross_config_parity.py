# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-config parity: serving configs must match a reference within noise.

The reference is bf16 / TP1 / FLASH_ATTN / one sequence at a time. Each test
config changes one serving axis (attention backend, chunked-prefill size,
prefix caching, speculative decoding, TP size, batch size) and is judged on
teacher-forced logprobs of the reference sequences, the greedy prefix it
shares with the reference, and the confidence of any greedy flip.

Thresholds are not hand-picked: each statistic's threshold is `safety` times
its largest value over noise pairs (reference vs itself in a new process, vs
batch 8, vs batch 64), floored. Under `VLLM_BATCH_INVARIANT=1` the same axes
must be bitwise equal.

`VLLM_PARITY_MODE=ci` (default) runs Qwen3-0.6B on 1-2 GPUs in a few
minutes; `full` runs the Qwen3-8B matrix with quantized families, GSM8K and
TP2 (see `parity_matrix.py` to run it outside pytest and write a report).
"""

import os

import pytest
import torch

from tests.correctness.parity import RunResult
from tests.correctness.parity_matrix import evaluate, matrix, run_matrix
from vllm.platforms import current_platform

MODE = os.getenv("VLLM_PARITY_MODE", "ci")
N_GPUS = torch.accelerator.device_count() if current_platform.is_cuda() else 0
_cap = current_platform.get_device_capability() if N_GPUS else None
SM = _cap.to_int() if _cap else 0

pytestmark = pytest.mark.skipif(
    N_GPUS == 0 or SM < 80, reason="Requires CUDA >= Ampere (SM80)"
)

ENTRIES = matrix(MODE, min(N_GPUS, 2), SM) if N_GPUS else []
DEFAULT_TESTS = [e.cfg.name for e in ENTRIES if e.role == "test"]
BI_TESTS = [e.cfg.name for e in ENTRIES if e.role == "bi_test" and "vs" not in e.axis]
RERUNS = [e.rerun_of for e in ENTRIES if e.role == "rerun"]


@pytest.fixture(scope="module")
def parity_report(tmp_path_factory):
    out = os.getenv("VLLM_PARITY_OUT") or str(tmp_path_factory.mktemp("parity"))
    gpus = [str(i) for i in range(min(N_GPUS, 2))]
    gsm8k_n = int(os.getenv("VLLM_PARITY_GSM8K", "200" if MODE == "full" else "0"))
    results = run_matrix(
        ENTRIES, out, gpus, gsm8k_n, suite_size="full" if MODE == "full" else "ci"
    )
    fam = "bf16" if MODE == "full" else "ci_bf16"
    rs = [f for f in sorted(os.listdir(out)) if f.startswith("refseq.")]
    refseq = RunResult.load(os.path.join(out, rs[0])) if rs else None
    report = evaluate(ENTRIES, results, bf16_family=fam, refseq=refseq)
    report["out_dir"] = out
    return report


def _rows(report, name):
    rows = [r for r in report["rows"] if r["name"].split(" (")[0] == name]
    assert rows, f"{name} did not run; see logs in {report['out_dir']}"
    return rows


def test_reference_stack_is_self_consistent(parity_report):
    """Batch-size, chunk-size and rerun variants of each family reference must
    agree with each other and with the other backends; a calibration run that
    does not is rejected (and the family's reference may be the odd one out)."""
    for fam, info in parity_report["families"].items():
        assert not info.get("reference_outlier"), (
            f"{fam}: {info['reference']} disagrees with {info['consensus']}"
        )
        assert not info.get("rejected_from_calibration"), (
            f"{fam}: inconsistent calibration runs {info['rejected_from_calibration']}"
        )


@pytest.mark.parametrize("name", DEFAULT_TESTS)
def test_default_mode_parity(parity_report, name):
    for row in _rows(parity_report, name):
        assert not row.get("error"), f"{row['name']} failed to run"
        v = row["verdict"]
        assert v is not None, f"no noise model for {row['family']}"
        assert not v["flagged"], (
            f"{row['name']} is outside the noise band "
            f"(stat / threshold: {v['violations']}); "
            f"stats {row['stats']}; out dir {parity_report['out_dir']}"
        )


@pytest.mark.parametrize("name", RERUNS)
def test_run_to_run_determinism(parity_report, name):
    """Greedy tokens must reproduce in a fresh process at least as well as the
    same batch size with prefix caching off: a difference is a race (e.g. a
    same-step prefix-cache hit reading state another request still writes),
    not batch-size noise."""
    rows = [d for d in parity_report["determinism"] if d["name"] == name]
    assert rows and not rows[0].get("error"), f"{name} did not run"
    d = rows[0]
    assert d["reproducible"], (
        f"{name} differs from its own rerun: greedy tokens in "
        f"{d['tokens_differ']} prompts (warm pass {d.get('warm_tokens_differ')}) "
        f"of {d['n_prompts']}; control {d.get('control')}"
    )


@pytest.mark.parametrize("name", BI_TESTS)
def test_batch_invariant_bitwise(parity_report, name):
    rows = [b for b in parity_report["bi"] if b["name"].split(" (")[0] == name]
    assert rows, f"{name} did not run; see logs in {parity_report['out_dir']}"
    for b in rows:
        assert not b.get("error"), f"{b['name']} failed to run"
        assert b["bitwise_equal"], (
            f"{b['name']}: tokens differ in {b['tokens_differ']}, logprobs in "
            f"{max(b['decode_logprobs_differ'], b['tf_logprobs_differ'])} of "
            f"{b['n_prompts']} prompts ({b['differing_prompts']})"
        )
