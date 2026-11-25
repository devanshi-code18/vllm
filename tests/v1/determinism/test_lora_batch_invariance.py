# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-LoRA batch invariance through `LLM`: a request's tokens and logprobs
must not depend on the other adapters, no-LoRA requests, adapter slots or
prefill chunks in its batch, and prompt_logprobs scoring of the generated
tokens must reproduce the generation logprobs, including for an lm_head LoRA.
"""

import contextlib
import json
import random

import torch
from safetensors.torch import save_file
from utils import skip_if_not_cuda

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.config import get_config

MODEL = "Qwen/Qwen3-1.7B"
MAX_TOKENS = 16


def _write_adapter(path, cfg, rank, with_lm_head, seed):
    """Random PEFT adapter; lora_B is nonzero (PEFT initializes it to zero)."""
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    hidden, kv = cfg.hidden_size, cfg.num_key_value_heads * head_dim
    q = cfg.num_attention_heads * head_dim
    shapes = {
        "self_attn.q_proj": (hidden, q),
        "self_attn.k_proj": (hidden, kv),
        "self_attn.v_proj": (hidden, kv),
        "self_attn.o_proj": (q, hidden),
        "mlp.gate_proj": (hidden, cfg.intermediate_size),
        "mlp.up_proj": (hidden, cfg.intermediate_size),
        "mlp.down_proj": (cfg.intermediate_size, hidden),
    }
    modules = {
        f"model.layers.{i}.{name}": shape
        for i in range(cfg.num_hidden_layers)
        for name, shape in shapes.items()
    }
    if with_lm_head:
        modules["lm_head"] = (hidden, cfg.vocab_size)
    gen = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, (fan_in, fan_out) in modules.items():
        b_std = 0.5 if name == "lm_head" else 0.02
        a = torch.randn(rank, fan_in, generator=gen) / fan_in**0.5
        b = torch.randn(fan_out, rank, generator=gen) * b_std / rank**0.5
        tensors[f"base_model.model.{name}.lora_A.weight"] = a.to(torch.bfloat16)
        tensors[f"base_model.model.{name}.lora_B.weight"] = b.to(torch.bfloat16)
    path.mkdir()
    save_file(tensors, path / "adapter_model.safetensors")
    targets = [name.split(".")[-1] for name in shapes]
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": MODEL,
        "r": rank,
        "lora_alpha": 2 * rank,
        "target_modules": targets + (["lm_head"] if with_lm_head else []),
        "bias": "none",
    }
    (path / "adapter_config.json").write_text(json.dumps(config))
    return str(path)


def _tokens_and_logprobs(output):
    completion = output.outputs[0]
    logprobs = [
        lp[t].logprob for lp, t in zip(completion.logprobs, completion.token_ids)
    ]
    return list(completion.token_ids), logprobs


@skip_if_not_cuda
def test_multi_lora_logprobs_bitwise(tmp_path):
    cfg = get_config(MODEL, trust_remote_code=False)
    specs = [("attn_mlp_r8", 8, False), ("attn_mlp_r16", 16, False)]
    specs.append(("lm_head_r8", 8, True))
    loras = [
        LoRARequest(name, i + 1, _write_adapter(tmp_path / name, cfg, r, head, i))
        for i, (name, r, head) in enumerate(specs)
    ]
    rng = random.Random(0)

    def prompt(num_tokens):
        ids = [rng.randint(1000, 30000) for _ in range(num_tokens)]
        return TokensPrompt(prompt_token_ids=ids)

    targets = [(prompt(rng.randint(16, 96)), lora) for lora in [*loras, None]]
    fillers = [
        (prompt(rng.randint(16, 600)), rng.choice([*loras, None])) for _ in range(12)
    ]
    sp = SamplingParams(
        temperature=0.0, max_tokens=MAX_TOKENS, logprobs=1, ignore_eos=True
    )
    # max_loras < number of adapters moves adapters between slots; the token
    # budget splits long fillers into prefill chunks next to decoding targets.
    llm = LLM(
        model=MODEL,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=16,
        max_num_seqs=len(targets) + len(fillers),
        max_num_batched_tokens=256,
        max_model_len=1024,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.2,
        kv_cache_memory_bytes=1 << 30,
    )

    def generate(requests, params):
        prompts, lora_requests = zip(*requests)
        return llm.generate(
            list(prompts), params, lora_request=list(lora_requests), use_tqdm=False
        )

    try:
        alone = [_tokens_and_logprobs(generate([t], sp)[0]) for t in targets]
        order = list(range(len(targets) + len(fillers)))
        rng.shuffle(order)
        requests = targets + fillers
        outputs = generate([requests[i] for i in order], sp)
        mixed = [
            _tokens_and_logprobs(outputs[order.index(i)]) for i in range(len(targets))
        ]
        # Score prompt + generated tokens in one batch with prompt_logprobs.
        scored = [
            (TokensPrompt(prompt_token_ids=p["prompt_token_ids"] + tokens), lora)
            for (p, lora), (tokens, _) in zip(targets, alone)
        ]
        score_sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        scores = generate(scored, score_sp)
    finally:
        with contextlib.suppress(Exception):
            llm.llm_engine.engine_core.shutdown()

    for (p, lora), (tok1, lp1), (tokN, lpN), score in zip(
        targets, alone, mixed, scores
    ):
        name = lora.lora_name if lora else "no LoRA"
        assert tok1 == tokN, f"{name}: tokens differ alone vs mixed batch"
        assert lp1 == lpN, f"{name}: logprobs differ alone vs mixed batch"
        ids = score.prompt_token_ids
        start = len(p["prompt_token_ids"])
        prompt_lp = [
            score.prompt_logprobs[i][ids[i]].logprob for i in range(start, len(ids))
        ]
        assert prompt_lp == lp1, f"{name}: prompt_logprobs differ from generation"
