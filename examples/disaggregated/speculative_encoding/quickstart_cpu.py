# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The whole speculative-encoding flow on a laptop CPU, with no downloads:
tiny random models and a random translator, so the answers are noise.

    pip install --no-deps "speculative-encoding @ git+https://github.com/devanshi-code18/speculative-encoding@e5856fbfbdb8b847d4c5aa099a45fe06e54d1c75"
    python examples/disaggregated/speculative_encoding/quickstart_cpu.py

1. A tiny random Llama reads a document, and a random translator turns its
   hidden states into the KV cache of a tiny random Qwen3.
2. The KV is published to a store.
3. vLLM serves the Qwen3 with `SpeculativeEncodingConnector`: a request whose
   prompt starts with the document loads the document's KV from the store;
   the same prompt without the document reference prefills it.
"""

import tempfile
from pathlib import Path

import torch
import transformers as T
from speculative_encoding.api import SpeculativeEncoder
from speculative_encoding.translator.hf import ModelInterface
from speculative_encoding.translator.model import UnifiedConfig, UnifiedTranslator
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.speculative_encoding import (
    write_document,
)
from vllm.inputs import TokensPrompt

DOCUMENT = (
    "The Harbor Lease was signed on 3 March 2021 by Mira Okafor for the landlord "
    "and by Tomas Reyes for Bluewater Cargo. The rent is 4,200 euros a month, due "
    "on the first working day. Either side may end the lease with ninety days of "
    "written notice."
)
QUESTION = "Who signed the lease for Bluewater Cargo?"
CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def tiny_tokenizer(vocab_size: int) -> T.PreTrainedTokenizerFast:
    specials = ["<|bos|>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>"]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator([DOCUMENT, QUESTION] * 4, trainer)
    return T.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|bos|>",
        eos_token="<|end|>",
        pad_token="<|end|>",
        chat_template=CHAT_TEMPLATE,
    )


def tiny_model(family: str, tokenizer, seed: int):
    config, cls = {
        "llama": (T.LlamaConfig, T.LlamaForCausalLM),
        "qwen3": (T.Qwen3Config, T.Qwen3ForCausalLM),
    }[family]
    torch.manual_seed(seed)
    model = cls(
        config(
            vocab_size=len(tokenizer),
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_hidden_layers=2,
            max_position_embeddings=1024,
            tie_word_embeddings=False,
        )
    )
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    return ModelInterface(model.eval().requires_grad_(False), family), tokenizer


def main():
    source, target = "tiny-llama", "tiny-qwen3"
    models = {
        source: tiny_model("llama", tiny_tokenizer(400), 0),
        target: tiny_model("qwen3", tiny_tokenizer(500), 1),
    }
    torch.manual_seed(2)
    translator = UnifiedTranslator(
        {target: models[target][0].target_geometry()},
        source_geometries={source: models[source][0].source_geometry()},
        config=UnifiedConfig(
            width=64, heads=4, ffn=128, experts=4, top_k=2, decoder_blocks=2, window=8
        ),
    )
    with torch.no_grad():  # a fresh translator is a no-op: randomize its outputs
        for parameter in translator.parameters():
            if parameter.dim() > 1 and not parameter.any():
                parameter.normal_(0.0, 0.05)
    encoder = SpeculativeEncoder(
        translator, source, target, device="cpu", models=models
    )
    encoded = encoder.encode(DOCUMENT).full
    question = encoder.question(QUESTION, max_tokens=8)

    with tempfile.TemporaryDirectory() as folder:
        store, model_dir = Path(folder) / "store", Path(folder) / "qwen3"
        write_document(store, "lease", encoded.prefix_ids, encoded.prefix)
        interface, tokenizer = models[target]
        interface.model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        llm = LLM(
            str(model_dir),
            dtype="float32",
            max_model_len=512,
            kv_cache_memory_bytes=64 << 20,
            kv_transfer_config=KVTransferConfig(
                kv_connector="SpeculativeEncodingConnector",
                kv_role="kv_both",
                kv_connector_extra_config={"store": str(store)},
            ),
        )
        prompt = list(encoded.prefix_ids) + list(question.after_ids)
        loads = {"speculative_encoding": {"document": "lease"}}
        for name, extra in (
            ("translated", {"kv_transfer_params": loads}),
            ("native", {}),
        ):
            output = llm.generate(
                TokensPrompt(prompt_token_ids=prompt, cache_salt=name),
                SamplingParams(temperature=0.0, max_tokens=8, extra_args=extra),
            )[0]
            print(
                f"{name}: {output.num_cached_tokens} of {len(prompt)} prompt tokens "
                f"from the store, answer {output.outputs[0].token_ids}"
            )


if __name__ == "__main__":
    main()
