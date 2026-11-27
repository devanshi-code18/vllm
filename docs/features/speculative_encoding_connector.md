# Speculative Encoding Connector

[Speculative encoding](https://github.com/b10-tv/speculative-encoding) lets one model read a document and hands another model its KV cache: a small translator maps the source model's per-layer hidden states to the served model's, and the served model's own K/V projections turn them into K/V. `SpeculativeEncodingConnector` serves such translated documents: a request whose prompt starts with a document's tokens loads their KV from a store instead of prefilling them.

The connector runs in the scheduler and in every worker and reads the store directly, so it works with `vllm serve`, streaming, and tensor or pipeline parallelism. Each tensor-parallel rank reads only its own KV heads.

## Publish a document

A producer translates a document for the served model and writes it to a store directory with `write_document` (`vllm.distributed.kv_transfer.kv_connector.v1.speculative_encoding`). Install the `speculative_encoding` package into the vLLM environment without its pinned dependencies, which are older torch and transformers. Until [b10-tv/speculative-encoding#4](https://github.com/b10-tv/speculative-encoding/pull/4) adds support for the torch vLLM uses, install its commit:

```bash
pip install --no-deps "speculative-encoding @ git+https://github.com/devanshi-code18/speculative-encoding@e5856fbfbdb8b847d4c5aa099a45fe06e54d1c75"
python examples/disaggregated/speculative_encoding/publish_document.py \
    --source ministral3-8b --target qwen3-14b \
    --document contract.txt --store /tmp/se-store
```

It prints the document key, the SHA-256 of the stripped, NFC-normalized text. Translation runs in the producer, ahead of the requests that read the document.

## Serve

```bash
vllm serve Qwen/Qwen3-14B --tensor-parallel-size 2 \
    --kv-transfer-config '{"kv_connector": "SpeculativeEncodingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"store": "/tmp/se-store"}}'
```

`kv_connector_extra_config`:

- `store` (required): the store directory, readable by the scheduler and every worker.
- `load_async` (default `true`): load documents in the background. A request waits for its document while the other requests keep running; with `false`, the step that schedules the request loads it and the other requests in that step wait for the load.
- `require_cache_salt` (default `true`): load only for requests that set `cache_salt`. Translated KV is approximate, so give translated requests a salt that only translated requests use: their blocks then never serve requests that prefill the same tokens themselves.

## Request

Put the document text in the system message, name the document in `kv_transfer_params`, and set a salt used only for translated requests:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="Qwen/Qwen3-14B",
    messages=[
        {"role": "system", "content": document_text.strip()},
        {"role": "user", "content": "Who signed the agreement?"},
    ],
    stream=True,
    extra_body={
        "kv_transfer_params": {"speculative_encoding": {"document": key}},
        "cache_salt": f"{tenant_salt}:translated:{key}",
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
```

The scheduler loads the document only when the request's prompt starts with the token ids the document was published for, and the document was published for a model with this model's KV heads and head size. Otherwise, or if the document is missing or unreadable, the request is prefilled normally and the connector logs a warning. The last prompt token is always computed. Later requests with the same `cache_salt` reuse the loaded blocks from the prefix cache. See [query_document.py](../../examples/disaggregated/speculative_encoding/query_document.py) and, for the whole flow on a laptop CPU with tiny random models, [quickstart_cpu.py](../../examples/disaggregated/speculative_encoding/quickstart_cpu.py).

## Store format

One directory per document:

- `<store>/<key>/meta.json`: `format` (`speculative-encoding-kv/1`), `prompt_token_ids` (the `N` tokens the KV covers), `num_layers`, `num_kv_heads`, `head_size`, `dtype`, `provenance`.
- `<store>/<key>/kv.safetensors`: per layer, `layers.<i>` of shape `[2, num_kv_heads, N, head_size]`: post-RoPE K and V of all heads.

`write_document` writes a document to a temporary directory and renames it into place, so readers never see a partial one, and refuses a key that is already published. Treat published documents as immutable: publish a changed translation under a new key. Keys may contain only letters, digits, `_`, `.` and `-`.

## Limitations

- Models with a single KV cache group only: full-attention models, or models whose sliding-window layers run with the hybrid KV cache manager disabled (the default with this connector). MLA, Mamba and gated-delta layers are not supported.
- No speculative decoding with a draft model (EAGLE, MTP, `draft_model` and the like): the store holds only the served model's K/V. `ngram` and `suffix` work.
- If a document's file disappears between scheduling and loading, `kv_load_failure_policy` applies to that request (with `load_async: false`, to the blocks the load would have written). With `load_async: false`, async scheduling and `recompute`, the request whose load failed never finishes and its blocks stay held ([#45138](https://github.com/vllm-project/vllm/issues/45138)): keep `load_async` on or the default policy `fail`.
