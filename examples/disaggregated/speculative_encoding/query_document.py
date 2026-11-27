# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ask a question about a translated document served by `vllm serve`.

    vllm serve Qwen/Qwen3-14B --tensor-parallel-size 2 --kv-transfer-config \
        '{"kv_connector": "SpeculativeEncodingConnector", "kv_role": "kv_both",
          "kv_connector_extra_config": {"store": "/tmp/se-store"}}'
    python query_document.py --document contract.txt --question "Who signed?"

The system message holds the document text; the server loads its KV from the
store instead of prefilling it. `--native` sends the same request without the
document reference, for comparison.
"""

import argparse
import hashlib
import time
import unicodedata
from pathlib import Path

from openai import OpenAI


def document_key(text: str) -> str:
    """speculative_encoding's key: sha256 of the NFC-normalized, stripped text."""
    text = unicodedata.normalize("NFC", text.strip())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--document", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--native", action="store_true")
    parser.add_argument(
        "--tenant-salt", default="", help="the client's own cache_salt, if any"
    )
    args = parser.parse_args()

    text = Path(args.document).read_text()
    key = document_key(text)
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    model = client.models.list().data[0].id
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    if args.tenant_salt:
        extra_body["cache_salt"] = args.tenant_salt
    if not args.native:
        extra_body["kv_transfer_params"] = {"speculative_encoding": {"document": key}}
    started = time.perf_counter()
    first = None
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": text.strip()},
            {"role": "user", "content": args.question},
        ],
        max_tokens=args.max_tokens,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body=extra_body,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first is None:
                first = time.perf_counter() - started
            print(chunk.choices[0].delta.content, end="", flush=True)
        if chunk.usage is not None:
            usage = chunk.usage
    print()
    details = usage.prompt_tokens_details
    print(
        f"ttft {first:.3f} s, prompt tokens {usage.prompt_tokens}, "
        f"cached {details.cached_tokens if details else None}"
    )


if __name__ == "__main__":
    main()
