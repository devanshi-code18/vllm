# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate a document into a target model's KV cache with speculative
encoding, and publish it to a store that `SpeculativeEncodingConnector` serves.

Needs a GPU and the `speculative_encoding` package in the vLLM environment,
without its pinned dependencies (it pins older torch and transformers). Until
b10-tv/speculative-encoding#4 adds support for the torch vLLM uses, install
its commit:

    pip install --no-deps "speculative-encoding @ git+https://github.com/devanshi-code18/speculative-encoding@e5856fbfbdb8b847d4c5aa099a45fe06e54d1c75"
    python publish_document.py --source ministral3-8b --target qwen3-14b \
        --document contract.txt --store /tmp/se-store

Prints the document key; requests name it in `kv_transfer_params` and put the
document text in the system message (see `query_document.py`).
"""

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

from speculative_encoding import checkpoints
from speculative_encoding import encode as E
from speculative_encoding.api import SpeculativeEncoder

from vllm.distributed.kv_transfer.kv_connector.v1.speculative_encoding import (
    write_document,
)


def document_key(text: str) -> str:
    """SHA-256 of the NFC-normalized, stripped text (speculative_encoding's key)."""
    text = unicodedata.normalize("NFC", text.strip())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True, help="catalog key of the reader")
    parser.add_argument(
        "--target", required=True, help="catalog key of the served model"
    )
    parser.add_argument("--document", required=True, help="path to a text file")
    parser.add_argument("--store", required=True)
    args = parser.parse_args()

    text = Path(args.document).read_text()
    found = checkpoints.resolve(args.source, args.target)
    encoder = SpeculativeEncoder.from_pretrained(
        found.checkpoint, source=args.source, target=args.target
    )
    if encoder.hybrid:
        raise SystemExit(f"{args.target} is not a full-attention model")
    encoded = encoder.encode(text).full
    key = document_key(text)
    fingerprint = E.translator_fingerprint(encoder.translator, args.source)
    write_document(
        args.store,
        key,
        encoded.prefix_ids,
        encoded.prefix,
        provenance=f"{args.source}->{args.target}:{fingerprint}",
    )
    print(json.dumps({"document": key, "num_tokens": len(encoded.prefix_ids)}))


if __name__ == "__main__":
    main()
