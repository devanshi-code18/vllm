# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The translated-document store: one directory per document.

    <store>/<key>/meta.json        format, prompt_token_ids, KV geometry
    <store>/<key>/kv.safetensors   "layers.<i>": [2, num_kv_heads, N, head_size]

or, for a document shared from a producer's GPU (`shared.share_document`),
meta.json with the CUDA IPC handle of its K/V and no file.

`write_document` writes into a temporary directory and renames it into place,
so readers never see a partial document. Published documents are immutable.
"""

import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import regex as re
import torch
from safetensors.torch import save_file

FORMAT = "speculative-encoding-kv/1"
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
# Keys name directories: one path component, no leading dot.
_KEY = re.compile(r"[0-9A-Za-z][0-9A-Za-z_.-]{0,127}")


@dataclass(frozen=True)
class StoredDocument:
    key: str
    prompt_token_ids: tuple[int, ...]
    num_layers: int
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    provenance: str
    # Per publish: tells a republished document from the one scheduled.
    nonce: str = ""
    # How to map the K/V on a producer's GPU (shared documents only).
    ipc: dict | None = None

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids)


def write_document(
    store: str | os.PathLike,
    key: str,
    prompt_token_ids: Sequence[int],
    kv: torch.Tensor | Sequence[torch.Tensor],
    provenance: str = "",
) -> Path:
    """Publish a translated document into `store`.

    Args:
        store: The store directory (`kv_connector_extra_config["store"]`).
        key: The document key requests name.
        prompt_token_ids: The `N` prompt tokens the KV covers, in the served
            model's vocabulary: what a request's prompt must start with.
        kv: Per layer, post-RoPE K and V of those tokens, `[2, N, num_kv_heads,
            head_size]` (all heads), or one `[num_layers, 2, N, num_kv_heads,
            head_size]` tensor.
        provenance: Where the KV came from (e.g. the translator fingerprint).

    Returns:
        The document's directory.

    Raises:
        FileExistsError: `key` is already published.

    """
    layers = list(kv.unbind(0)) if isinstance(kv, torch.Tensor) else list(kv)
    meta = check_document(key, prompt_token_ids, layers, provenance)
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    final = store / key
    if final.exists():
        raise FileExistsError(
            f"{final} exists: publish a changed document under a new key"
        )
    tmp = Path(tempfile.mkdtemp(dir=store, prefix=f".{key}."))
    try:
        tensors = {
            f"layers.{i}": layer.permute(0, 2, 1, 3).contiguous().cpu()
            for i, layer in enumerate(layers)
        }
        save_file(tensors, tmp / "kv.safetensors")
        (tmp / "meta.json").write_text(json.dumps(meta))
        # Fails, rather than replaces, if another writer published `key` first.
        os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def check_document(
    key: str,
    prompt_token_ids: Sequence[int],
    layers: list[torch.Tensor],
    provenance: str,
) -> dict:
    """The meta.json of a document to publish, after checking its K/V."""
    if not _KEY.fullmatch(key):
        raise ValueError(f"invalid document key {key!r}")
    num_tokens = len(prompt_token_ids)
    _, tokens, heads, head_size = layers[0].shape
    if tokens != num_tokens:
        raise ValueError(f"KV holds {tokens} tokens, prompt_token_ids {num_tokens}")
    if any(layer.shape != layers[0].shape for layer in layers):
        raise ValueError("every layer's KV must have the same shape")
    dtype = str(layers[0].dtype).removeprefix("torch.")
    if dtype not in DTYPES:
        raise ValueError(f"KV dtype {dtype} is not one of {list(DTYPES)}")
    return {
        "format": FORMAT,
        "prompt_token_ids": [int(t) for t in prompt_token_ids],
        "num_layers": len(layers),
        "num_kv_heads": heads,
        "head_size": head_size,
        "dtype": dtype,
        "provenance": provenance,
        "nonce": uuid.uuid4().hex,
    }


def read_document(store: Path, key: str) -> StoredDocument:
    """The metadata of a published document.

    Raises:
        FileNotFoundError: No document `key` in `store`.
        OSError: The document could not be read.
        ValueError: `key` or its directory is not a valid document.

    """
    if not _KEY.fullmatch(key):
        raise ValueError(f"invalid document key {key!r}")
    folder = store / key
    with open(folder / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    ipc = meta.get("ipc")
    shared = isinstance(ipc, dict)
    if not shared and not (folder / "kv.safetensors").is_file():
        raise FileNotFoundError(folder / "kv.safetensors")
    try:
        if meta["format"] != FORMAT:
            raise ValueError(f"unknown format {meta['format']!r}")
        return StoredDocument(
            key=key,
            prompt_token_ids=tuple(int(t) for t in meta["prompt_token_ids"]),
            num_layers=int(meta["num_layers"]),
            num_kv_heads=int(meta["num_kv_heads"]),
            head_size=int(meta["head_size"]),
            dtype=DTYPES[meta["dtype"]],
            provenance=str(meta.get("provenance", "")),
            nonce=str(meta.get("nonce", "")),
            ipc=ipc if shared else None,
        )
    except (KeyError, TypeError) as e:
        raise ValueError(f"{folder / 'meta.json'}: missing or invalid {e}") from e
