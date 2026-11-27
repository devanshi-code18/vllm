# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The translated-document store: one directory per document.

    <store>/<key>/meta.json        format, prompt_token_ids, KV geometry
    <store>/<key>/kv.safetensors   "layers.<i>": [2, num_kv_heads, N, head_size]

or, with the layer-inputs payload (the served model writes its own K/V):

    <store>/<key>/meta.json                  format, prompt_token_ids, geometry
    <store>/<key>/layer_inputs.safetensors   "layers.<i>": [N, hidden_size]

`write_document` writes into a temporary directory and renames it into place,
so readers never see a partial document. Published documents are immutable.
"""

import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import regex as re
import torch
from safetensors.torch import save_file

FORMAT = "speculative-encoding-kv/1"
LAYER_INPUTS_FORMAT = "speculative-encoding-layer-inputs/1"
PAYLOAD_FILES = {
    FORMAT: "kv.safetensors",
    LAYER_INPUTS_FORMAT: "layer_inputs.safetensors",
}
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
    format: str = FORMAT
    hidden_size: int = 0

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def layer_inputs(self) -> bool:
        return self.format == LAYER_INPUTS_FORMAT

    @property
    def file(self) -> str:
        return PAYLOAD_FILES[self.format]


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
    if not _KEY.fullmatch(key):
        raise ValueError(f"invalid document key {key!r}")
    layers = list(kv.unbind(0)) if isinstance(kv, torch.Tensor) else list(kv)
    num_tokens = len(prompt_token_ids)
    _, tokens, heads, head_size = layers[0].shape
    if tokens != num_tokens:
        raise ValueError(f"KV holds {tokens} tokens, prompt_token_ids {num_tokens}")
    dtype = str(layers[0].dtype).removeprefix("torch.")
    if dtype not in DTYPES:
        raise ValueError(f"KV dtype {dtype} is not one of {list(DTYPES)}")
    tensors = {
        f"layers.{i}": layer.permute(0, 2, 1, 3).contiguous().cpu()
        for i, layer in enumerate(layers)
    }
    meta = {
        "format": FORMAT,
        "prompt_token_ids": [int(t) for t in prompt_token_ids],
        "num_layers": len(layers),
        "num_kv_heads": heads,
        "head_size": head_size,
        "dtype": dtype,
        "provenance": provenance,
    }
    return _publish(Path(store), key, tensors, meta)


def write_document_layer_inputs(
    store: str | os.PathLike,
    key: str,
    prompt_token_ids: Sequence[int],
    layer_inputs: torch.Tensor | Sequence[torch.Tensor],
    provenance: str = "",
) -> Path:
    """Publish a translated document as per-layer inputs: the served model
    computes K/V from them with its own weights (quantized ones included),
    through `KVConnectorBase_V1.register_kv_from_layer_inputs_writer`.

    Args:
        store: The store directory.
        key: The document key requests name.
        prompt_token_ids: The `N` prompt tokens the inputs cover.
        layer_inputs: Per decoder layer, its `[N, hidden_size]` input (layer
            0: the input embeddings), or one `[num_layers, N, hidden_size]`
            tensor.
        provenance: Where the inputs came from.

    Returns:
        The document's directory.

    """
    if not _KEY.fullmatch(key):
        raise ValueError(f"invalid document key {key!r}")
    if isinstance(layer_inputs, torch.Tensor):
        layers = list(layer_inputs.unbind(0))
    else:
        layers = list(layer_inputs)
    tokens, hidden = layers[0].shape
    if tokens != len(prompt_token_ids):
        raise ValueError(
            f"inputs hold {tokens} tokens, prompt_token_ids {len(prompt_token_ids)}"
        )
    dtype = str(layers[0].dtype).removeprefix("torch.")
    if dtype not in DTYPES:
        raise ValueError(f"dtype {dtype} is not one of {list(DTYPES)}")
    tensors = {f"layers.{i}": x.contiguous().cpu() for i, x in enumerate(layers)}
    meta = {
        "format": LAYER_INPUTS_FORMAT,
        "prompt_token_ids": [int(t) for t in prompt_token_ids],
        "num_layers": len(layers),
        "hidden_size": hidden,
        "dtype": dtype,
        "provenance": provenance,
    }
    return _publish(Path(store), key, tensors, meta)


def _publish(
    store: Path, key: str, tensors: dict[str, torch.Tensor], meta: dict
) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    final = store / key
    if final.exists():
        raise FileExistsError(
            f"{final} exists: publish a changed document under a new key"
        )
    tmp = Path(tempfile.mkdtemp(dir=store, prefix=f".{key}."))
    try:
        save_file(tensors, tmp / PAYLOAD_FILES[meta["format"]])
        (tmp / "meta.json").write_text(json.dumps(meta))
        # Fails, rather than replaces, if another writer published `key` first.
        os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


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
    file = PAYLOAD_FILES.get(meta.get("format"))
    if file is None:
        raise ValueError(f"unknown format {meta.get('format')!r}")
    if not (folder / file).is_file():
        raise FileNotFoundError(folder / file)
    try:
        layer_inputs = meta["format"] == LAYER_INPUTS_FORMAT
        return StoredDocument(
            key=key,
            prompt_token_ids=tuple(int(t) for t in meta["prompt_token_ids"]),
            num_layers=int(meta["num_layers"]),
            num_kv_heads=0 if layer_inputs else int(meta["num_kv_heads"]),
            head_size=0 if layer_inputs else int(meta["head_size"]),
            dtype=DTYPES[meta["dtype"]],
            provenance=str(meta.get("provenance", "")),
            format=meta["format"],
            hidden_size=int(meta["hidden_size"]) if layer_inputs else 0,
        )
    except (KeyError, TypeError) as e:
        raise ValueError(f"{folder / 'meta.json'}: missing or invalid {e}") from e
