# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The translated-document store: one directory per document.

    <store>/<key>/meta.json        format, prompt_token_ids, KV geometry
    <store>/<key>/kv.safetensors   "layers.<i>": [2, num_kv_heads, n, head_size]
                                   per attention layer; for gated delta net
                                   layers "conv_q.<i>", "conv_k.<i>",
                                   "conv_v.<i>" and "recurrent.<i>"

`write_document` writes into a temporary directory and renames it into place,
so readers never see a partial document. Published documents are immutable.
"""

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

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


class GDNState(NamedTuple):
    """A gated delta net layer's state after the document's last token.

    `conv_q`, `conv_k`: `[num_k_heads, head_k_dim, kernel - 1]` and `conv_v`:
    `[num_v_heads, head_v_dim, kernel - 1]`, the last `kernel - 1` inputs of
    the causal convolution (`in_proj_qkv` outputs); `recurrent`:
    `[num_v_heads, head_v_dim, head_k_dim]`.
    """

    conv_q: torch.Tensor
    conv_k: torch.Tensor
    conv_v: torch.Tensor
    recurrent: torch.Tensor


def gdn_state_from_hf(
    conv: torch.Tensor, recurrent: torch.Tensor, num_k_heads: int, head_k_dim: int
) -> GDNState:
    """A `GDNState` from one sequence's state in a transformers cache layer:
    `conv_states` `[conv_dim, kernel]` (q, k and v channels) and
    `recurrent_states` `[num_v_heads, head_k_dim, head_v_dim]`."""
    state = conv[:, 1:]  # vLLM keeps the last kernel - 1 inputs
    q_dim = num_k_heads * head_k_dim
    q, k, v = state.split([q_dim, q_dim, state.shape[0] - 2 * q_dim])
    num_v_heads = recurrent.shape[0]
    return GDNState(
        q.reshape(num_k_heads, head_k_dim, -1),
        k.reshape(num_k_heads, head_k_dim, -1),
        v.reshape(num_v_heads, -1, state.shape[1]),
        recurrent.transpose(-1, -2),
    )


@dataclass(frozen=True)
class StoredDocument:
    key: str
    prompt_token_ids: tuple[int, ...]
    num_layers: int
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    provenance: str
    # Attention layers stored, and the first token each holds (sliding window
    # layers may store only the window).
    kv_layers: tuple[int, ...] = ()
    kv_first: Mapping[int, int] = field(default_factory=dict)
    state_layers: tuple[int, ...] = ()

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids)


def write_document(
    store: str | os.PathLike,
    key: str,
    prompt_token_ids: Sequence[int],
    kv: torch.Tensor | Sequence[torch.Tensor] | Mapping[int, torch.Tensor],
    provenance: str = "",
    *,
    kv_first: Mapping[int, int] | None = None,
    states: Mapping[int, GDNState] | None = None,
) -> Path:
    """Publish a translated document into `store`.

    Args:
        store: The store directory (`kv_connector_extra_config["store"]`).
        key: The document key requests name.
        prompt_token_ids: The `N` prompt tokens the KV covers, in the served
            model's vocabulary: what a request's prompt must start with.
        kv: Per attention layer, post-RoPE K and V of those tokens, `[2, n,
            num_kv_heads, head_size]` (all heads): a sequence for layers
            `0..L-1`, a mapping from layer index, or one `[L, 2, N,
            num_kv_heads, head_size]` tensor.
        provenance: Where the KV came from (e.g. the translator fingerprint).
        kv_first: Per layer, the first token its K/V holds (default 0); a
            layer with `kv_first[i] = f` holds `n = N - f` tokens.
        states: Per gated delta net layer, its state after all `N` tokens.

    Returns:
        The document's directory.

    Raises:
        FileExistsError: `key` is already published.

    """
    if not _KEY.fullmatch(key):
        raise ValueError(f"invalid document key {key!r}")
    if isinstance(kv, Mapping):
        layers = dict(kv)
    else:
        layers = dict(enumerate(kv.unbind(0) if isinstance(kv, torch.Tensor) else kv))
    kv_first = dict(kv_first or {})
    states = dict(states or {})
    num_tokens = len(prompt_token_ids)
    first_layer = next(iter(layers.values()))
    _, _, heads, head_size = first_layer.shape
    for i, layer in layers.items():
        if layer.shape[1] != num_tokens - kv_first.get(i, 0):
            raise ValueError(
                f"layer {i} KV holds {layer.shape[1]} tokens, prompt_token_ids "
                f"{num_tokens} from token {kv_first.get(i, 0)}"
            )
    dtype = str(first_layer.dtype).removeprefix("torch.")
    if dtype not in DTYPES:
        raise ValueError(f"KV dtype {dtype} is not one of {list(DTYPES)}")
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
            for i, layer in layers.items()
        }
        for i, state in states.items():
            for name, tensor in state._asdict().items():
                tensors[f"{name}.{i}"] = tensor.contiguous().cpu()
        save_file(tensors, tmp / "kv.safetensors")
        meta = {
            "format": FORMAT,
            "prompt_token_ids": [int(t) for t in prompt_token_ids],
            "num_layers": len(layers),
            "num_kv_heads": heads,
            "head_size": head_size,
            "dtype": dtype,
            "provenance": provenance,
            "kv_layers": sorted(layers),
            "kv_first": {str(i): f for i, f in sorted(kv_first.items()) if f},
            "state_layers": sorted(states),
        }
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
    if not (folder / "kv.safetensors").is_file():
        raise FileNotFoundError(folder / "kv.safetensors")
    try:
        if meta["format"] != FORMAT:
            raise ValueError(f"unknown format {meta['format']!r}")
        num_layers = int(meta["num_layers"])
        return StoredDocument(
            key=key,
            prompt_token_ids=tuple(int(t) for t in meta["prompt_token_ids"]),
            num_layers=num_layers,
            num_kv_heads=int(meta["num_kv_heads"]),
            head_size=int(meta["head_size"]),
            dtype=DTYPES[meta["dtype"]],
            provenance=str(meta.get("provenance", "")),
            kv_layers=tuple(int(i) for i in meta.get("kv_layers", range(num_layers))),
            kv_first={int(i): int(f) for i, f in meta.get("kv_first", {}).items()},
            state_layers=tuple(int(i) for i in meta.get("state_layers", ())),
        )
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"{folder / 'meta.json'}: missing or invalid {e}") from e
