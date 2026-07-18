"""axis.pretrained — load real pretrained weights (HuggingFace safetensors).

This is what turns Axis from "train from scratch" into "fine-tune real models".
Dependency-light on purpose: a pure-Python safetensors reader (no `safetensors`
or `torch` needed), plus the HF-Llama -> Axis key/layout mapping.

Layout notes (must be exact or the model is silently wrong):
- Axis `Linear.weight` is [in, out] (forward does x @ W); HF Linear.weight is
  [out, in] -> every projection weight is TRANSPOSED on load.
- Embedding [vocab, dim] and RMSNorm [dim] weights map directly (no transpose).
- RoPE: Axis uses the half-split `rotate_half` convention, identical to HF
  Llama, so q/k weights need NO permutation.
"""
from __future__ import annotations

import json
import os
import struct
from typing import Dict

import numpy as np

# safetensors dtype string -> (numpy dtype, is_float)
_ST_DTYPES = {
    "F64": (np.float64, True), "F32": (np.float32, True), "F16": (np.float16, True),
    "BF16": (None, True),  # handled specially (no native numpy bf16)
    "I64": (np.int64, False), "I32": (np.int32, False), "I16": (np.int16, False),
    "I8": (np.int8, False), "U8": (np.uint8, False), "BOOL": (np.bool_, False),
}


def _bf16_to_f32(raw: bytes) -> np.ndarray:
    """bfloat16 = the upper 16 bits of a float32. Widen back to float32."""
    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32)


def read_safetensors(path: str) -> Dict[str, np.ndarray]:
    """Parse a single .safetensors file into {name: ndarray}. Float tensors
    (incl. fp16/bf16) are returned as float32; int/bool keep their kind."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len).decode("utf-8"))
        data = f.read()  # the tensor byte buffer (offsets are relative to here)

    out: Dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt = meta["dtype"]
        if dt not in _ST_DTYPES:
            raise ValueError(f"{name}: unsupported safetensors dtype {dt}")
        begin, end = meta["data_offsets"]
        raw = data[begin:end]
        shape = tuple(meta["shape"])
        if dt == "BF16":
            arr = _bf16_to_f32(raw)
        else:
            np_dt, is_float = _ST_DTYPES[dt]
            arr = np.frombuffer(raw, dtype=np_dt)
            if is_float:
                arr = arr.astype(np.float32)
        out[name] = arr.reshape(shape) if arr.size else arr.reshape(shape)
    return out


def _load_all_shards(model_dir: str) -> Dict[str, np.ndarray]:
    """Load a single model.safetensors or all shards from an index."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    single = os.path.join(model_dir, "model.safetensors")
    state: Dict[str, np.ndarray] = {}
    if os.path.exists(index):
        with open(index) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
        for fn in files:
            state.update(read_safetensors(os.path.join(model_dir, fn)))
    elif os.path.exists(single):
        state.update(read_safetensors(single))
    else:
        # any *.safetensors in the dir
        shards = sorted(fn for fn in os.listdir(model_dir) if fn.endswith(".safetensors"))
        if not shards:
            raise FileNotFoundError(f"no .safetensors found in {model_dir}")
        for fn in shards:
            state.update(read_safetensors(os.path.join(model_dir, fn)))
    return state


def convert_hf_llama(hf: Dict[str, np.ndarray], n_layers: int) -> Dict[str, np.ndarray]:
    """Remap a HuggingFace Llama-family state dict to Axis Transformer keys,
    transposing every Linear weight ([out,in] -> [in,out])."""
    def T(w):
        return np.ascontiguousarray(w.T)

    axis: Dict[str, np.ndarray] = {}
    axis["embed.weight"] = hf["model.embed_tokens.weight"].astype(np.float32)
    axis["norm.weight"] = hf["model.norm.weight"].astype(np.float32)
    if "lm_head.weight" in hf:
        axis["lm_head.weight"] = T(hf["lm_head.weight"].astype(np.float32))

    for i in range(n_layers):
        h = f"model.layers.{i}."
        a = f"blocks.{i}."
        axis[a + "attn_norm.weight"] = hf[h + "input_layernorm.weight"].astype(np.float32)
        axis[a + "mlp_norm.weight"] = hf[h + "post_attention_layernorm.weight"].astype(np.float32)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            axis[a + f"attn.{proj}.weight"] = T(hf[h + f"self_attn.{proj}.weight"].astype(np.float32))
        for proj in ("gate_proj", "up_proj", "down_proj"):
            axis[a + f"mlp.{proj}.weight"] = T(hf[h + f"mlp.{proj}.weight"].astype(np.float32))
    return axis


# HF Llama config.json field -> Axis Transformer kwarg
def _config_to_kwargs(cfg: dict) -> dict:
    return dict(
        vocab_size=cfg["vocab_size"],
        dim=cfg["hidden_size"],
        n_layers=cfg["num_hidden_layers"],
        n_heads=cfg["num_attention_heads"],
        n_kv_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        mlp_hidden=cfg["intermediate_size"],
        max_seq_len=cfg.get("max_position_embeddings", 2048),
        rope_theta=cfg.get("rope_theta", 10000.0),
        norm_eps=cfg.get("rms_norm_eps", 1e-5),
        tie_embeddings=cfg.get("tie_word_embeddings", False),
    )


def from_pretrained(model_dir: str):
    """Build an Axis Transformer from a HuggingFace Llama-family checkpoint
    directory (config.json + safetensors) and load its weights.

    Returns the model in eval() mode, ready for inference / fine-tuning.
    """
    from axis.nn import Transformer

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    kwargs = _config_to_kwargs(cfg)

    model = Transformer(**kwargs)
    hf = _load_all_shards(model_dir)
    state = convert_hf_llama(hf, kwargs["n_layers"])
    # Ensure every model parameter is actually covered (strict=False alone would
    # silently leave a param at its random init). Extra HF keys (rotary inv_freq,
    # tied lm_head, ...) are ignored.
    own = dict(model.named_parameters())
    missing = [k for k in own if k not in state]
    if missing:
        raise KeyError(
            f"from_pretrained: no weights for {len(missing)} params, e.g. {missing[:5]}")
    model.load_state_dict(state, strict=False)
    return model.eval()
