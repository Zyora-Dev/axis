"""Round-trip test for pretrained loading: build an Axis model, write its
weights in HuggingFace-Llama layout (safetensors + config.json), load it back
with from_pretrained, and confirm the logits match exactly. Validates the
safetensors reader, the HF->Axis key map, the Linear transposes and RoPE — with
no network / no torch / no safetensors library.
"""
import json
import struct

import numpy as np

import axis
from axis import nn
from axis.tensor import Tensor


def _write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        b = arr.tobytes()
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(b)]}
        blobs.append(b)
        offset += len(b)
    hjson = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for b in blobs:
            f.write(b)


def _axis_to_hf(model, n_layers):
    """Inverse of convert_hf_llama — for producing a fake HF checkpoint."""
    p = dict(model.named_parameters())
    hf = {
        "model.embed_tokens.weight": p["embed.weight"].data,
        "model.norm.weight": p["norm.weight"].data,
        "lm_head.weight": p["lm_head.weight"].data.T,
    }
    for i in range(n_layers):
        hf[f"model.layers.{i}.input_layernorm.weight"] = p[f"blocks.{i}.attn_norm.weight"].data
        hf[f"model.layers.{i}.post_attention_layernorm.weight"] = p[f"blocks.{i}.mlp_norm.weight"].data
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            hf[f"model.layers.{i}.self_attn.{proj}.weight"] = p[f"blocks.{i}.attn.{proj}.weight"].data.T
        for proj in ("gate_proj", "up_proj", "down_proj"):
            hf[f"model.layers.{i}.mlp.{proj}.weight"] = p[f"blocks.{i}.mlp.{proj}.weight"].data.T
    return hf


def test_from_pretrained_roundtrip(tmp_path):
    axis.manual_seed(0)
    cfg = dict(vocab_size=64, hidden_size=32, num_hidden_layers=2,
               num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
               max_position_embeddings=128, rope_theta=10000.0, rms_norm_eps=1e-5,
               tie_word_embeddings=False)
    ref = nn.Transformer(vocab_size=64, dim=32, n_layers=2, n_heads=4,
                         n_kv_heads=2, mlp_hidden=64, max_seq_len=128,
                         tie_embeddings=False)

    # write fake HF checkpoint
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    _write_safetensors(str(tmp_path / "model.safetensors"),
                       _axis_to_hf(ref, cfg["num_hidden_layers"]))

    loaded = axis.from_pretrained(str(tmp_path))

    toks = Tensor(np.array([[1, 5, 9, 2, 7]], dtype=np.int64))
    a = ref.forward(toks).data
    b = loaded.forward(toks).data
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=1e-5, atol=1e-5), \
        f"logits differ: max|Δ|={np.abs(a - b).max()}"


def test_bf16_reader(tmp_path):
    """bf16 tensors decode to the right float32 values."""
    import json as _json
    vals = np.array([1.0, -2.5, 0.0, 3.14159], dtype=np.float32)
    bf16_bits = (vals.view(np.uint32) >> 16).astype(np.uint16)  # truncate to bf16
    blob = bf16_bits.tobytes()
    header = {"t": {"dtype": "BF16", "shape": [4], "data_offsets": [0, len(blob)]}}
    hjson = _json.dumps(header).encode()
    path = tmp_path / "bf16.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson))); f.write(hjson); f.write(blob)
    got = axis.read_safetensors(str(path))["t"]
    # bf16 truncation error is < 1% relative
    assert np.allclose(got, vals, rtol=1e-2, atol=1e-2)
