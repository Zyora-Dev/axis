"""Confirm Axis == PyTorch at 125M in PURE fp32 (TF32 off) — rules out a real
correctness bug at scale (the benchmark's TF32 parity is precision-limited).

    modal run modal_parity_big.py
"""
import pathlib
import modal

REPO = pathlib.Path(__file__).parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "torch", "transformers")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-parity-big")
VOCAB, DIM, LAYERS, HEADS, KV, MLP, SEQ = 32000, 768, 12, 12, 4, 2048, 128


@app.function(image=image, gpu="A100", timeout=1200)
def parity():
    import os
    os.environ["AXIS_TF32"] = "0"          # Axis pure fp32
    import sys
    import numpy as np
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False   # PyTorch pure fp32
    torch.backends.cudnn.allow_tf32 = False
    from transformers import LlamaConfig, LlamaForCausalLM
    sys.path.insert(0, "/root")
    import axis
    from axis import nn, backend
    from axis.pretrained import convert_hf_llama

    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=DIM, num_hidden_layers=LAYERS,
                      num_attention_heads=HEADS, num_key_value_heads=KV,
                      intermediate_size=MLP, max_position_embeddings=SEQ,
                      rms_norm_eps=1e-5, tie_word_embeddings=False)
    hf = LlamaForCausalLM(cfg).eval().float()
    ids = [[1, 5, 9, 2, 7, 3, 8, 4, 11, 6]]
    with torch.no_grad():
        hf_logits = hf(torch.tensor(ids)).logits[0].float().numpy()

    hf_sd = {k: v.detach().cpu().numpy() for k, v in hf.state_dict().items()
             if "rotary" not in k and "inv_freq" not in k}
    state = convert_hf_llama(hf_sd, LAYERS)
    axis.manual_seed(0)
    am = nn.Transformer(vocab_size=VOCAB, dim=DIM, n_layers=LAYERS, n_heads=HEADS,
                        n_kv_heads=KV, mlp_hidden=MLP, max_seq_len=SEQ, tie_embeddings=False)
    am.load_state_dict(state, strict=False)
    ax_logits = am.forward(axis.Tensor(np.array(ids, dtype=np.int64))).numpy()

    d = np.abs(hf_logits - ax_logits)
    print(f"\n125M pure-fp32 parity: max|Δ|={d.max():.3e} mean|Δ|={d.mean():.3e} "
          f"argmax_match={bool((hf_logits.argmax(-1)==ax_logits.argmax(-1)).all())} "
          f"allclose(1e-3)={np.allclose(hf_logits, ax_logits, rtol=1e-3, atol=1e-3)}", flush=True)


@app.local_entrypoint()
def main():
    parity.remote()
