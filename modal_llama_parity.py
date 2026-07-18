"""Definitive validation: load the SAME real HuggingFace Llama checkpoint in
both HuggingFace transformers and Axis, and confirm the logits match. This
proves axis.from_pretrained loads and runs real Llama-family models correctly
(weight mapping, transposes, RoPE, GQA, RMSNorm, SwiGLU all consistent).

    modal run modal_llama_parity.py
"""
import pathlib
import modal

REPO = pathlib.Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.24", "torch", "transformers", "huggingface_hub", "safetensors", "regex")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-llama-parity")

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@app.function(image=image, timeout=900)
def parity():
    import sys
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, "/root")
    import axis

    path = snapshot_download(MODEL)
    ids = [1, 5, 9, 2, 7, 3]

    # HuggingFace reference
    hf = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        hf_logits = hf(torch.tensor([ids])).logits[0].float().numpy()

    # Axis
    model = axis.from_pretrained(path)
    ax_logits = model.forward(axis.Tensor(np.array([ids], dtype=np.int64))).data[0]

    print("hf   logits:", hf_logits.shape, flush=True)
    print("axis logits:", ax_logits.shape, flush=True)
    diff = np.abs(hf_logits - ax_logits)
    print(f"max|Δ| = {diff.max():.4e}  mean|Δ| = {diff.mean():.4e}", flush=True)
    # same argmax per position = same predictions
    same_pred = int((hf_logits.argmax(-1) == ax_logits.argmax(-1)).all())
    print(f"argmax match all positions: {bool(same_pred)}", flush=True)
    ok = np.allclose(hf_logits, ax_logits, rtol=1e-3, atol=1e-3)
    print(f"LOGITS MATCH (rtol/atol 1e-3): {ok}", flush=True)


@app.local_entrypoint()
def main():
    parity.remote()
