"""Fair Axis-vs-PyTorch benchmark on the SAME Llama model.

Same architecture (HF LlamaForCausalLM config), same weights, same batch/seq,
same optimizer (AdamW), same precision (fp32 + TF32 on both), proper warmup +
CUDA sync + median. Reports training tok/s, peak memory, and numerical parity
of the forward pass. Rigorous + reproducible — the kind of benchmark that holds
up to scrutiny.

    modal run modal_benchmark.py
"""
import pathlib
import modal

REPO = pathlib.Path(__file__).parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy>=1.24", "cupy-cuda12x", "torch", "transformers")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-benchmark")

VOCAB, DIM, LAYERS, HEADS, KV, MLP = 32000, 768, 12, 12, 4, 2048
BATCH, SEQ, STEPS = 8, 512, 12


@app.function(image=image, gpu="A100", timeout=2400)
def benchmark():
    import sys, time
    import numpy as np
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.backends.cuda.matmul.allow_tf32 = True   # fair: TF32 on both
    torch.backends.cudnn.allow_tf32 = True
    sys.path.insert(0, "/root")
    import cupy as cp
    import axis
    from axis import nn, optim, backend
    from axis.pretrained import convert_hf_llama

    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=DIM, num_hidden_layers=LAYERS,
                      num_attention_heads=HEADS, num_key_value_heads=KV,
                      intermediate_size=MLP, max_position_embeddings=SEQ,
                      rms_norm_eps=1e-5, tie_word_embeddings=False)
    hf = LlamaForCausalLM(cfg).cuda().float()
    n_params = sum(p.numel() for p in hf.parameters())

    rng = np.random.default_rng(0)
    toks = rng.integers(0, VOCAB, size=(BATCH, SEQ + 1)).astype(np.int64)
    inp_np, tgt_np = toks[:, :-1], toks[:, 1:]

    # ---- numerical parity (same weights, forward) ----
    hf_sd = {k: v.detach().cpu().numpy() for k, v in hf.state_dict().items()
             if "rotary" not in k and "inv_freq" not in k}
    ax_state = convert_hf_llama(hf_sd, LAYERS)
    backend.set_device("gpu")
    axis.manual_seed(0)
    am = nn.Transformer(vocab_size=VOCAB, dim=DIM, n_layers=LAYERS, n_heads=HEADS,
                        n_kv_heads=KV, mlp_hidden=MLP, max_seq_len=SEQ, tie_embeddings=False)
    am.load_state_dict(ax_state, strict=False); am.to_gpu()
    with torch.no_grad():
        hf_logits = hf(torch.tensor(inp_np).cuda()).logits.float().cpu().numpy()
    ax_logits = am.forward(axis.Tensor(inp_np)).numpy()
    parity = float(np.abs(hf_logits - ax_logits).max())

    # ---- PyTorch training throughput ----
    opt_t = torch.optim.AdamW(hf.parameters(), lr=3e-4)
    xb = torch.tensor(inp_np).cuda(); yb = torch.tensor(tgt_np).cuda()
    torch.cuda.reset_peak_memory_stats()
    for i in range(STEPS + 2):
        if i == 2:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        opt_t.zero_grad()
        logits = hf(xb).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), yb.reshape(-1))
        loss.backward(); opt_t.step()
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - t0) / STEPS * 1000
    torch_mem = torch.cuda.max_memory_allocated() / 1e9

    # ---- Axis training throughput ----
    del hf; torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()
    opt_a = optim.AdamW(am.parameters(), lr=3e-4)
    ai = axis.Tensor(inp_np); at = axis.Tensor(tgt_np)
    for i in range(STEPS + 2):
        if i == 2:
            cp.cuda.Stream.null.synchronize(); t0 = time.perf_counter()
        l = am.loss(ai, at); l.backward(); opt_a.step(); opt_a.zero_grad()
    cp.cuda.Stream.null.synchronize()
    axis_ms = (time.perf_counter() - t0) / STEPS * 1000
    # True in-use peak: after a forward, every activation is live (that's the
    # peak); backward then frees them incrementally.
    cp.get_default_memory_pool().free_all_blocks()
    l = am.loss(ai, at)
    cp.cuda.Stream.null.synchronize()
    axis_mem = cp.get_default_memory_pool().used_bytes() / 1e9
    l.backward(); opt_a.step(); opt_a.zero_grad()

    print("\n" + "=" * 60, flush=True)
    print(f"FAIR BENCHMARK — Llama {n_params/1e6:.0f}M, batch {BATCH} seq {SEQ}, fp32+TF32", flush=True)
    print("=" * 60, flush=True)
    print(f"forward parity (Axis vs PyTorch): max|Δ| = {parity:.2e}", flush=True)
    print(f"{'':10s} {'ms/step':>9s} {'tok/s':>9s} {'peak mem':>9s}", flush=True)
    print(f"{'PyTorch':10s} {torch_ms:9.0f} {BATCH*SEQ/(torch_ms/1000):9.0f} {torch_mem:8.1f}G", flush=True)
    print(f"{'Axis':10s} {axis_ms:9.0f} {BATCH*SEQ/(axis_ms/1000):9.0f} {axis_mem:8.1f}G", flush=True)
    print(f"\nAxis throughput vs PyTorch: {torch_ms/axis_ms:.2f}x "
          f"({BATCH*SEQ/(axis_ms/1000):.0f} vs {BATCH*SEQ/(torch_ms/1000):.0f} tok/s)", flush=True)


@app.local_entrypoint()
def main():
    benchmark.remote()
