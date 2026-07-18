"""End-to-end LoRA fine-tune demo on a real tiny Llama: load model + its real
tokenizer, LoRA-adapt, train on a short text, and report the loss curve +
performance (step time, tokens/s). Also shows generation before vs after.

    modal run modal_finetune_demo.py
"""
import pathlib
import modal

REPO = pathlib.Path(__file__).parent
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.24", "huggingface_hub", "safetensors", "regex")
    .add_local_dir(str(REPO / "axis"), remote_path="/root/axis")
)
app = modal.App("axis-finetune-demo")

MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TEXT = (
    "Axis is a reliable framework for training and fine-tuning transformers. "
    "It loads pretrained models, applies LoRA adapters, and runs on NVIDIA and "
    "AMD. Zyora Labs builds Axis to be dependency-light and correct. "
) * 6
STEPS, BATCH, SEQ, LR = 80, 4, 32, 1e-2


@app.function(image=image, timeout=1800)
def finetune():
    import sys, time
    import numpy as np
    from huggingface_hub import snapshot_download
    sys.path.insert(0, "/root")
    import axis
    from axis import optim, lora, HFTokenizer

    path = snapshot_download(MODEL)
    print("loading model + tokenizer ...", flush=True)
    model = axis.from_pretrained(path)
    tok = HFTokenizer.from_pretrained(path)
    print(f"model params: {model.num_parameters()/1e6:.2f}M | vocab {tok.vocab_size}", flush=True)

    ids = tok.encode(TEXT)
    ds = axis.LMDataset(ids, seq_len=SEQ)
    dl = axis.DataLoader(ds, batch_size=BATCH, shuffle=True, seed=0)

    prompt = tok.encode("Axis is")
    before = tok.decode(axis.generate(model, prompt, max_new_tokens=20, temperature=0.0))

    # Full fine-tune (all params) — a random-init tiny model needs its backbone
    # updated to actually learn; LoRA (adapters only) is validated separately.
    trainable = list(model.parameters())
    n_train = sum(p.size for p in trainable)
    print(f"full fine-tune: {n_train/1e6:.2f}M trainable params\n", flush=True)
    opt = optim.AdamW(trainable, lr=LR)

    losses, step_ms = [], []
    step = 0
    print("step   loss     ms/step   tok/s", flush=True)
    while step < STEPS:
        for inp, tgt in dl:
            t0 = time.perf_counter()
            loss = model.loss(inp, tgt)
            loss.backward()
            opt.step(); opt.zero_grad()
            dt = time.perf_counter() - t0
            losses.append(float(loss.data)); step_ms.append(dt * 1000)
            if step % 4 == 0 or step == STEPS - 1:
                print(f"{step:4d}   {losses[-1]:6.3f}   {dt*1000:7.1f}   {BATCH*SEQ/dt:6.0f}", flush=True)
            step += 1
            if step >= STEPS:
                break

    after = tok.decode(axis.generate(model, prompt, max_new_tokens=20, temperature=0.0))

    print("\n=== LOSS CURVE ===", flush=True)
    print(f"first: {losses[0]:.3f}  ->  last: {losses[-1]:.3f}  "
          f"(drop {losses[0]-losses[-1]:.3f}, {100*(losses[0]-losses[-1])/losses[0]:.0f}%)", flush=True)
    # compact ASCII sparkline of the curve (min..max)
    lo, hi = min(losses), max(losses)
    bars = "▁▂▃▄▅▆▇█"
    spark = "".join(bars[min(7, int((l - lo) / (hi - lo + 1e-9) * 7))] for l in losses)
    print("curve:", spark, flush=True)
    print(f"\n=== PERFORMANCE ===", flush=True)
    print(f"avg step: {np.mean(step_ms):.1f} ms | median {np.median(step_ms):.1f} ms | "
          f"{BATCH*SEQ/(np.mean(step_ms)/1000):.0f} tok/s", flush=True)
    print(f"\n=== GENERATION (prompt 'Axis is', greedy) ===", flush=True)
    print(f"before FT: {before!r}", flush=True)
    print(f"after  FT: {after!r}", flush=True)


@app.local_entrypoint()
def main():
    finetune.remote()
