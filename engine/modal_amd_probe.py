"""Probe: can this Modal account provision AMD GPUs now?"""
import modal

image = modal.Image.from_registry("rocm/dev-ubuntu-22.04:6.2-complete", add_python="3.11")
app = modal.App("axis-amd-probe")


@app.function(image=image, gpu="MI300X", timeout=300)
def probe():
    import subprocess
    r = subprocess.run(["rocm-smi"], capture_output=True, text=True)
    print(r.stdout[-1500:] or r.stderr[-1500:], flush=True)
    r2 = subprocess.run(["hipcc", "--version"], capture_output=True, text=True)
    print(r2.stdout[-400:] or r2.stderr[-400:], flush=True)


@app.local_entrypoint()
def main():
    probe.remote()
