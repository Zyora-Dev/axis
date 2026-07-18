"""Compile + run the C++ engine de-risk probe on A100: can our own cuBLAS
handle + custom kernels be captured into a CUDA graph and replayed?

    modal run engine/modal_probe.py
"""
import pathlib
import modal

HERE = pathlib.Path(__file__).parent
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .add_local_file(str(HERE / "probe_graph.cu"), remote_path="/root/probe_graph.cu")
)
app = modal.App("axis-engine-probe")


@app.function(image=image, gpu="A100", timeout=900)
def run():
    import subprocess
    r = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_80", "/root/probe_graph.cu", "-lcublas", "-o", "/root/probe"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("COMPILE FAILED:\n", r.stderr[-2000:], flush=True)
        return
    print("compile: OK", flush=True)
    r = subprocess.run(["/root/probe"], capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print("RUN FAILED:", r.stderr[-1000:], flush=True)


@app.local_entrypoint()
def main():
    run.remote()
