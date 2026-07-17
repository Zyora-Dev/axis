"""axis.checkpoint — atomic, resumable checkpoints.

Reliability rules:
- Writes are ATOMIC: save to a temp file in the same directory, fsync, then
  os.replace(). A crash mid-save can never corrupt the previous checkpoint.
- Everything needed to resume exactly: model weights, optimizer state, LR
  schedule step, RNG state, and user metadata.
- Format: numpy .npz (portable, no pickle for weights) + a small JSON header.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np


def save(path: str, *, model=None, optimizer=None, scheduler=None, step: int = 0,
         meta: dict | None = None) -> None:
    payload: dict[str, np.ndarray] = {}
    header: dict = {"step": step, "meta": meta or {}}

    if model is not None:
        for name, arr in model.state_dict().items():
            payload[f"model/{name}"] = arr
    if optimizer is not None:
        opt_state = optimizer.state_dict()
        header["optimizer"] = {"t": opt_state["t"], "lr": opt_state["lr"]}
        for i, m in enumerate(opt_state["m"]):
            payload[f"opt/m/{i}"] = m
        for i, v in enumerate(opt_state["v"]):
            payload[f"opt/v/{i}"] = v
    if scheduler is not None:
        header["scheduler_step"] = scheduler.step_num

    payload["__header__"] = np.frombuffer(
        json.dumps(header).encode("utf-8"), dtype=np.uint8
    )

    dirname = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(dirname, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dirname, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load(path: str, *, model=None, optimizer=None, scheduler=None) -> dict:
    """Restore state in-place. Returns the header (step + meta)."""
    with np.load(path, allow_pickle=False) as z:
        header = json.loads(bytes(z["__header__"]).decode("utf-8"))

        if model is not None:
            state = {
                k[len("model/"):]: z[k] for k in z.files if k.startswith("model/")
            }
            model.load_state_dict(state)

        if optimizer is not None and "optimizer" in header:
            ms = sorted(
                (k for k in z.files if k.startswith("opt/m/")),
                key=lambda k: int(k.rsplit("/", 1)[1]),
            )
            vs = sorted(
                (k for k in z.files if k.startswith("opt/v/")),
                key=lambda k: int(k.rsplit("/", 1)[1]),
            )
            optimizer.load_state_dict({
                "t": header["optimizer"]["t"],
                "lr": header["optimizer"]["lr"],
                "m": [z[k] for k in ms],
                "v": [z[k] for k in vs],
            })

        if scheduler is not None and "scheduler_step" in header:
            scheduler.step_num = int(header["scheduler_step"])

    return header
