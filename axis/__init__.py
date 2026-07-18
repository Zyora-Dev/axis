"""Axis — a reliable framework for training and fine-tuning transformers.

By Zyora Labs. Phase 1: pure-NumPy reference engine with full autograd.
Every op is gradient-checked; the NumPy engine is the ground truth that the
locomp GPU backend (Phase 2) must match bit-for-bit.
"""
from axis.tensor import Tensor, tensor, zeros, ones, randn, arange, manual_seed, no_grad
from axis import ops
from axis import nn
from axis import optim
from axis import accel
from axis.checkpoint import save, load
from axis.gradcheck import gradcheck
from axis.pretrained import from_pretrained, read_safetensors
from axis.data import ByteTokenizer, LMDataset, DataLoader
from axis import lora
from axis.generate import generate
from axis.tokenizer import HFTokenizer

__version__ = "0.13.0"

__all__ = [
    "Tensor", "tensor", "zeros", "ones", "randn", "arange",
    "manual_seed", "no_grad",
    "ops", "nn", "optim", "accel",
    "save", "load", "gradcheck",
    "from_pretrained", "read_safetensors",
    "ByteTokenizer", "LMDataset", "DataLoader",
    "lora", "generate",
    "HFTokenizer",
    "__version__",
]
