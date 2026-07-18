"""axis.compile — compile a Transformer training step to the C++ engine.

Generalizes the validated single-block lowering (engine/modal_step_test.py) to
N blocks, any width/depth: builds ONE execution plan for the complete training
step (embed -> N blocks -> norm -> lm_head -> CE -> full backward -> AdamW),
executed natively (one Python call per step) or as a captured CUDA graph.

Scale-agnostic by construction: the same lowering loop handles any config.
Per-block activations needed by backward are kept; pure scratch is shared
across blocks. The eager Axis engine remains the numerical oracle.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from axis.engine import (Engine, op, GEMM, ADD, RMSNORM, SILU_MUL, SCALE, COPY,
                         GEMM_SB, PERM_0213, ROPE, SOFTMAX_CAUSAL, REPEAT_KV,
                         RMSNORM_BWD, COLSUM, REPEAT_KV_BWD, SOFTMAX_BWD,
                         SILU_BWD, EMBED, EMBED_BWD, CE, ADAMW, TICK)


class CompiledTransformer:
    def __init__(self, lib_path: str, cfg: dict, weights: Dict[str, np.ndarray],
                 batch: int, seq: int, lr: float = 3e-4, wd: float = 0.1,
                 eps: float = 1e-5, tf32: bool = True, recompute_attn: bool = True):
        """recompute_attn=True (default): attention probabilities are NOT saved
        per block — backward recomputes them (flash-attention-style). Removes
        the O(T^2)-per-block activation, enabling long sequences at depth."""
        self.cfg = cfg
        self.B, self.T = batch, seq
        self.lr, self.wd = lr, wd
        V, D = cfg["vocab_size"], cfg["dim"]
        L, H = cfg["n_layers"], cfg["n_heads"]
        KV = cfg.get("n_kv_heads") or H
        DH = D // H
        MLP = cfg["mlp_hidden"]
        tied = cfg.get("tie_embeddings", True)
        theta = cfg.get("rope_theta", 10000.0)
        B, T = batch, seq
        N = B * T
        BH, BKV = B * H, B * KV
        scale = 1.0 / np.sqrt(DH)
        self.N, self.V = N, V

        eng = Engine(lib_path)
        eng.set_tf32(tf32)
        self.eng = eng
        A = eng.alloc

        # rope cache
        half = DH // 2
        freqs = 1.0 / (theta ** (np.arange(half, dtype=np.float32) / half))
        ang = np.outer(np.arange(T, dtype=np.float32), freqs)
        b_cos = eng.new_tensor(np.cos(ang).astype(np.float32))
        b_sin = eng.new_tensor(np.sin(ang).astype(np.float32))

        # params (+grad/m/v)
        self.pnames = ["embed.weight"]
        for i in range(L):
            p = f"blocks.{i}."
            self.pnames += [p + "attn_norm.weight", p + "attn.q_proj.weight",
                            p + "attn.k_proj.weight", p + "attn.v_proj.weight",
                            p + "attn.o_proj.weight", p + "mlp_norm.weight",
                            p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight",
                            p + "mlp.down_proj.weight"]
        self.pnames.append("norm.weight")
        if not tied:
            self.pnames.append("lm_head.weight")
        self.shapes = {nm: weights[nm].shape for nm in self.pnames}
        self.pb, self.gbuf, mb, vb = {}, {}, {}, {}
        for nm in self.pnames:
            w = np.ascontiguousarray(weights[nm], dtype=np.float32)
            self.pb[nm] = eng.new_tensor(w)
            self.gbuf[nm] = A(w.size)
            mb[nm] = eng.new_tensor(np.zeros_like(w))
            vb[nm] = eng.new_tensor(np.zeros_like(w))

        # io buffers
        self.b_ids = A(N)
        self.b_tgt = A(N)
        self.b_loss = A(1)

        # per-block saved activations (needed by backward)
        xs = [A(N * D) for _ in range(L + 1)]        # block inputs (x0 = embed out)
        ys = [A(N * D) for _ in range(L)]
        q2s = [A(N * H * DH) for _ in range(L)]
        krs = [A(N * H * DH) for _ in range(L)]
        vrs = [A(N * H * DH) for _ in range(L)]
        # attention probs: with recompute_attn, ONE shared buffer (recomputed in
        # bwd); otherwise saved per block (O(T^2) each — short seq only).
        pr_shared = A(BH * T * T)
        prs = [pr_shared] * L if recompute_attn else [A(BH * T * T) for _ in range(L)]
        at2s = [A(N * H * DH) for _ in range(L)]
        r1s = [A(N * D) for _ in range(L)]
        zs = [A(N * D) for _ in range(L)]
        gs = [A(N * MLP) for _ in range(L)]
        us = [A(N * MLP) for _ in range(L)]
        hs = [A(N * MLP) for _ in range(L)]

        # shared scratch (reused every block)
        q0 = A(N * H * DH); k0 = A(N * KV * DH); v0 = A(N * KV * DH)
        q1 = A(N * H * DH); k1 = A(N * KV * DH); v1 = A(N * KV * DH)
        k2 = A(N * KV * DH); sc_ = A(BH * T * T)
        at = A(N * H * DH); o_ = A(N * D); mo = A(N * D)
        xf = A(N * D)
        logits = A(N * V); dlogits = A(N * V)
        # backward scratch
        dxf = A(N * D); dcur = A(N * D); tmpD = A(N * D)
        dh_ = A(N * MLP); dg_ = A(N * MLP); du_ = A(N * MLP)
        dz1 = A(N * D); dz2 = A(N * D); dz = A(N * D)
        dr1b = A(N * D); dr1 = A(N * D)
        dat2 = A(N * H * DH); dat = A(N * H * DH)
        dpr = A(BH * T * T); dvr = A(N * H * DH); dsc = A(BH * T * T)
        dq2 = A(N * H * DH); dkr = A(N * H * DH)
        dk2 = A(N * KV * DH); dv1 = A(N * KV * DH)
        dq1 = A(N * H * DH); dk1 = A(N * KV * DH)
        dq0 = A(N * H * DH); dk0 = A(N * KV * DH); dv0 = A(N * KV * DH)
        dy1 = A(N * D); dy2 = A(N * D); dy3 = A(N * D); dy = A(N * D)
        dx1 = A(N * D); dxin = A(N * D)

        P, G = self.pb, self.gbuf
        eps_n = eps
        plan = [op(EMBED, a=P["embed.weight"], b=self.b_ids, c=xs[0], m=N, n=D)]

        # ---- forward blocks ----
        for i in range(L):
            p = f"blocks.{i}."
            plan += [
                op(RMSNORM, a=xs[i], b=P[p + "attn_norm.weight"], c=ys[i], m=N, n=D, alpha=eps_n),
                op(GEMM, a=ys[i], b=P[p + "attn.q_proj.weight"], c=q0, m=N, k=D, n=H * DH),
                op(GEMM, a=ys[i], b=P[p + "attn.k_proj.weight"], c=k0, m=N, k=D, n=KV * DH),
                op(GEMM, a=ys[i], b=P[p + "attn.v_proj.weight"], c=v0, m=N, k=D, n=KV * DH),
                op(PERM_0213, a=q0, c=q1, m=B, n=T, k=H, batch=DH),
                op(PERM_0213, a=k0, c=k1, m=B, n=T, k=KV, batch=DH),
                op(PERM_0213, a=v0, c=v1, m=B, n=T, k=KV, batch=DH),
                op(ROPE, a=q1, b=b_cos, d=b_sin, c=q2s[i], batch=BH, m=T, n=DH),
                op(ROPE, a=k1, b=b_cos, d=b_sin, c=k2, batch=BKV, m=T, n=DH),
                op(REPEAT_KV, a=k2, c=krs[i], batch=B, tb=KV, n=H, m=T, k=DH),
                op(REPEAT_KV, a=v1, c=vrs[i], batch=B, tb=KV, n=H, m=T, k=DH),
                op(GEMM_SB, a=q2s[i], b=krs[i], c=sc_, m=T, k=DH, n=T, batch=BH, tb=1,
                   sa=T * DH, sb=T * DH, sc=T * T, alpha=scale),
                op(SOFTMAX_CAUSAL, a=sc_, c=prs[i], batch=BH, m=T),
                op(GEMM_SB, a=prs[i], b=vrs[i], c=at, m=T, k=T, n=DH, batch=BH,
                   sa=T * T, sb=T * DH, sc=T * DH),
                op(PERM_0213, a=at, c=at2s[i], m=B, n=H, k=T, batch=DH),
                op(GEMM, a=at2s[i], b=P[p + "attn.o_proj.weight"], c=o_, m=N, k=H * DH, n=D),
                op(ADD, a=xs[i], b=o_, c=r1s[i], n=N * D),
                op(RMSNORM, a=r1s[i], b=P[p + "mlp_norm.weight"], c=zs[i], m=N, n=D, alpha=eps_n),
                op(GEMM, a=zs[i], b=P[p + "mlp.gate_proj.weight"], c=gs[i], m=N, k=D, n=MLP),
                op(GEMM, a=zs[i], b=P[p + "mlp.up_proj.weight"], c=us[i], m=N, k=D, n=MLP),
                op(SILU_MUL, a=gs[i], b=us[i], c=hs[i], n=N * MLP),
                op(GEMM, a=hs[i], b=P[p + "mlp.down_proj.weight"], c=mo, m=N, k=MLP, n=D),
                op(ADD, a=r1s[i], b=mo, c=xs[i + 1], n=N * D),
            ]

        # ---- head + CE ----
        lm_w = P["embed.weight"] if tied else P["lm_head.weight"]
        plan += [
            op(RMSNORM, a=xs[L], b=P["norm.weight"], c=xf, m=N, n=D, alpha=eps_n),
            # tied: logits = xf @ E^T (E=[V,D] row-major -> tb=1); untied: xf @ Wlm
            (op(GEMM_SB, a=xf, b=lm_w, c=logits, m=N, k=D, n=V, batch=1, tb=1)
             if tied else
             op(GEMM, a=xf, b=lm_w, c=logits, m=N, k=D, n=V)),
            op(SCALE, a=self.b_loss, c=self.b_loss, n=1, alpha=0.0),
            op(CE, a=logits, b=self.b_tgt, c=dlogits, d=self.b_loss, m=N, n=V),
        ]

        # ---- head backward ----
        if tied:
            plan += [
                op(GEMM_SB, a=dlogits, b=lm_w, c=dxf, m=N, k=V, n=D, batch=1, tb=0),
                # dE(head) = dlogits^T @ xf  -> [V, D]
                op(GEMM_SB, a=dlogits, b=xf, c=G["embed.weight"], m=V, n=D, k=N, batch=1, tb=2),
            ]
        else:
            plan += [
                op(GEMM_SB, a=dlogits, b=lm_w, c=dxf, m=N, k=V, n=D, batch=1, tb=1),
                op(GEMM_SB, a=xf, b=dlogits, c=G["lm_head.weight"], m=D, n=V, k=N, batch=1, tb=2),
            ]
        plan += [
            op(RMSNORM_BWD, a=xs[L], b=P["norm.weight"], d=dxf, c=dcur, tb=tmpD, m=N, n=D, alpha=eps_n),
            op(COLSUM, a=tmpD, c=G["norm.weight"], m=N, n=D),
        ]

        # ---- backward blocks (reverse) ----
        for i in reversed(range(L)):
            p = f"blocks.{i}."
            plan += [
                # x_{i+1} = r1 + mo ; d_mo = dcur
                op(GEMM_SB, a=dcur, b=P[p + "mlp.down_proj.weight"], c=dh_, m=N, k=D, n=MLP, batch=1, tb=1),
                op(GEMM_SB, a=hs[i], b=dcur, c=G[p + "mlp.down_proj.weight"], m=MLP, n=D, k=N, batch=1, tb=2),
                op(SILU_BWD, a=gs[i], b=us[i], d=dh_, c=dg_, tb=du_, n=N * MLP),
                op(GEMM_SB, a=dg_, b=P[p + "mlp.gate_proj.weight"], c=dz1, m=N, k=MLP, n=D, batch=1, tb=1),
                op(GEMM_SB, a=du_, b=P[p + "mlp.up_proj.weight"], c=dz2, m=N, k=MLP, n=D, batch=1, tb=1),
                op(ADD, a=dz1, b=dz2, c=dz, n=N * D),
                op(GEMM_SB, a=zs[i], b=dg_, c=G[p + "mlp.gate_proj.weight"], m=D, n=MLP, k=N, batch=1, tb=2),
                op(GEMM_SB, a=zs[i], b=du_, c=G[p + "mlp.up_proj.weight"], m=D, n=MLP, k=N, batch=1, tb=2),
                op(RMSNORM_BWD, a=r1s[i], b=P[p + "mlp_norm.weight"], d=dz, c=dr1b, tb=tmpD, m=N, n=D, alpha=eps_n),
                op(COLSUM, a=tmpD, c=G[p + "mlp_norm.weight"], m=N, n=D),
                op(ADD, a=dcur, b=dr1b, c=dr1, n=N * D),
                # attention
                op(GEMM_SB, a=dr1, b=P[p + "attn.o_proj.weight"], c=dat2, m=N, k=D, n=H * DH, batch=1, tb=1),
                op(GEMM_SB, a=at2s[i], b=dr1, c=G[p + "attn.o_proj.weight"], m=H * DH, n=D, k=N, batch=1, tb=2),
                op(PERM_0213, a=dat2, c=dat, m=B, n=T, k=H, batch=DH),
            ]
            if recompute_attn:
                # rebuild probs from saved q2/kr (flash-attn-style memory)
                plan += [
                    op(GEMM_SB, a=q2s[i], b=krs[i], c=sc_, m=T, k=DH, n=T, batch=BH, tb=1,
                       sa=T * DH, sb=T * DH, sc=T * T, alpha=scale),
                    op(SOFTMAX_CAUSAL, a=sc_, c=prs[i], batch=BH, m=T),
                ]
            plan += [
                op(GEMM_SB, a=dat, b=vrs[i], c=dpr, m=T, k=DH, n=T, batch=BH, tb=1,
                   sa=T * DH, sb=T * DH, sc=T * T),
                op(GEMM_SB, a=prs[i], b=dat, c=dvr, m=T, n=DH, k=T, batch=BH, tb=2,
                   sa=T * T, sb=T * DH, sc=T * DH),
                op(SOFTMAX_BWD, a=prs[i], b=dpr, c=dsc, batch=BH, m=T),
                op(GEMM_SB, a=dsc, b=krs[i], c=dq2, m=T, k=T, n=DH, batch=BH,
                   sa=T * T, sb=T * DH, sc=T * DH, alpha=scale),
                op(GEMM_SB, a=dsc, b=q2s[i], c=dkr, m=T, n=DH, k=T, batch=BH, tb=2,
                   sa=T * T, sb=T * DH, sc=T * DH, alpha=scale),
                op(REPEAT_KV_BWD, a=dkr, c=dk2, batch=B, tb=KV, n=H, m=T, k=DH),
                op(REPEAT_KV_BWD, a=dvr, c=dv1, batch=B, tb=KV, n=H, m=T, k=DH),
                op(ROPE, a=dq2, b=b_cos, d=b_sin, c=dq1, batch=BH, m=T, n=DH, tb=1),
                op(ROPE, a=dk2, b=b_cos, d=b_sin, c=dk1, batch=BKV, m=T, n=DH, tb=1),
                op(PERM_0213, a=dq1, c=dq0, m=B, n=H, k=T, batch=DH),
                op(PERM_0213, a=dk1, c=dk0, m=B, n=KV, k=T, batch=DH),
                op(PERM_0213, a=dv1, c=dv0, m=B, n=KV, k=T, batch=DH),
                op(GEMM_SB, a=dq0, b=P[p + "attn.q_proj.weight"], c=dy1, m=N, k=H * DH, n=D, batch=1, tb=1),
                op(GEMM_SB, a=dk0, b=P[p + "attn.k_proj.weight"], c=dy2, m=N, k=KV * DH, n=D, batch=1, tb=1),
                op(GEMM_SB, a=dv0, b=P[p + "attn.v_proj.weight"], c=dy3, m=N, k=KV * DH, n=D, batch=1, tb=1),
                op(ADD, a=dy1, b=dy2, c=dy, n=N * D),
                op(ADD, a=dy, b=dy3, c=dy, n=N * D),
                op(GEMM_SB, a=ys[i], b=dq0, c=G[p + "attn.q_proj.weight"], m=D, n=H * DH, k=N, batch=1, tb=2),
                op(GEMM_SB, a=ys[i], b=dk0, c=G[p + "attn.k_proj.weight"], m=D, n=KV * DH, k=N, batch=1, tb=2),
                op(GEMM_SB, a=ys[i], b=dv0, c=G[p + "attn.v_proj.weight"], m=D, n=KV * DH, k=N, batch=1, tb=2),
                op(RMSNORM_BWD, a=xs[i], b=P[p + "attn_norm.weight"], d=dy, c=dx1, tb=tmpD, m=N, n=D, alpha=eps_n),
                op(COLSUM, a=tmpD, c=G[p + "attn_norm.weight"], m=N, n=D),
                op(ADD, a=dr1, b=dx1, c=dxin, n=N * D),
                op(COPY, a=dxin, c=dcur, n=N * D),
            ]
        # embed backward: scatter d_x0 on top of tied head grad (or zeroed buf)
        if not tied:
            plan += [op(SCALE, a=G["embed.weight"], c=G["embed.weight"], n=V * D, alpha=0.0)]
        plan += [op(EMBED_BWD, a=dcur, b=self.b_ids, c=G["embed.weight"], m=N, n=D)]

        self.plan_fb = plan          # forward+backward (loss+grads)
        self._mb, self._vb = mb, vb
        self._captured = False
        # device-side step counter: TICK increments, AdamW reads it and does
        # bias correction ON DEVICE — exact under CUDA graph replay.
        self.b_t = eng.new_tensor(np.zeros(1, dtype=np.float32))
        self.plan_step = (self.plan_fb
                          + [op(TICK, a=self.b_t)]
                          + [op(ADAMW, a=self.pb[nm], b=self.gbuf[nm],
                                c=self._mb[nm], d=self._vb[nm], tb=self.b_t,
                                n=int(np.prod(self.shapes[nm])),
                                alpha=self.lr, beta=self.lr * self.wd, gamma=1e-8)
                             for nm in self.pnames])

    def _adamw_ops(self, t: int):
        bc1 = 1.0 - 0.9 ** t
        bc2 = 1.0 - 0.95 ** t
        a_lr = self.lr * float(np.sqrt(bc2)) / bc1
        g_eps = 1e-8 * float(np.sqrt(bc2))
        return [op(ADAMW, a=self.pb[nm], b=self.gbuf[nm], c=self._mb[nm], d=self._vb[nm],
                   tb=-1, n=int(np.prod(self.shapes[nm])), alpha=a_lr,
                   beta=self.lr * self.wd, gamma=g_eps) for nm in self.pnames]

    # ── API ──
    def step(self, tokens: np.ndarray, targets: np.ndarray, t: int = None) -> float:
        """One full training step (single native call). If t is given, uses
        host-folded bias correction (parity tests); otherwise the device-side
        step counter (TICK) handles it — the graph-correct default."""
        self.eng.upload(self.b_ids, tokens.reshape(-1).astype(np.float32))
        self.eng.upload(self.b_tgt, targets.reshape(-1).astype(np.float32))
        if t is None:
            self.eng.run(self.plan_step, sync=False)
        else:
            self.eng.run(self.plan_fb + self._adamw_ops(t), sync=False)
        return float(self.eng.download(self.b_loss, (1,))[0])

    def capture(self, t: int = None) -> None:
        """Capture the whole step as a CUDA graph. Default: device-side t —
        bias correction stays exact across replays."""
        if t is None:
            self.eng.capture(self.plan_step)
        else:
            self.eng.capture(self.plan_fb + self._adamw_ops(t))
        self._captured = True

    def replay_step(self, tokens: np.ndarray, targets: np.ndarray) -> float:
        self.eng.upload(self.b_ids, tokens.reshape(-1).astype(np.float32))
        self.eng.upload(self.b_tgt, targets.reshape(-1).astype(np.float32))
        self.eng.replay(1, sync=False)
        return float(self.eng.download(self.b_loss, (1,))[0])

    def grads(self) -> Dict[str, np.ndarray]:
        return {nm: self.eng.download(self.gbuf[nm], self.shapes[nm])
                for nm in self.pnames}


def compile_model(model, batch: int, seq: int, lib_path: str = "libaxeng.so",
                  lr: float = 3e-4, wd: float = 0.1, tf32: bool = True,
                  recompute_attn: bool = True) -> CompiledTransformer:
    """Compile an axis.nn.Transformer instance into a native training step.

        ct = axis.compile_model(model, batch=8, seq=2048)
        for step, (x, y) in enumerate(loader):
            loss = ct.step(x, y)          # ONE native call — fwd+bwd+AdamW
    """
    blk = model.blocks[0]
    cfg = dict(vocab_size=model.embed.weight.shape[0],
               dim=model.embed.weight.shape[1],
               n_layers=len(model.blocks),
               n_heads=blk.attn.n_heads,
               n_kv_heads=blk.attn.n_kv_heads,
               mlp_hidden=blk.mlp.gate_proj.weight.shape[1],
               tie_embeddings=model.tie_embeddings)
    weights = {n: p.data for n, p in model.named_parameters()}
    return CompiledTransformer(lib_path, cfg, weights, batch, seq,
                               lr=lr, wd=wd, tf32=tf32,
                               recompute_attn=recompute_attn)
