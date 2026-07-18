"""axis.compile — compile a Transformer training step to the C++ engine.

Builds ONE execution plan for the complete training step (embed -> N blocks ->
norm -> head -> CE -> full backward -> AdamW), executed natively (one Python
call per step) or as a captured CUDA graph. Scale-agnostic: the same lowering
loop handles any depth/width/config. The eager Axis engine is the oracle.

dtype="fp32" (reference) or "bf16" (params + activations in bf16, tensor-core
GEMMs; fp32 master weights, fp32 gradient accumulation for weights, fp32
reductions inside norm/softmax/CE kernels — standard mixed precision, no loss
scaling needed thanks to bf16's fp32 exponent range).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from axis.engine import (Engine, op, GEMM, ADD, RMSNORM, SILU_MUL, SCALE, COPY,
                         GEMM_SB, PERM_0213, ROPE, SOFTMAX_CAUSAL,
                         RMSNORM_BWD, COLSUM, SOFTMAX_BWD,
                         SILU_BWD, EMBED, EMBED_BWD, CE, ADAMW, TICK, CAST)


class CompiledTransformer:
    def __init__(self, lib_path: str, cfg: dict, weights: Dict[str, np.ndarray],
                 batch: int, seq: int, lr: float = 3e-4, wd: float = 0.1,
                 eps: float = 1e-5, tf32: bool = True, recompute_attn: bool = True,
                 dtype: str = "fp32", attn_tile: int = 256):
        # recompute_attn: kept for API compat — tiled attention always
        # recomputes probs per tile in backward (flash-style memory).
        assert dtype in ("fp32", "bf16")
        self.cfg = cfg
        self.B, self.T = batch, seq
        self.lr, self.wd = lr, wd
        self.dtype = dtype
        hf = dtype == "bf16"               # half storage
        DT = 1 if hf else 0                # per-op dt flag for storage ops
        DTW = 2 if hf else 0               # bf16-in / fp32-out (weight grads)
        isz = 2 if hf else 4               # activation/param element size
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

        # rope cache (always fp32)
        half = DH // 2
        freqs = 1.0 / (theta ** (np.arange(half, dtype=np.float32) / half))
        ang = np.outer(np.arange(T, dtype=np.float32), freqs)
        b_cos = eng.new_tensor(np.cos(ang).astype(np.float32))
        b_sin = eng.new_tensor(np.sin(ang).astype(np.float32))

        # ── params ──
        # masters: fp32 (uploaded). In bf16 mode each param also has a bf16
        # mirror used by ALL compute; AdamW updates master then refreshes the
        # mirror. Grads: fp32 always (GemmEx bf16-in/fp32-out).
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
        self.master, self.gbuf, mb, vb = {}, {}, {}, {}
        P = {}                              # compute-view of each param
        cast_in = []
        for nm in self.pnames:
            w = np.ascontiguousarray(weights[nm], dtype=np.float32)
            self.master[nm] = eng.new_tensor(w)
            self.gbuf[nm] = A(w.size)                       # fp32 grads
            mb[nm] = eng.new_tensor(np.zeros_like(w))
            vb[nm] = eng.new_tensor(np.zeros_like(w))
            if hf:
                P[nm] = A(w.size, isz)                      # bf16 mirror
                cast_in.append(op(CAST, a=self.master[nm], c=P[nm], m=w.size, tb=0))
            else:
                P[nm] = self.master[nm]
        self.pb = P
        if cast_in:
            eng.run(cast_in)                                # one-time weight cast

        # io buffers
        self.b_ids = A(N)
        self.b_tgt = A(N)
        self.b_loss = A(1)

        # ── per-block saved activations (dtype isz) ──
        # Attention is streaming (query-tiled) AND GQA-native:
        #  - probs are NEVER materialized at [T,T]: per tile of QT query rows
        #    only the causal key range [0, qs+qt) is computed (masked work
        #    skipped), scratch is O(QT*T); backward recomputes probs per tile.
        #  - k/v are NEVER repeated to H heads: GEMMs batch over kv groups
        #    (B*KV) and address each group's `rep` query heads via offsets, so
        #    saved k/v are KV-sized and dk/dv group-sums happen inside the
        #    accumulating GEMMs (beta=1).
        QT = max(1, min(attn_tile, T))
        tiles = [(qs, min(QT, T - qs)) for qs in range(0, T, QT)]
        self.attn_tile = QT
        rep = H // KV
        xs = [A(N * D, isz) for _ in range(L + 1)]
        ys = [A(N * D, isz) for _ in range(L)]
        q2s = [A(N * H * DH, isz) for _ in range(L)]
        k2s = [A(N * KV * DH, isz) for _ in range(L)]
        v1s = [A(N * KV * DH, isz) for _ in range(L)]
        at2s = [A(N * H * DH, isz) for _ in range(L)]
        r1s = [A(N * D, isz) for _ in range(L)]
        zs = [A(N * D, isz) for _ in range(L)]
        gs = [A(N * MLP, isz) for _ in range(L)]
        us = [A(N * MLP, isz) for _ in range(L)]
        hs = [A(N * MLP, isz) for _ in range(L)]

        # ── shared scratch ──
        q0 = A(N * H * DH, isz); k0 = A(N * KV * DH, isz); v0 = A(N * KV * DH, isz)
        q1 = A(N * H * DH, isz); k1 = A(N * KV * DH, isz)
        b_sct = A(BKV * QT * T, isz)         # score tile   [BKV, qt, kl]
        b_prt = A(BKV * QT * T, isz)         # probs tile
        at = A(N * H * DH, isz); o_ = A(N * D, isz); mo = A(N * D, isz)
        xf = A(N * D, isz)
        logits = A(N * V, isz); dlogits = A(N * V, isz)
        dxf = A(N * D, isz); dcur = A(N * D, isz); tmpD = A(N * D, isz)
        dh_ = A(N * MLP, isz); dg_ = A(N * MLP, isz); du_ = A(N * MLP, isz)
        dz1 = A(N * D, isz); dz2 = A(N * D, isz); dz = A(N * D, isz)
        dr1b = A(N * D, isz); dr1 = A(N * D, isz)
        dat2 = A(N * H * DH, isz); dat = A(N * H * DH, isz)
        b_dprt = A(BKV * QT * T, isz)        # dprobs tile
        b_dsct = A(BKV * QT * T, isz)        # dscores tile
        dq2 = A(N * H * DH, isz)
        dk2 = A(N * KV * DH, isz); dv1 = A(N * KV * DH, isz)
        dq1 = A(N * H * DH, isz); dk1 = A(N * KV * DH, isz)
        dq0 = A(N * H * DH, isz); dk0 = A(N * KV * DH, isz); dv0 = A(N * KV * DH, isz)
        dy1 = A(N * D, isz); dy2 = A(N * D, isz); dy3 = A(N * D, isz); dy = A(N * D, isz)
        dx1 = A(N * D, isz); dxin = A(N * D, isz)

        G = self.gbuf
        eps_n = eps
        plan = [op(EMBED, a=P["embed.weight"], b=self.b_ids, c=xs[0], m=N, n=D, dt=DT)]

        # ── forward blocks ──
        for i in range(L):
            p = f"blocks.{i}."
            plan += [
                op(RMSNORM, a=xs[i], b=P[p + "attn_norm.weight"], c=ys[i], m=N, n=D, dt=DT, alpha=eps_n),
                op(GEMM_SB, a=ys[i], b=P[p + "attn.q_proj.weight"], c=q0, m=N, k=D, n=H * DH, batch=1, dt=DT),
                op(GEMM_SB, a=ys[i], b=P[p + "attn.k_proj.weight"], c=k0, m=N, k=D, n=KV * DH, batch=1, dt=DT),
                op(GEMM_SB, a=ys[i], b=P[p + "attn.v_proj.weight"], c=v0, m=N, k=D, n=KV * DH, batch=1, dt=DT),
                op(PERM_0213, a=q0, c=q1, m=B, n=T, k=H, batch=DH, dt=DT),
                op(PERM_0213, a=k0, c=k1, m=B, n=T, k=KV, batch=DH, dt=DT),
                op(PERM_0213, a=v0, c=v1s[i], m=B, n=T, k=KV, batch=DH, dt=DT),
                op(ROPE, a=q1, b=b_cos, d=b_sin, c=q2s[i], batch=BH, m=T, n=DH, dt=DT),
                op(ROPE, a=k1, b=b_cos, d=b_sin, c=k2s[i], batch=BKV, m=T, n=DH, dt=DT),
            ]
            # streaming GQA attention: batch over kv groups; per group-head j
            # and query tile, only the causal key range
            for j in range(rep):
                for qs, qt in tiles:
                    kl = qs + qt
                    plan += [
                        op(GEMM_SB, a=q2s[i], b=k2s[i], c=b_sct, m=qt, k=DH, n=kl,
                           batch=BKV, tb=1, sa=rep * T * DH, sb=T * DH, sc=qt * kl,
                           oa=(j * T + qs) * DH, dt=DT, alpha=scale),
                        op(SOFTMAX_CAUSAL, a=b_sct, c=b_prt, batch=BKV, m=qt, n=kl,
                           k=qs, dt=DT),
                        op(GEMM_SB, a=b_prt, b=v1s[i], c=at, m=qt, k=kl, n=DH,
                           batch=BKV, sa=qt * kl, sb=T * DH, sc=rep * T * DH,
                           oc=(j * T + qs) * DH, dt=DT),
                    ]
            plan += [
                op(PERM_0213, a=at, c=at2s[i], m=B, n=H, k=T, batch=DH, dt=DT),
                op(GEMM_SB, a=at2s[i], b=P[p + "attn.o_proj.weight"], c=o_, m=N, k=H * DH, n=D, batch=1, dt=DT),
                op(ADD, a=xs[i], b=o_, c=r1s[i], n=N * D, dt=DT),
                op(RMSNORM, a=r1s[i], b=P[p + "mlp_norm.weight"], c=zs[i], m=N, n=D, dt=DT, alpha=eps_n),
                op(GEMM_SB, a=zs[i], b=P[p + "mlp.gate_proj.weight"], c=gs[i], m=N, k=D, n=MLP, batch=1, dt=DT),
                op(GEMM_SB, a=zs[i], b=P[p + "mlp.up_proj.weight"], c=us[i], m=N, k=D, n=MLP, batch=1, dt=DT),
                op(SILU_MUL, a=gs[i], b=us[i], c=hs[i], n=N * MLP, dt=DT),
                op(GEMM_SB, a=hs[i], b=P[p + "mlp.down_proj.weight"], c=mo, m=N, k=MLP, n=D, batch=1, dt=DT),
                op(ADD, a=r1s[i], b=mo, c=xs[i + 1], n=N * D, dt=DT),
            ]

        # ── head + CE ──
        lm_w = P["embed.weight"] if tied else P["lm_head.weight"]
        plan += [
            op(RMSNORM, a=xs[L], b=P["norm.weight"], c=xf, m=N, n=D, dt=DT, alpha=eps_n),
            (op(GEMM_SB, a=xf, b=lm_w, c=logits, m=N, k=D, n=V, batch=1, tb=1, dt=DT)
             if tied else
             op(GEMM_SB, a=xf, b=lm_w, c=logits, m=N, k=D, n=V, batch=1, dt=DT)),
            op(SCALE, a=self.b_loss, c=self.b_loss, n=1, alpha=0.0),
            op(CE, a=logits, b=self.b_tgt, c=dlogits, d=self.b_loss, m=N, n=V, dt=DT),
        ]

        # ── head backward ──
        if tied:
            plan += [
                op(GEMM_SB, a=dlogits, b=lm_w, c=dxf, m=N, k=V, n=D, batch=1, tb=0, dt=DT),
                op(GEMM_SB, a=dlogits, b=xf, c=G["embed.weight"], m=V, n=D, k=N, batch=1, tb=2, dt=DTW),
            ]
        else:
            plan += [
                op(GEMM_SB, a=dlogits, b=lm_w, c=dxf, m=N, k=V, n=D, batch=1, tb=1, dt=DT),
                op(GEMM_SB, a=xf, b=dlogits, c=G["lm_head.weight"], m=D, n=V, k=N, batch=1, tb=2, dt=DTW),
            ]
        plan += [
            op(RMSNORM_BWD, a=xs[L], b=P["norm.weight"], d=dxf, c=dcur, tb=tmpD, m=N, n=D, dt=DT, alpha=eps_n),
            op(COLSUM, a=tmpD, c=G["norm.weight"], m=N, n=D, dt=DTW),
        ]

        # ── backward blocks (reverse) ──
        for i in reversed(range(L)):
            p = f"blocks.{i}."
            plan += [
                op(GEMM_SB, a=dcur, b=P[p + "mlp.down_proj.weight"], c=dh_, m=N, k=D, n=MLP, batch=1, tb=1, dt=DT),
                op(GEMM_SB, a=hs[i], b=dcur, c=G[p + "mlp.down_proj.weight"], m=MLP, n=D, k=N, batch=1, tb=2, dt=DTW),
                op(SILU_BWD, a=gs[i], b=us[i], d=dh_, c=dg_, tb=du_, n=N * MLP, dt=DT),
                op(GEMM_SB, a=dg_, b=P[p + "mlp.gate_proj.weight"], c=dz1, m=N, k=MLP, n=D, batch=1, tb=1, dt=DT),
                op(GEMM_SB, a=du_, b=P[p + "mlp.up_proj.weight"], c=dz2, m=N, k=MLP, n=D, batch=1, tb=1, dt=DT),
                op(ADD, a=dz1, b=dz2, c=dz, n=N * D, dt=DT),
                op(GEMM_SB, a=zs[i], b=dg_, c=G[p + "mlp.gate_proj.weight"], m=D, n=MLP, k=N, batch=1, tb=2, dt=DTW),
                op(GEMM_SB, a=zs[i], b=du_, c=G[p + "mlp.up_proj.weight"], m=D, n=MLP, k=N, batch=1, tb=2, dt=DTW),
                op(RMSNORM_BWD, a=r1s[i], b=P[p + "mlp_norm.weight"], d=dz, c=dr1b, tb=tmpD, m=N, n=D, dt=DT, alpha=eps_n),
                op(COLSUM, a=tmpD, c=G[p + "mlp_norm.weight"], m=N, n=D, dt=DTW),
                op(ADD, a=dcur, b=dr1b, c=dr1, n=N * D, dt=DT),
                op(GEMM_SB, a=dr1, b=P[p + "attn.o_proj.weight"], c=dat2, m=N, k=D, n=H * DH, batch=1, tb=1, dt=DT),
                op(GEMM_SB, a=at2s[i], b=dr1, c=G[p + "attn.o_proj.weight"], m=H * DH, n=D, k=N, batch=1, tb=2, dt=DTW),
                op(PERM_0213, a=dat2, c=dat, m=B, n=T, k=H, batch=DH, dt=DT),
                # dk/dv accumulate over group heads + query tiles (beta=1)
                op(SCALE, a=dk2, c=dk2, n=N * KV * DH, dt=DT, alpha=0.0),
                op(SCALE, a=dv1, c=dv1, n=N * KV * DH, dt=DT, alpha=0.0),
            ]
            # streaming GQA attention backward: recompute probs per tile;
            # the beta=1 GEMMs also perform the GQA group-sum for dk/dv
            for j in range(rep):
                for qs, qt in tiles:
                    kl = qs + qt
                    plan += [
                        op(GEMM_SB, a=q2s[i], b=k2s[i], c=b_sct, m=qt, k=DH, n=kl,
                           batch=BKV, tb=1, sa=rep * T * DH, sb=T * DH, sc=qt * kl,
                           oa=(j * T + qs) * DH, dt=DT, alpha=scale),
                        op(SOFTMAX_CAUSAL, a=b_sct, c=b_prt, batch=BKV, m=qt, n=kl,
                           k=qs, dt=DT),
                        # dprobs = dat_tile @ v^T
                        op(GEMM_SB, a=dat, b=v1s[i], c=b_dprt, m=qt, k=DH, n=kl,
                           batch=BKV, tb=1, sa=rep * T * DH, sb=T * DH, sc=qt * kl,
                           oa=(j * T + qs) * DH, dt=DT),
                        # dv[:kl] += probs^T @ dat_tile
                        op(GEMM_SB, a=b_prt, b=dat, c=dv1, m=kl, n=DH, k=qt,
                           batch=BKV, tb=2, sa=qt * kl, sb=rep * T * DH, sc=T * DH,
                           ob=(j * T + qs) * DH, dt=DT, beta=1.0),
                        op(SOFTMAX_BWD, a=b_prt, b=b_dprt, c=b_dsct, batch=BKV,
                           m=qt, n=kl, dt=DT),
                        # dq_tile = dscores @ k[:kl] * scale
                        op(GEMM_SB, a=b_dsct, b=k2s[i], c=dq2, m=qt, k=kl, n=DH,
                           batch=BKV, sa=qt * kl, sb=T * DH, sc=rep * T * DH,
                           oc=(j * T + qs) * DH, dt=DT, alpha=scale),
                        # dk[:kl] += dscores^T @ q_tile * scale
                        op(GEMM_SB, a=b_dsct, b=q2s[i], c=dk2, m=kl, n=DH, k=qt,
                           batch=BKV, tb=2, sa=qt * kl, sb=rep * T * DH, sc=T * DH,
                           ob=(j * T + qs) * DH, dt=DT, alpha=scale, beta=1.0),
                    ]
            plan += [
                op(ROPE, a=dq2, b=b_cos, d=b_sin, c=dq1, batch=BH, m=T, n=DH, tb=1, dt=DT),
                op(ROPE, a=dk2, b=b_cos, d=b_sin, c=dk1, batch=BKV, m=T, n=DH, tb=1, dt=DT),
                op(PERM_0213, a=dq1, c=dq0, m=B, n=H, k=T, batch=DH, dt=DT),
                op(PERM_0213, a=dk1, c=dk0, m=B, n=KV, k=T, batch=DH, dt=DT),
                op(PERM_0213, a=dv1, c=dv0, m=B, n=KV, k=T, batch=DH, dt=DT),
                op(GEMM_SB, a=dq0, b=P[p + "attn.q_proj.weight"], c=dy1, m=N, k=H * DH, n=D, batch=1, tb=1, dt=DT),
                op(GEMM_SB, a=dk0, b=P[p + "attn.k_proj.weight"], c=dy2, m=N, k=KV * DH, n=D, batch=1, tb=1, dt=DT),
                op(GEMM_SB, a=dv0, b=P[p + "attn.v_proj.weight"], c=dy3, m=N, k=KV * DH, n=D, batch=1, tb=1, dt=DT),
                op(ADD, a=dy1, b=dy2, c=dy, n=N * D, dt=DT),
                op(ADD, a=dy, b=dy3, c=dy, n=N * D, dt=DT),
                op(GEMM_SB, a=ys[i], b=dq0, c=G[p + "attn.q_proj.weight"], m=D, n=H * DH, k=N, batch=1, tb=2, dt=DTW),
                op(GEMM_SB, a=ys[i], b=dk0, c=G[p + "attn.k_proj.weight"], m=D, n=KV * DH, k=N, batch=1, tb=2, dt=DTW),
                op(GEMM_SB, a=ys[i], b=dv0, c=G[p + "attn.v_proj.weight"], m=D, n=KV * DH, k=N, batch=1, tb=2, dt=DTW),
                op(RMSNORM_BWD, a=xs[i], b=P[p + "attn_norm.weight"], d=dy, c=dx1, tb=tmpD, m=N, n=D, dt=DT, alpha=eps_n),
                op(COLSUM, a=tmpD, c=G[p + "attn_norm.weight"], m=N, n=D, dt=DTW),
                op(ADD, a=dr1, b=dx1, c=dxin, n=N * D, dt=DT),
                op(COPY, a=dxin, c=dcur, n=N * D, dt=DT),
            ]
        if not tied:
            plan += [op(SCALE, a=G["embed.weight"], c=G["embed.weight"], n=V * D, alpha=0.0)]
        plan += [op(EMBED_BWD, a=dcur, b=self.b_ids, c=G["embed.weight"], m=N, n=D, dt=DTW if hf else 0)]

        self.plan_fb = plan
        self._mb, self._vb = mb, vb
        self._captured = False
        # device-side step counter — AdamW bias correction on device (graph-exact)
        self.b_t = eng.new_tensor(np.zeros(1, dtype=np.float32))
        self.plan_step = (self.plan_fb
                          + [op(TICK, a=self.b_t)]
                          + [op(ADAMW, a=self.master[nm], b=self.gbuf[nm],
                                c=self._mb[nm], d=self._vb[nm], tb=self.b_t,
                                sa=(self.pb[nm] if hf else 0),
                                n=int(np.prod(self.shapes[nm])),
                                alpha=self.lr, beta=self.lr * self.wd, gamma=1e-8)
                             for nm in self.pnames])

    def _adamw_ops(self, t: int):
        hf = self.dtype == "bf16"
        bc1 = 1.0 - 0.9 ** t
        bc2 = 1.0 - 0.95 ** t
        a_lr = self.lr * float(np.sqrt(bc2)) / bc1
        g_eps = 1e-8 * float(np.sqrt(bc2))
        return [op(ADAMW, a=self.master[nm], b=self.gbuf[nm], c=self._mb[nm],
                   d=self._vb[nm], tb=-1, sa=(self.pb[nm] if hf else 0),
                   n=int(np.prod(self.shapes[nm])), alpha=a_lr,
                   beta=self.lr * self.wd, gamma=g_eps) for nm in self.pnames]

    # ── API ──
    def step(self, tokens: np.ndarray, targets: np.ndarray, t: int = None) -> float:
        self.eng.upload(self.b_ids, tokens.reshape(-1).astype(np.float32))
        self.eng.upload(self.b_tgt, targets.reshape(-1).astype(np.float32))
        if t is None:
            self.eng.run(self.plan_step, sync=False)
        else:
            self.eng.run(self.plan_fb + self._adamw_ops(t), sync=False)
        return float(self.eng.download(self.b_loss, (1,))[0])

    def capture(self, t: int = None) -> None:
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

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Current (master, fp32) weights — for checkpointing/export."""
        return {nm: self.eng.download(self.master[nm], self.shapes[nm])
                for nm in self.pnames}


def compile_model(model, batch: int, seq: int, lib_path: str = "libaxeng.so",
                  lr: float = 3e-4, wd: float = 0.1, tf32: bool = True,
                  recompute_attn: bool = True, dtype: str = "fp32",
                  attn_tile: int = 256) -> CompiledTransformer:
    """Compile an axis.nn.Transformer instance into a native training step.

        ct = axis.compile_model(model, batch=8, seq=2048, dtype="bf16")
        for x, y in loader:
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
                               recompute_attn=recompute_attn, dtype=dtype,
                               attn_tile=attn_tile)
