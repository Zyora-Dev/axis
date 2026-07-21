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
                         SILU_BWD, EMBED, EMBED_BWD, CE, ADAMW, TICK, CAST,
                         FLASH, ROWDOT, FLASH_BWD, ALLREDUCE, GROUP,
                         L2ACC, CLIPSCALE, COUNTVALID, BROADCAST)


class CompiledTransformer:
    def __init__(self, lib_path: str, cfg: dict, weights: Dict[str, np.ndarray],
                 batch: int, seq: int, lr: float = 3e-4, wd: float = 0.1,
                 eps: float = 1e-5, tf32: bool = True, recompute_attn: bool = True,
                 dtype: str = "fp32", attn_tile: int = 256, attn_impl: str = "auto",
                 lora_r: int = 0, lora_alpha: float = 16.0,
                 lora_targets=("q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"),
                 grad_sync: bool = False, max_grad_norm: float = 0.0,
                 zero_stage: int = 0, rank: int = 0, world: int = 1):
        # recompute_attn: kept for API compat — tiled attention always
        # recomputes probs per tile in backward (flash-style memory).
        # attn_impl: "auto" (fused flash kernel when eligible: bf16, head_dim
        # multiple of 16, <=128), "flash", or "tiled".
        # lora_r > 0: LoRA fine-tuning — base weights FROZEN (no grads, no
        # optimizer state, no fp32 masters in bf16 mode); trainable params are
        # the adapters only: W_eff = W + (alpha/r) * A @ B (A [in,r], B [r,out],
        # matching axis.lora). Adapter weights read from `weights` when present
        # (keys "...lora_a"/"...lora_b"), else initialized (A random, B zero).
        assert dtype in ("fp32", "bf16")
        assert attn_impl in ("auto", "flash", "tiled")
        self.cfg = cfg
        self.B, self.T = batch, seq
        self.lr, self.wd = lr, wd
        self.max_grad_norm = max_grad_norm
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
        lora = lora_r > 0
        sc_l = float(lora_alpha) / lora_r if lora else 0.0
        self.lora_r = lora_r
        lin_geom = {"attn.q_proj": (D, H * DH), "attn.k_proj": (D, KV * DH),
                    "attn.v_proj": (D, KV * DH), "attn.o_proj": (H * DH, D),
                    "mlp.gate_proj": (D, MLP), "mlp.up_proj": (D, MLP),
                    "mlp.down_proj": (MLP, D)}
        lset = set(lora_targets) if lora else set()

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
        # Trainable params: fp32 masters (uploaded) + fp32 grads + m/v; in bf16
        # mode also a bf16 mirror used by ALL compute (AdamW updates master
        # then refreshes the mirror). Frozen base weights (LoRA mode): compute
        # view only — no masters/grads/optimizer state.
        base_names = ["embed.weight"]
        for i in range(L):
            p = f"blocks.{i}."
            base_names += [p + "attn_norm.weight", p + "attn.q_proj.weight",
                           p + "attn.k_proj.weight", p + "attn.v_proj.weight",
                           p + "attn.o_proj.weight", p + "mlp_norm.weight",
                           p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight",
                           p + "mlp.down_proj.weight"]
        base_names.append("norm.weight")
        if not tied:
            base_names.append("lm_head.weight")
        anames = []
        if lora:
            for i in range(L):
                for pref in lin_geom:
                    if pref.split(".")[-1] in lset:
                        anames += [f"blocks.{i}.{pref}.lora_a",
                                   f"blocks.{i}.{pref}.lora_b"]
        self.pnames = anames if lora else base_names       # TRAINABLE params
        if lora:
            self.shapes = {}
            for i in range(L):
                for pref, (fin, fout) in lin_geom.items():
                    if pref.split(".")[-1] in lset:
                        self.shapes[f"blocks.{i}.{pref}.lora_a"] = (fin, lora_r)
                        self.shapes[f"blocks.{i}.{pref}.lora_b"] = (lora_r, fout)
        else:
            self.shapes = {nm: weights[nm].shape for nm in self.pnames}
        # ZeRO stage 1: shard the optimizer state (fp32 master + m + v) across
        # ranks. Each param is OWNED by one rank (greedy element-count balance);
        # only the owner holds/updates its optimizer state, then the updated
        # bf16 weights are broadcast so every rank has the full model for the
        # next forward. bf16 mirrors + grads stay replicated. Requires bf16 DP.
        zero = zero_stage >= 1 and world > 1
        if zero:
            assert hf, "ZeRO sharding requires dtype='bf16'"
            assert grad_sync, "ZeRO sharding requires grad_sync=True (multi-GPU)"
        load = [0] * world
        owner = {}
        for nm in sorted(self.pnames, key=lambda n: -int(np.prod(self.shapes[n]))):
            r = int(np.argmin(load)) if zero else 0
            owner[nm] = r
            load[r] += int(np.prod(self.shapes[nm]))
        self.owner, self.rank, self.world, self.zero = owner, rank, world, zero
        self.owned = [nm for nm in self.pnames if not zero or owner[nm] == rank]

        self.master, self.gbuf, mb, vb = {}, {}, {}, {}
        P = {}                              # compute-view of each param
        cast_in = []
        if lora:
            # frozen base: bf16 via ONE shared fp32 staging buffer (no
            # persistent masters — big memory win), or plain fp32 buffers
            if hf:
                stage = A(max(int(np.prod(weights[nm].shape)) for nm in base_names))
                for nm in base_names:
                    w = np.ascontiguousarray(weights[nm], dtype=np.float32)
                    P[nm] = A(w.size, isz)
                    eng.upload(stage, w.reshape(-1))
                    eng.run([op(CAST, a=stage, c=P[nm], m=w.size, tb=0)])
            else:
                for nm in base_names:
                    P[nm] = eng.new_tensor(
                        np.ascontiguousarray(weights[nm], dtype=np.float32))
            rng_l = np.random.default_rng(0)
            src = {}
            for i in range(L):
                for pref, (fin, fout) in lin_geom.items():
                    if pref.split(".")[-1] in lset:
                        ka, kb = f"blocks.{i}.{pref}.lora_a", f"blocks.{i}.{pref}.lora_b"
                        src[ka] = weights[ka] if ka in weights else \
                            (rng_l.standard_normal((fin, lora_r)) / np.sqrt(lora_r)).astype(np.float32)
                        src[kb] = weights[kb] if kb in weights else \
                            np.zeros((lora_r, fout), dtype=np.float32)
        else:
            src = weights
        if zero:
            # sharded: bf16 mirror + grad on ALL ranks (forward + reduce);
            # fp32 master/m/v only on the owner. mirror inited via staging.
            stage_z = A(max(int(np.prod(self.shapes[nm])) for nm in self.pnames))
            for nm in self.pnames:
                w = np.ascontiguousarray(src[nm], dtype=np.float32)
                P[nm] = A(w.size, isz)
                eng.upload(stage_z, w.reshape(-1))
                eng.run([op(CAST, a=stage_z, c=P[nm], m=w.size, tb=0)])
                self.gbuf[nm] = A(w.size)
                if owner[nm] == rank:
                    self.master[nm] = eng.new_tensor(w)
                    mb[nm] = eng.new_tensor(np.zeros_like(w))
                    vb[nm] = eng.new_tensor(np.zeros_like(w))
            self.pb = P
        else:
            for nm in self.pnames:
                w = np.ascontiguousarray(src[nm], dtype=np.float32)
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
        self.b_denom = A(1)          # valid-token count for CE mean (ignore_index)
        # device learning rate (schedulable under CUDA graphs) + grad-clip scalars
        self.b_lr = eng.new_tensor(np.array([lr], dtype=np.float32))
        self.b_gnorm = A(1)          # global grad sumsq accumulator
        self.b_gscale = A(1)         # clip scale = min(1, max_norm/||g||)

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
        flash_ok = hf and DH % 16 == 0 and DH <= 128 and eng.has_flash
        if attn_impl == "flash" and not flash_ok:
            raise ValueError("flash attention needs a CUDA build (WMMA) with "
                             "dtype=bf16 and head_dim multiple of 16, <=128")
        use_flash = flash_ok if attn_impl == "auto" else attn_impl == "flash"
        self.attn_impl = "flash" if use_flash else "tiled"
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
        # flash: per-block LSE (fp32) saved by fwd for the fused backward
        lses = [A(BH * T) for _ in range(L)] if use_flash else []

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
        if use_flash:
            # fused-bwd scratch: D rowdot + fp32 grad targets (atomics need
            # fp32; CAST back to the bf16 dq2/dk2/dv1 afterwards)
            bD = A(N * H)
            dqf = A(N * H * DH); dkf = A(N * KV * DH); dvf = A(N * KV * DH)
        if lora:
            u_l = {}                        # (i, pref) -> saved x@A [N, r]
            for i in range(L):
                for pref in lin_geom:
                    if pref.split(".")[-1] in lset:
                        u_l[(i, pref)] = A(N * lora_r, isz)
            t1 = A(N * lora_r, isz)         # g @ B^T scratch

        G = self.gbuf
        eps_n = eps

        def lin_f(x, i, pref, y):
            """y = x@W (+ sc_l*(x@A)@B accumulated, u saved for backward)"""
            fin, fout = lin_geom[pref]
            key = f"blocks.{i}.{pref}"
            o_ = [op(GEMM_SB, a=x, b=P[key + ".weight"], c=y, m=N, k=fin, n=fout, batch=1, dt=DT)]
            if lora and pref.split(".")[-1] in lset:
                u = u_l[(i, pref)]
                o_ += [op(GEMM_SB, a=x, b=P[key + ".lora_a"], c=u, m=N, k=fin, n=lora_r, batch=1, dt=DT),
                       op(GEMM_SB, a=u, b=P[key + ".lora_b"], c=y, m=N, k=lora_r, n=fout, batch=1, dt=DT, alpha=sc_l, beta=1.0)]
            return o_

        def lin_b(g, x, i, pref, dx):
            """dx = g@W_eff^T; grads: base dW (full) or dA/dB (LoRA)"""
            fin, fout = lin_geom[pref]
            key = f"blocks.{i}.{pref}"
            o_ = [op(GEMM_SB, a=g, b=P[key + ".weight"], c=dx, m=N, k=fout, n=fin, batch=1, tb=1, dt=DT)]
            if not lora:
                o_ += [op(GEMM_SB, a=x, b=g, c=G[key + ".weight"], m=fin, n=fout, k=N, batch=1, tb=2, dt=DTW)]
            elif pref.split(".")[-1] in lset:
                u = u_l[(i, pref)]
                o_ += [op(GEMM_SB, a=g, b=P[key + ".lora_b"], c=t1, m=N, k=fout, n=lora_r, batch=1, tb=1, dt=DT),
                       op(GEMM_SB, a=t1, b=P[key + ".lora_a"], c=dx, m=N, k=lora_r, n=fin, batch=1, tb=1, dt=DT, alpha=sc_l, beta=1.0),
                       op(GEMM_SB, a=x, b=t1, c=G[key + ".lora_a"], m=fin, n=lora_r, k=N, batch=1, tb=2, dt=DTW, alpha=sc_l),
                       op(GEMM_SB, a=u, b=g, c=G[key + ".lora_b"], m=lora_r, n=fout, k=N, batch=1, tb=2, dt=DTW, alpha=sc_l)]
            return o_

        plan = [op(EMBED, a=P["embed.weight"], b=self.b_ids, c=xs[0], m=N, n=D, dt=DT)]

        # ── forward blocks ──
        for i in range(L):
            p = f"blocks.{i}."
            plan += [
                op(RMSNORM, a=xs[i], b=P[p + "attn_norm.weight"], c=ys[i], m=N, n=D, dt=DT, alpha=eps_n),
            ]
            plan += lin_f(ys[i], i, "attn.q_proj", q0)
            plan += lin_f(ys[i], i, "attn.k_proj", k0)
            plan += lin_f(ys[i], i, "attn.v_proj", v0)
            plan += [
                op(PERM_0213, a=q0, c=q1, m=B, n=T, k=H, batch=DH, dt=DT),
                op(PERM_0213, a=k0, c=k1, m=B, n=T, k=KV, batch=DH, dt=DT),
                op(PERM_0213, a=v0, c=v1s[i], m=B, n=T, k=KV, batch=DH, dt=DT),
                op(ROPE, a=q1, b=b_cos, d=b_sin, c=q2s[i], batch=BH, m=T, n=DH, dt=DT),
                op(ROPE, a=k1, b=b_cos, d=b_sin, c=k2s[i], batch=BKV, m=T, n=DH, dt=DT),
            ]
            if use_flash:
                # fused kernel: scores+softmax+pv in ONE launch, probs stay
                # on-chip (online softmax), GQA + causal handled in-kernel;
                # LSE saved for the fused backward
                plan += [op(FLASH, a=q2s[i], b=k2s[i], d=v1s[i], c=at,
                            m=T, n=DH, k=KV, batch=B, tb=H, sa=lses[i],
                            dt=DT, alpha=scale)]
            else:
                # streaming GQA attention: batch over kv groups; per group-head
                # j and query tile, only the causal key range
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
            ]
            plan += lin_f(at2s[i], i, "attn.o_proj", o_)
            plan += [
                op(ADD, a=xs[i], b=o_, c=r1s[i], n=N * D, dt=DT),
                op(RMSNORM, a=r1s[i], b=P[p + "mlp_norm.weight"], c=zs[i], m=N, n=D, dt=DT, alpha=eps_n),
            ]
            plan += lin_f(zs[i], i, "mlp.gate_proj", gs[i])
            plan += lin_f(zs[i], i, "mlp.up_proj", us[i])
            plan += [op(SILU_MUL, a=gs[i], b=us[i], c=hs[i], n=N * MLP, dt=DT)]
            plan += lin_f(hs[i], i, "mlp.down_proj", mo)
            plan += [op(ADD, a=r1s[i], b=mo, c=xs[i + 1], n=N * D, dt=DT)]

        # ── head + CE ──
        lm_w = P["embed.weight"] if tied else P["lm_head.weight"]
        plan += [
            op(RMSNORM, a=xs[L], b=P["norm.weight"], c=xf, m=N, n=D, dt=DT, alpha=eps_n),
            (op(GEMM_SB, a=xf, b=lm_w, c=logits, m=N, k=D, n=V, batch=1, tb=1, dt=DT)
             if tied else
             op(GEMM_SB, a=xf, b=lm_w, c=logits, m=N, k=D, n=V, batch=1, dt=DT)),
            op(SCALE, a=self.b_loss, c=self.b_loss, n=1, alpha=0.0),
            op(SCALE, a=self.b_denom, c=self.b_denom, n=1, alpha=0.0),
            op(COUNTVALID, a=self.b_tgt, c=self.b_denom, m=N),
            op(CE, a=logits, b=self.b_tgt, c=dlogits, d=self.b_loss, sa=self.b_denom, m=N, n=V, dt=DT),
        ]

        # ── head backward ──
        if tied:
            plan += [op(GEMM_SB, a=dlogits, b=lm_w, c=dxf, m=N, k=V, n=D, batch=1, tb=0, dt=DT)]
            if not lora:
                plan += [op(GEMM_SB, a=dlogits, b=xf, c=G["embed.weight"], m=V, n=D, k=N, batch=1, tb=2, dt=DTW)]
        else:
            plan += [op(GEMM_SB, a=dlogits, b=lm_w, c=dxf, m=N, k=V, n=D, batch=1, tb=1, dt=DT)]
            if not lora:
                plan += [op(GEMM_SB, a=xf, b=dlogits, c=G["lm_head.weight"], m=D, n=V, k=N, batch=1, tb=2, dt=DTW)]
        plan += [op(RMSNORM_BWD, a=xs[L], b=P["norm.weight"], d=dxf, c=dcur, tb=tmpD, m=N, n=D, dt=DT, alpha=eps_n)]
        if not lora:
            plan += [op(COLSUM, a=tmpD, c=G["norm.weight"], m=N, n=D, dt=DTW)]

        # ── backward blocks (reverse) ──
        for i in reversed(range(L)):
            p = f"blocks.{i}."
            plan += lin_b(dcur, hs[i], i, "mlp.down_proj", dh_)
            plan += [op(SILU_BWD, a=gs[i], b=us[i], d=dh_, c=dg_, tb=du_, n=N * MLP, dt=DT)]
            plan += lin_b(dg_, zs[i], i, "mlp.gate_proj", dz1)
            plan += lin_b(du_, zs[i], i, "mlp.up_proj", dz2)
            plan += [
                op(ADD, a=dz1, b=dz2, c=dz, n=N * D, dt=DT),
                op(RMSNORM_BWD, a=r1s[i], b=P[p + "mlp_norm.weight"], d=dz, c=dr1b, tb=tmpD, m=N, n=D, dt=DT, alpha=eps_n),
            ]
            if not lora:
                plan += [op(COLSUM, a=tmpD, c=G[p + "mlp_norm.weight"], m=N, n=D, dt=DTW)]
            plan += [op(ADD, a=dcur, b=dr1b, c=dr1, n=N * D, dt=DT)]
            plan += lin_b(dr1, at2s[i], i, "attn.o_proj", dat2)
            plan += [
                op(PERM_0213, a=dat2, c=dat, m=B, n=T, k=H, batch=DH, dt=DT),
            ]
            if use_flash:
                plan += [
                    # D = rowsum(dO * O): both pre-perm [B,T,H,DH] -> rows B*T*H
                    op(ROWDOT, a=dat2, b=at2s[i], c=bD, m=N * H, n=DH, dt=DT),
                    # fp32 grad targets (atomics accumulate; GQA group-sum
                    # happens inside the kernel's dk/dv atomicAdds)
                    op(SCALE, a=dqf, c=dqf, n=N * H * DH, alpha=0.0),
                    op(SCALE, a=dkf, c=dkf, n=N * KV * DH, alpha=0.0),
                    op(SCALE, a=dvf, c=dvf, n=N * KV * DH, alpha=0.0),
                    op(FLASH_BWD, a=q2s[i], b=k2s[i], d=v1s[i], c=dat,
                       sa=lses[i], sb=bD, tb=dqf, sc=dkf, oa=dvf, ob=H,
                       m=T, n=DH, k=KV, batch=B, dt=DT, alpha=scale),
                    op(CAST, a=dqf, c=dq2, m=N * H * DH, tb=0),
                    op(CAST, a=dkf, c=dk2, m=N * KV * DH, tb=0),
                    op(CAST, a=dvf, c=dv1, m=N * KV * DH, tb=0),
                ]
            else:
                plan += [
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
            ]
            plan += lin_b(dq0, ys[i], i, "attn.q_proj", dy1)
            plan += lin_b(dk0, ys[i], i, "attn.k_proj", dy2)
            plan += lin_b(dv0, ys[i], i, "attn.v_proj", dy3)
            plan += [
                op(ADD, a=dy1, b=dy2, c=dy, n=N * D, dt=DT),
                op(ADD, a=dy, b=dy3, c=dy, n=N * D, dt=DT),
                op(RMSNORM_BWD, a=xs[i], b=P[p + "attn_norm.weight"], d=dy, c=dx1, tb=tmpD, m=N, n=D, dt=DT, alpha=eps_n),
            ]
            if not lora:
                plan += [op(COLSUM, a=tmpD, c=G[p + "attn_norm.weight"], m=N, n=D, dt=DTW)]
            plan += [
                op(ADD, a=dr1, b=dx1, c=dxin, n=N * D, dt=DT),
                op(COPY, a=dxin, c=dcur, n=N * D, dt=DT),
            ]
        if not lora:
            if not tied:
                plan += [op(SCALE, a=G["embed.weight"], c=G["embed.weight"], n=V * D, alpha=0.0)]
            plan += [op(EMBED_BWD, a=dcur, b=self.b_ids, c=G["embed.weight"], m=N, n=D, dt=DTW if hf else 0)]

        self.plan_fb = plan
        self._mb, self._vb = mb, vb
        self._captured = False
        # data parallel: average grads across ranks between backward and AdamW
        # (ONE batched NCCL group over all trainable grads; caller must
        # eng.nccl_init(rank, world, uid) before stepping)
        self.plan_ar = []
        if grad_sync:
            self.plan_ar = ([op(GROUP, tb=0)]
                            + [op(ALLREDUCE, a=self.gbuf[nm],
                                  n=int(np.prod(self.shapes[nm])))
                               for nm in self.pnames]
                            + [op(GROUP, tb=1)])
        # device-side step counter — AdamW bias correction on device (graph-exact)
        self.b_t = eng.new_tensor(np.zeros(1, dtype=np.float32))
        # global-norm gradient clipping (device, graph-safe): zero the accumulator,
        # sum-of-squares over every trainable grad, then scale = min(1, mgn/||g||).
        # Runs AFTER the allreduce so it clips the averaged (global) gradient.
        self._clip = []
        if max_grad_norm and max_grad_norm > 0:
            self._clip = ([op(SCALE, a=self.b_gnorm, c=self.b_gnorm, n=1, alpha=0.0)]
                          + [op(L2ACC, a=self.gbuf[nm], c=self.b_gnorm,
                                n=int(np.prod(self.shapes[nm]))) for nm in self.pnames]
                          + [op(CLIPSCALE, a=self.b_gnorm, c=self.b_gscale,
                                alpha=max_grad_norm)])
        # wd only on >=2D params (norms/biases excluded — standard LLM recipe)
        self._wd = {nm: (self.wd if len(self.shapes[nm]) >= 2 else 0.0)
                    for nm in self.pnames}
        gsc = self.b_gscale if self._clip else 0
        # ZeRO stage 1: only the owner runs AdamW for a param; the updated bf16
        # weights are then broadcast from each owner so every rank has the full
        # model for the next forward. Non-ZeRO: owned == all params.
        self._bcast = []
        if self.zero:
            self._bcast = ([op(GROUP, tb=0)]
                           + [op(BROADCAST, a=self.pb[nm],
                                 n=int(np.prod(self.shapes[nm])),
                                 tb=self.owner[nm], dt=1) for nm in self.pnames]
                           + [op(GROUP, tb=1)])
        self.plan_step = (self.plan_fb + self.plan_ar + self._clip
                          + [op(TICK, a=self.b_t)]
                          + [op(ADAMW, a=self.master[nm], b=self.gbuf[nm],
                                c=self._mb[nm], d=self._vb[nm], tb=self.b_t,
                                sa=(self.pb[nm] if hf else 0),
                                sb=self.b_lr, oa=gsc,
                                n=int(np.prod(self.shapes[nm])),
                                alpha=self.lr, beta=self._wd[nm], gamma=1e-8)
                             for nm in self.owned]
                          + self._bcast)

    def _adamw_ops(self, t: int):
        hf = self.dtype == "bf16"
        bc1 = 1.0 - 0.9 ** t
        bc2 = 1.0 - 0.95 ** t
        a_lr = self.lr * float(np.sqrt(bc2)) / bc1
        g_eps = 1e-8 * float(np.sqrt(bc2))
        # host-folded bias correction (parity path); no device lr, no clip
        return [op(ADAMW, a=self.master[nm], b=self.gbuf[nm], c=self._mb[nm],
                   d=self._vb[nm], tb=-1, sa=(self.pb[nm] if hf else 0),
                   n=int(np.prod(self.shapes[nm])), alpha=a_lr,
                   beta=self._wd[nm], gamma=g_eps) for nm in self.owned] + self._bcast

    # ── API ──
    def set_lr(self, lr: float) -> None:
        """Update the learning rate for subsequent steps (works under CUDA
        graph replay — lr lives in a device buffer). Pair with any schedule."""
        self.lr = float(lr)
        self.eng.upload(self.b_lr, np.array([lr], dtype=np.float32))

    def step(self, tokens: np.ndarray, targets: np.ndarray, t: int = None) -> float:
        self.eng.upload(self.b_ids, tokens.reshape(-1).astype(np.float32))
        self.eng.upload(self.b_tgt, targets.reshape(-1).astype(np.float32))
        if t is None:
            self.eng.run(self.plan_step, sync=False)
        else:
            self.eng.run(self.plan_fb + self.plan_ar + self._adamw_ops(t), sync=False)
        return float(self.eng.download(self.b_loss, (1,))[0])

    def capture(self, t: int = None) -> None:
        if t is None:
            self.eng.capture(self.plan_step)
        else:
            self.eng.capture(self.plan_fb + self.plan_ar + self._adamw_ops(t))
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
        """Current (master, fp32) weights — for checkpointing/export. Under
        ZeRO, returns only this rank's OWNED shard (merge across ranks for the
        full model)."""
        return {nm: self.eng.download(self.master[nm], self.shapes[nm])
                for nm in self.owned}

    def state_dict(self) -> Dict[str, np.ndarray]:
        """Full resumable training state: trainable weights + AdamW moments +
        step counter + lr. `load_state_dict` restores it exactly. Under ZeRO,
        this rank saves only its owned shard (each rank saves its own)."""
        st = {}
        for nm in self.owned:
            sh = self.shapes[nm]
            st[f"w/{nm}"] = self.eng.download(self.master[nm], sh)
            st[f"m/{nm}"] = self.eng.download(self._mb[nm], sh)
            st[f"v/{nm}"] = self.eng.download(self._vb[nm], sh)
        st["_t"] = self.eng.download(self.b_t, (1,))
        st["_lr"] = np.array([self.lr], dtype=np.float32)
        return st

    def load_state_dict(self, st: Dict[str, np.ndarray]) -> None:
        """Restore weights + optimizer moments + step counter + lr. Refreshes
        the bf16 mirrors so the compute view matches the restored masters."""
        hf = self.dtype == "bf16"
        refresh = []
        for nm in self.owned:
            self.eng.upload(self.master[nm], st[f"w/{nm}"].reshape(-1))
            self.eng.upload(self._mb[nm], st[f"m/{nm}"].reshape(-1))
            self.eng.upload(self._vb[nm], st[f"v/{nm}"].reshape(-1))
            if hf:
                refresh.append(op(CAST, a=self.master[nm], c=self.pb[nm],
                                  m=int(np.prod(self.shapes[nm])), tb=0))
        if refresh:
            self.eng.run(refresh)
        self.eng.upload(self.b_t, np.asarray(st["_t"], dtype=np.float32).reshape(-1))
        self.set_lr(float(np.asarray(st["_lr"]).reshape(-1)[0]))


def compile_model(model, batch: int, seq: int, lib_path: str = None,
                  lr: float = 3e-4, wd: float = 0.1, tf32: bool = True,
                  recompute_attn: bool = True, dtype: str = "fp32",
                  attn_tile: int = 256, attn_impl: str = "auto",
                  grad_sync: bool = False, max_grad_norm: float = 0.0,
                  zero_stage: int = 0, rank: int = 0, world: int = 1) -> CompiledTransformer:
    """Compile an axis.nn.Transformer instance into a native training step.

        ct = axis.compile_model(model, batch=8, seq=2048, dtype="bf16")
        for x, y in loader:
            loss = ct.step(x, y)          # ONE native call — fwd+bwd+AdamW

    LoRA models (axis.lora.apply_lora) are detected automatically: base
    weights compile as frozen, only the adapters train.
    """
    if lib_path is None:                 # build/resolve the CUDA engine on demand
        from axis._build import engine_lib
        lib_path = engine_lib(nccl=grad_sync)
    blk = model.blocks[0]
    # normalize LoRA wrapping: "...q_proj.base.weight" -> "...q_proj.weight"
    weights = {n.replace(".base.weight", ".weight"): p.data
               for n, p in model.named_parameters()}
    from axis.lora import LoRALinear, _all_modules
    lora_r, lora_alpha = 0, 16.0
    for m in _all_modules(model):
        if isinstance(m, LoRALinear):
            lora_r = m.rank
            lora_alpha = m.scaling * m.rank
            break
    targets = tuple(sorted({k.split(".")[-2] for k in weights
                            if k.endswith(".lora_a")})) if lora_r else ()
    cfg = dict(vocab_size=weights["embed.weight"].shape[0],
               dim=weights["embed.weight"].shape[1],
               n_layers=len(model.blocks),
               n_heads=blk.attn.n_heads,
               n_kv_heads=blk.attn.n_kv_heads,
               mlp_hidden=weights["blocks.0.mlp.gate_proj.weight"].shape[1],
               tie_embeddings=model.tie_embeddings)
    return CompiledTransformer(lib_path, cfg, weights, batch, seq,
                               lr=lr, wd=wd, tf32=tf32,
                               recompute_attn=recompute_attn, dtype=dtype,
                               attn_tile=attn_tile, attn_impl=attn_impl,
                               lora_r=lora_r, lora_alpha=lora_alpha,
                               lora_targets=targets or ("q_proj", "k_proj",
                                                        "v_proj", "o_proj"),
                               grad_sync=grad_sync, max_grad_norm=max_grad_norm,
                               zero_stage=zero_stage, rank=rank, world=world)
