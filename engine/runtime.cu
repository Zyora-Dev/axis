// Axis engine runtime — C ABI, own cuBLAS handle, execution plans, CUDA graphs.
//
// Dtype support: fp32 and bf16 storage. All math accumulates in fp32; bf16 is
// storage/tensor-core format (same exponent range as fp32 -> no loss scaling).
// Per-op `dt` flag: 0 = fp32; 1 = bf16 in/out; 2 = bf16 inputs, fp32 output
// (weight-grad GEMMs, colsum). Buffer table holds raw pointers; ops interpret.
//
// Build: nvcc -O3 -arch=sm_80 --shared -Xcompiler -fPIC runtime.cu -lcublas -o libaxeng.so
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#ifdef AXIS_NCCL
#include <nccl.h>
#endif

#define API extern "C" __attribute__((visibility("default")))
#define MAX_BUFS 8192
typedef __nv_bfloat16 bf16;

// ── op kinds ────────────────────────────────────────────────────────────────
enum OpKind {
    OP_GEMM = 0, OP_ADD = 1, OP_MUL = 2, OP_SILU_MUL = 3, OP_RMSNORM = 4,
    OP_ADAMW = 5, OP_SCALE = 6, OP_COPY = 7, OP_GEMM_SB = 8, OP_PERM_0213 = 9,
    OP_ROPE = 10, OP_SOFTMAX_CAUSAL = 11, OP_REPEAT_KV = 12, OP_RMSNORM_BWD = 13,
    OP_COLSUM = 14, OP_REPEAT_KV_BWD = 15, OP_SOFTMAX_BWD = 16, OP_SILU_BWD = 17,
    OP_EMBED = 18, OP_EMBED_BWD = 19, OP_CE = 20, OP_TICK = 21, OP_CAST = 22,
    OP_FLASH = 23, OP_ROWDOT = 24, OP_FLASH_BWD = 25,
    OP_ALLREDUCE = 26, OP_GROUP = 27,
};

typedef struct {
    int   kind;
    int   a, b, c, d;
    int   m, n, k;
    int   batch, tb;
    int   sa, sb, sc;
    int   dt;                    // 0 fp32 | 1 bf16 | 2 bf16-in fp32-out
    int   oa, ob, oc;            // element offsets into a/b/c (GEMM tiling)
    float alpha, beta, gamma;
} EngOp;

// ── global state ────────────────────────────────────────────────────────────
static cublasHandle_t g_blas = nullptr;
static cudaStream_t   g_stream = nullptr;
static void*          g_ws = nullptr;
static void*          g_bufs[MAX_BUFS];
static cudaGraphExec_t g_graph_exec = nullptr;
#ifdef AXIS_NCCL
static ncclComm_t     g_comm = nullptr;
#endif

// ── load/store helpers (fp32 math, T storage) ───────────────────────────────
__device__ __forceinline__ float ldf(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float ldf(const bf16* p, long long i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ void stf(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void stf(bf16* p, long long i, float v) { p[i] = __float2bfloat16(v); }

// ── kernels (templated on storage type) ─────────────────────────────────────
template <typename T>
__global__ void k_add(const T* a, const T* b, T* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) stf(c, i, ldf(a, i) + ldf(b, i));
}
template <typename T>
__global__ void k_mul(const T* a, const T* b, T* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) stf(c, i, ldf(a, i) * ldf(b, i));
}
template <typename T>
__global__ void k_silu_mul(const T* a, const T* b, T* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = ldf(a, i);
        stf(c, i, (x / (1.f + expf(-x))) * ldf(b, i));
    }
}
template <typename T>
__global__ void k_scale(const T* a, T* c, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // alpha==0 is an explicit ZERO (never read a: garbage NaN * 0 = NaN)
    if (i < n) stf(c, i, alpha == 0.f ? 0.f : ldf(a, i) * alpha);
}
template <typename T>
__global__ void k_copy(const T* a, T* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) stf(c, i, ldf(a, i));
}
template <typename T>
__global__ void k_rmsnorm(const T* x, const T* w, T* o, int rows, int dim, float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    extern __shared__ float sh[];
    float acc = 0.f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float v = ldf(x, (long long)r * dim + j);
        acc += v * v;
    }
    sh[threadIdx.x] = acc; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / dim + eps);
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        stf(o, (long long)r * dim + j, ldf(x, (long long)r * dim + j) * inv * ldf(w, j));
}
// AdamW. tbuf: device step counter (bias correction on device — graph-exact).
// pbf: optional bf16 param mirror (master p stays fp32; pbf gets the cast).
__global__ void k_adamw(float* p, const float* g, float* m, float* v,
                        const float* tbuf, bf16* pbf,
                        float alpha, float beta, float gamma, int n) {
    const float b1 = 0.9f, b2 = 0.95f;
    __shared__ float s_alpha, s_gamma;
    if (threadIdx.x == 0) {
        if (tbuf) {
            float t = tbuf[0];
            float bc1 = 1.f - powf(b1, t);
            float bc2 = 1.f - powf(b2, t);
            s_alpha = alpha * sqrtf(bc2) / bc1;
            s_gamma = gamma * sqrtf(bc2);
        } else { s_alpha = alpha; s_gamma = gamma; }
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float mi = b1 * m[i] + (1.f - b1) * gi;
    float vi = b2 * v[i] + (1.f - b2) * gi * gi;
    m[i] = mi; v[i] = vi;
    float pi = p[i] * (1.f - beta) - s_alpha * mi / (sqrtf(vi) + s_gamma);
    p[i] = pi;
    if (pbf) pbf[i] = __float2bfloat16(pi);
}
__global__ void k_tick(float* t) { if (threadIdx.x == 0 && blockIdx.x == 0) t[0] += 1.f; }

template <typename T>
__global__ void k_perm0213(const T* x, T* o, int d0, int d1, int d2, int d3) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)d0 * d1 * d2 * d3;
    if (i >= total) return;
    int j3 = i % d3;
    long long r = i / d3;
    int j2 = r % d2; r /= d2;
    int j1 = r % d1; r /= d1;
    int j0 = (int)r;
    o[(((long long)j0 * d2 + j2) * d1 + j1) * d3 + j3] = x[i];
}
template <typename T>
__global__ void k_rope(const T* x, const float* cs, const float* sn,
                       T* o, int T_, int dh, long long rows) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = rows * dh;
    if (i >= total) return;
    int j = i % dh;
    long long r = i / dh;
    int t = (int)(r % T_);
    int half = dh / 2;
    const T* xr = x + r * dh;
    if (j < half)
        stf(o, i, ldf(xr, j) * cs[t * half + j] - ldf(xr, j + half) * sn[t * half + j]);
    else {
        int jj = j - half;
        stf(o, i, ldf(xr, jj) * sn[t * half + jj] + ldf(xr, j) * cs[t * half + jj]);
    }
}
template <typename T>
__global__ void k_rope_inv(const T* g, const float* cs, const float* sn,
                           T* o, int T_, int dh, long long rows) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = rows * dh;
    if (i >= total) return;
    int j = i % dh;
    long long r = i / dh;
    int t = (int)(r % T_);
    int half = dh / 2;
    const T* gr = g + r * dh;
    if (j < half)
        stf(o, i, ldf(gr, j) * cs[t * half + j] + ldf(gr, j + half) * sn[t * half + j]);
    else {
        int jj = j - half;
        stf(o, i, -ldf(gr, jj) * sn[t * half + jj] + ldf(gr, j) * cs[t * half + jj]);
    }
}
// causal softmax over a query tile: row r covers global query row qoff+(r%rpb);
// valid keys j <= that. width = key range of this tile. rpb=width for untiled.
template <typename T>
__global__ void k_softmax_causal(const T* x, T* o, int rpb, int width, int qoff, long long rows) {
    long long r = blockIdx.x;
    if (r >= rows) return;
    int t = qoff + (int)(r % rpb);
    if (t > width - 1) t = width - 1;
    extern __shared__ float sh[];
    const T* xr = x + r * width;
    float mx = -1e30f;
    for (int j = threadIdx.x; j <= t; j += blockDim.x) mx = fmaxf(mx, ldf(xr, j));
    sh[threadIdx.x] = mx; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    mx = sh[0]; __syncthreads();
    float acc = 0.f;
    for (int j = threadIdx.x; j <= t; j += blockDim.x) acc += expf(ldf(xr, j) - mx);
    sh[threadIdx.x] = acc; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float denom = sh[0];
    T* orow = o + r * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x)
        stf(orow, j, (j <= t) ? expf(ldf(xr, j) - mx) / denom : 0.f);
}
template <typename T>
__global__ void k_repeat_kv(const T* x, T* o, int B, int KV, int H, int T_, int dh) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)B * H * T_ * dh;
    if (i >= total) return;
    int j = i % dh;
    long long r = i / dh;
    int t = r % T_; r /= T_;
    int h = r % H; r /= H;
    int b = (int)r;
    int rep = H / KV;
    o[i] = x[(((long long)b * KV + h / rep) * T_ + t) * dh + j];
}
// rmsnorm backward: dx (dtype T), tmp (dtype T) for dw colsum.
template <typename T>
__global__ void k_rmsnorm_bwd(const T* x, const T* w, const T* g,
                              T* dx, T* tmp, int rows, int dim, float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    extern __shared__ float sh[];
    float ss = 0.f, gwx = 0.f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float xv = ldf(x, (long long)r * dim + j);
        float gv = ldf(g, (long long)r * dim + j);
        ss += xv * xv;
        gwx += gv * ldf(w, j) * xv;
    }
    sh[threadIdx.x] = ss; sh[blockDim.x + threadIdx.x] = gwx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh[threadIdx.x] += sh[threadIdx.x + s];
            sh[blockDim.x + threadIdx.x] += sh[blockDim.x + threadIdx.x + s];
        }
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / dim + eps);
    float coef = inv * inv * inv * (sh[blockDim.x] / dim);
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        long long idx = (long long)r * dim + j;
        float xv = ldf(x, idx), gv = ldf(g, idx);
        stf(dx, idx, gv * ldf(w, j) * inv - xv * coef);
        stf(tmp, idx, gv * xv * inv);
    }
}
// colsum: T in -> fp32 out (weight grads stay fp32)
template <typename T>
__global__ void k_colsum(const T* a, float* c, int rows, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float acc = 0.f;
    for (int r = 0; r < rows; r++) acc += ldf(a, (long long)r * n + j);
    c[j] = acc;
}
template <typename T>
__global__ void k_repeat_kv_bwd(const T* g, T* o, int B, int KV, int H, int T_, int dh) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)B * KV * T_ * dh;
    if (i >= total) return;
    int j = i % dh;
    long long r = i / dh;
    int t = r % T_; r /= T_;
    int kv = r % KV; r /= KV;
    int b = (int)r;
    int rep = H / KV;
    float acc = 0.f;
    for (int q = 0; q < rep; q++)
        acc += ldf(g, (((long long)b * H + kv * rep + q) * T_ + t) * dh + j);
    stf(o, i, acc);
}
template <typename T>
__global__ void k_softmax_bwd(const T* p, const T* dp, T* ds, int width, long long rows) {
    long long r = blockIdx.x;
    if (r >= rows) return;
    extern __shared__ float sh[];
    float dot = 0.f;
    for (int j = threadIdx.x; j < width; j += blockDim.x)
        dot += ldf(dp, r * width + j) * ldf(p, r * width + j);
    sh[threadIdx.x] = dot; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    dot = sh[0];
    for (int j = threadIdx.x; j < width; j += blockDim.x)
        stf(ds, r * width + j, ldf(p, r * width + j) * (ldf(dp, r * width + j) - dot));
}
template <typename T>
__global__ void k_silu_bwd(const T* g, const T* u, const T* grad,
                           T* dg, T* du, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = ldf(g, i);
    float sig = 1.f / (1.f + expf(-x));
    float silu = x * sig;
    float dsilu = sig * (1.f + x * (1.f - sig));
    float gr = ldf(grad, i);
    stf(dg, i, gr * ldf(u, i) * dsilu);
    stf(du, i, gr * silu);
}
template <typename T>
__global__ void k_embed(const T* table, const float* ids, T* o, int N, int D) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)N * D) return;
    int r = (int)(i / D), j = (int)(i % D);
    o[i] = table[(long long)((int)ids[r]) * D + j];
}
// embed backward: grads accumulate fp32 (atomicAdd float)
template <typename T>
__global__ void k_embed_bwd(const T* g, const float* ids, float* dT, int N, int D) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)N * D) return;
    int r = (int)(i / D), j = (int)(i % D);
    atomicAdd(&dT[(long long)((int)ids[r]) * D + j], ldf(g, i));
}
template <typename T>
__global__ void k_ce(const T* logits, const float* tgt, T* dlogits,
                     float* loss, int N, int V) {
    int r = blockIdx.x;
    if (r >= N) return;
    extern __shared__ float sh[];
    const T* lr = logits + (long long)r * V;
    float mx = -1e30f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) mx = fmaxf(mx, ldf(lr, j));
    sh[threadIdx.x] = mx; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    mx = sh[0]; __syncthreads();
    float se = 0.f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) se += expf(ldf(lr, j) - mx);
    sh[threadIdx.x] = se; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    se = sh[0];
    int t = (int)tgt[r];
    float inv = 1.f / N;
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        float p = expf(ldf(lr, j) - mx) / se;
        stf(dlogits, (long long)r * V + j, (p - (j == t ? 1.f : 0.f)) * inv);
    }
    if (threadIdx.x == 0)
        atomicAdd(loss, (logf(se) - (ldf(lr, t) - mx)) * inv);
}
__global__ void k_cast_f2b(const float* a, bf16* c, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = __float2bfloat16(a[i]);
}
__global__ void k_cast_b2f(const bf16* a, float* c, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = __bfloat162float(a[i]);
}

// ── fused flash attention forward (bf16, WMMA tensor cores) ───────────────
// One block per (b*h, query tile of BR rows). K/V stream through shared memory
// in BC-column tiles with ONLINE softmax — probabilities never touch HBM.
// GQA-native: kv head = h / (H/KV). Causal: tiles fully above the diagonal are
// skipped; the diagonal tile is masked per element. fp32 accumulation.
// q,k,v,o: [B,H|KV,T,DH] row-major bf16. lse (optional): [B*H,T] fp32.
template <int BR, int BC>
__global__ void k_flash_fwd(const bf16* __restrict__ q, const bf16* __restrict__ k,
                            const bf16* __restrict__ v, bf16* __restrict__ o,
                            float* lse, int T, int DH, int H, int KV, float scale) {
    const int qb = blockIdx.x * BR;
    const int bh = blockIdx.y;
    const int b = bh / H, h = bh % H;
    const int g = h / (H / KV);
    const long long qoff = (long long)bh * T * DH;
    const long long koff = ((long long)b * KV + g) * T * DH;
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int warp = tid >> 5, nwarp = nthr >> 5;

    // padded row strides kill wmma shared-memory bank conflicts (128B rows
    // all land on the same banks otherwise)
    const int QLD = DH + 8;                  // bf16 rows of Q/K/V tiles
    const int PLD = BC + 8;                  // bf16 rows of P
    const int SLD = BC + 4;                  // fp32 rows of S
    const int OLD = DH + 4;                  // fp32 rows of P@V result
    extern __shared__ char smem[];
    const int SFP = BR * ((SLD > OLD) ? SLD : OLD);
    float* Sfp  = (float*)smem;              // scores, then P@V (union)
    float* Oacc = Sfp + SFP;                 // BR*DH (linear access, unpadded)
    float* mrow = Oacc + BR * DH;            // BR
    float* lrow = mrow + BR;                 // BR
    float* rrow = lrow + BR;                 // BR per-tile rescale
    bf16* Qs = (bf16*)(rrow + BR);           // BR*QLD
    bf16* Ks = Qs + BR * QLD;                // BC*QLD
    bf16* Vs = Ks + BC * QLD;                // BC*QLD
    bf16* Pb = Vs + BC * QLD;                // BR*PLD

    const bf16 zb = __float2bfloat16(0.f);
    // vectorized 128-bit loads: 8 bf16/thread (DH%16==0, QLD%8==0)
    const int V8 = DH / 8, Q8 = QLD / 8;
    const uint4 z4 = make_uint4(0, 0, 0, 0);
    {
        const uint4* q4 = (const uint4*)(q + qoff);
        uint4* Qs4 = (uint4*)Qs;
        for (int i = tid; i < BR * V8; i += nthr) {
            int r = i / V8, j = i % V8;
            Qs4[r * Q8 + j] = (qb + r < T) ? q4[(long long)(qb + r) * V8 + j] : z4;
        }
    }
    for (int i = tid; i < BR * DH; i += nthr) Oacc[i] = 0.f;
    if (tid < BR) { mrow[tid] = -1e30f; lrow[tid] = 0.f; }
    __syncthreads();

    const int qend = min(qb + BR - 1, T - 1);
    const int nkt = qend / BC + 1;
    const int RT = BR / 16, CT = BC / 16, DT_ = DH / 16;
    const int lane = tid & 31;
    for (int kt = 0; kt < nkt; kt++) {
        const int kb = kt * BC;
        const bool full = kb + BC - 1 <= qb;   // tile fully below diagonal
        {
            const uint4* k4 = (const uint4*)(k + koff);
            const uint4* v4 = (const uint4*)(v + koff);
            uint4* Ks4 = (uint4*)Ks;
            uint4* Vs4 = (uint4*)Vs;
            for (int i = tid; i < BC * V8; i += nthr) {
                int r = i / V8, j = i % V8;
                bool in = kb + r < T;
                long long src = (long long)(kb + r) * V8 + j;
                Ks4[r * Q8 + j] = in ? k4[src] : z4;
                Vs4[r * Q8 + j] = in ? v4[src] : z4;
            }
        }
        __syncthreads();
        // S = Q @ K^T
        for (int t = warp; t < RT * CT; t += nwarp) {
            const int r0 = (t / CT) * 16, c0 = (t % CT) * 16;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
            nvcuda::wmma::fill_fragment(acc, 0.f);
            for (int kk = 0; kk < DH; kk += 16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, bf16, nvcuda::wmma::row_major> af;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, bf16, nvcuda::wmma::col_major> bfr;
                nvcuda::wmma::load_matrix_sync(af, Qs + r0 * QLD + kk, QLD);
                nvcuda::wmma::load_matrix_sync(bfr, Ks + c0 * QLD + kk, QLD);
                nvcuda::wmma::mma_sync(acc, af, bfr, acc);
            }
            nvcuda::wmma::store_matrix_sync(Sfp + r0 * SLD + c0, acc, SLD, nvcuda::wmma::mem_row_major);
        }
        __syncthreads();
        // online softmax: one WARP per query row (shuffle reductions)
        for (int r = warp; r < BR; r += nwarp) {
            const int qg = qb + r;
            if (qg >= T) {
                for (int c = lane; c < BC; c += 32) Pb[r * PLD + c] = zb;
                if (!lane) rrow[r] = 1.f;
                continue;
            }
            float mo = mrow[r], mx = mo;
            for (int c = lane; c < BC; c += 32) {
                float s = (full || kb + c <= qg) ? Sfp[r * SLD + c] * scale : -1e30f;
                mx = fmaxf(mx, s);
            }
            for (int off = 16; off; off >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            float sum = 0.f;
            for (int c = lane; c < BC; c += 32) {
                float p = (full || kb + c <= qg) ? __expf(Sfp[r * SLD + c] * scale - mx) : 0.f;
                Pb[r * PLD + c] = __float2bfloat16(p);
                sum += p;
            }
            for (int off = 16; off; off >>= 1)
                sum += __shfl_xor_sync(0xffffffffu, sum, off);
            if (!lane) {
                float rescale = __expf(mo - mx);     // mo=-1e30 -> 0, no NaN
                lrow[r] = lrow[r] * rescale + sum;
                mrow[r] = mx;
                rrow[r] = rescale;
            }
        }
        __syncthreads();
        // Sfp = P @ V (S already consumed — union buffer)
        for (int t = warp; t < RT * DT_; t += nwarp) {
            const int r0 = (t / DT_) * 16, d0 = (t % DT_) * 16;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
            nvcuda::wmma::fill_fragment(acc, 0.f);
            for (int cc = 0; cc < BC; cc += 16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, bf16, nvcuda::wmma::row_major> af;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, bf16, nvcuda::wmma::row_major> bfr;
                nvcuda::wmma::load_matrix_sync(af, Pb + r0 * PLD + cc, PLD);
                nvcuda::wmma::load_matrix_sync(bfr, Vs + cc * QLD + d0, QLD);
                nvcuda::wmma::mma_sync(acc, af, bfr, acc);
            }
            nvcuda::wmma::store_matrix_sync(Sfp + r0 * OLD + d0, acc, OLD, nvcuda::wmma::mem_row_major);
        }
        __syncthreads();
        // fold rescale into the accumulate: O = O*rescale + P@V
        for (int i = tid; i < BR * DH; i += nthr) {
            int r = i / DH, d = i % DH;
            Oacc[i] = Oacc[i] * rrow[r] + Sfp[r * OLD + d];
        }
        __syncthreads();
    }
    for (int i = tid; i < BR * DH; i += nthr) {
        int r = i / DH;
        if (qb + r < T)
            o[qoff + (long long)(qb + r) * DH + i % DH] = __float2bfloat16(Oacc[i] / lrow[r]);
    }
    if (lse && tid < BR && qb + tid < T)
        lse[(long long)bh * T + qb + tid] = mrow[tid] + logf(lrow[tid]);
}

static size_t flash_smem(int BR, int BC, int DH) {
    int sld = BC + 4, old_ = DH + 4;
    int sfp = BR * ((sld > old_) ? sld : old_);
    return (size_t)(sfp + BR * DH + 3 * BR) * 4
         + (size_t)(BR * (DH + 8) + 2 * BC * (DH + 8) + BR * (BC + 8)) * 2;
}

// ── fused flash attention backward ─────────────────────────────────────
// D[r] = rowsum(dO[r] * O[r]) — one warp per row. bf16 in, fp32 out.
template <typename T>
__global__ void k_rowdot(const T* a, const T* b, float* c, long long rows, int dim) {
    long long r = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= rows) return;
    const int lane = threadIdx.x & 31;
    float acc = 0.f;
    for (int d = lane; d < dim; d += 32)
        acc += ldf(a, r * dim + d) * ldf(b, r * dim + d);
    for (int off = 16; off; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (!lane) c[r] = acc;
}

// One block per (b*h, K-TILE of BC rows) — FA2 layout. K/V load once; inner
// loop over causal q-tiles. Probs recomputed ON-CHIP from LSE (no max pass):
// P = exp(S*scale - lse); dS = P*(dP - D)*scale. dK/dV accumulate in
// PERSISTENT wmma register fragments across the whole q-loop (each warp owns
// fixed tiles), one atomicAdd flush at the end — atomics also perform the GQA
// group-sum across the rep q-heads sharing a kv head. dQ atomicAdds per
// q-tile (fp32). dq/dk/dv are fp32 (cast to bf16 by a following CAST op).
template <int BR, int BC>
__global__ void k_flash_bwd(const bf16* __restrict__ q, const bf16* __restrict__ k,
                            const bf16* __restrict__ v, const bf16* __restrict__ dO,
                            const float* __restrict__ lse, const float* __restrict__ Dd,
                            float* dq, float* dk, float* dv,
                            int T, int DH, int H, int KV, float scale) {
    const int kb = blockIdx.x * BC;
    if (kb >= T) return;
    const int bh = blockIdx.y;
    const int b = bh / H, h = bh % H;
    const int g = h / (H / KV);
    const long long qoff = (long long)bh * T * DH;
    const long long koff = ((long long)b * KV + g) * T * DH;
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int warp = tid >> 5, nwarp = nthr >> 5, lane = tid & 31;

    const int QLD = DH + 8, PLD = BC + 8;
    const int SLD = ((BC > DH) ? BC : DH) + 4;   // fp32 staging stride (union)
    extern __shared__ char smem[];
    float* Sfp  = (float*)smem;                  // BR*SLD: S -> dP -> dq/dk/dv staging
    float* lse_s = Sfp + BR * SLD;               // BR
    float* D_s  = lse_s + BR;                    // BR
    bf16* Ks  = (bf16*)(D_s + BR);               // BC*QLD (resident)
    bf16* Vs  = Ks + BC * QLD;                   // BC*QLD (resident)
    bf16* Qs  = Vs + BC * QLD;                   // BR*QLD
    bf16* dOs = Qs + BR * QLD;                   // BR*QLD
    bf16* Pb  = dOs + BR * QLD;                  // BR*PLD
    bf16* dSb = Pb + BR * PLD;                   // BR*PLD

    const bf16 zb = __float2bfloat16(0.f);
    const int V8 = DH / 8, Q8 = QLD / 8;
    const uint4 z4 = make_uint4(0, 0, 0, 0);
    {
        const uint4* k4 = (const uint4*)(k + koff);
        const uint4* v4 = (const uint4*)(v + koff);
        uint4* Ks4 = (uint4*)Ks; uint4* Vs4 = (uint4*)Vs;
        for (int i = tid; i < BC * V8; i += nthr) {
            int r = i / V8, j = i % V8;
            bool in = kb + r < T;
            long long src = (long long)(kb + r) * V8 + j;
            Ks4[r * Q8 + j] = in ? k4[src] : z4;
            Vs4[r * Q8 + j] = in ? v4[src] : z4;
        }
    }
    const int RT = BR / 16, CT = BC / 16, DT_ = DH / 16;
    const int NKV = CT * DT_;                    // dk/dv output tiles
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> dvA[2], dkA[2];
    nvcuda::wmma::fill_fragment(dvA[0], 0.f); nvcuda::wmma::fill_fragment(dvA[1], 0.f);
    nvcuda::wmma::fill_fragment(dkA[0], 0.f); nvcuda::wmma::fill_fragment(dkA[1], 0.f);
    __syncthreads();

    for (int qb = (kb / BR) * BR; qb < T; qb += BR) {
        {
            const uint4* q4 = (const uint4*)(q + qoff);
            const uint4* o4 = (const uint4*)(dO + qoff);
            uint4* Qs4 = (uint4*)Qs; uint4* dOs4 = (uint4*)dOs;
            for (int i = tid; i < BR * V8; i += nthr) {
                int r = i / V8, j = i % V8;
                bool in = qb + r < T;
                long long src = (long long)(qb + r) * V8 + j;
                Qs4[r * Q8 + j] = in ? q4[src] : z4;
                dOs4[r * Q8 + j] = in ? o4[src] : z4;
            }
            for (int r = tid; r < BR; r += nthr) {
                bool in = qb + r < T;
                lse_s[r] = in ? lse[(long long)bh * T + qb + r] : 0.f;
                D_s[r] = in ? Dd[((long long)b * T + qb + r) * H + h] : 0.f;
            }
        }
        __syncthreads();
        // S = Q @ K^T
        for (int t = warp; t < RT * CT; t += nwarp) {
            const int r0 = (t / CT) * 16, c0 = (t % CT) * 16;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
            nvcuda::wmma::fill_fragment(acc, 0.f);
            for (int kk = 0; kk < DH; kk += 16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, bf16, nvcuda::wmma::row_major> af;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, bf16, nvcuda::wmma::col_major> bfr;
                nvcuda::wmma::load_matrix_sync(af, Qs + r0 * QLD + kk, QLD);
                nvcuda::wmma::load_matrix_sync(bfr, Ks + c0 * QLD + kk, QLD);
                nvcuda::wmma::mma_sync(acc, af, bfr, acc);
            }
            nvcuda::wmma::store_matrix_sync(Sfp + r0 * SLD + c0, acc, SLD, nvcuda::wmma::mem_row_major);
        }
        __syncthreads();
        // P = exp(S*scale - lse), causal mask
        for (int r = warp; r < BR; r += nwarp) {
            const int qg = qb + r;
            if (qg >= T) {
                for (int c = lane; c < BC; c += 32) Pb[r * PLD + c] = zb;
                continue;
            }
            const float l = lse_s[r];
            for (int c = lane; c < BC; c += 32) {
                float p = (kb + c <= qg) ? __expf(Sfp[r * SLD + c] * scale - l) : 0.f;
                Pb[r * PLD + c] = __float2bfloat16(p);
            }
        }
        __syncthreads();
        // dV += P^T @ dO (persistent fragments; k-dim = BR)
        for (int t = warp; t < NKV; t += nwarp) {
            const int s = (t - warp) / nwarp;
            const int j0 = (t / DT_) * 16, d0 = (t % DT_) * 16;
            for (int q0 = 0; q0 < BR; q0 += 16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, bf16, nvcuda::wmma::col_major> af;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, bf16, nvcuda::wmma::row_major> bfr;
                nvcuda::wmma::load_matrix_sync(af, Pb + q0 * PLD + j0, PLD);
                nvcuda::wmma::load_matrix_sync(bfr, dOs + q0 * QLD + d0, QLD);
                nvcuda::wmma::mma_sync(dvA[s], af, bfr, dvA[s]);
            }
        }
        // dP = dO @ V^T -> Sfp (S consumed)
        for (int t = warp; t < RT * CT; t += nwarp) {
            const int r0 = (t / CT) * 16, c0 = (t % CT) * 16;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
            nvcuda::wmma::fill_fragment(acc, 0.f);
            for (int dd = 0; dd < DH; dd += 16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, bf16, nvcuda::wmma::row_major> af;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, bf16, nvcuda::wmma::col_major> bfr;
                nvcuda::wmma::load_matrix_sync(af, dOs + r0 * QLD + dd, QLD);
                nvcuda::wmma::load_matrix_sync(bfr, Vs + c0 * QLD + dd, QLD);
                nvcuda::wmma::mma_sync(acc, af, bfr, acc);
            }
            nvcuda::wmma::store_matrix_sync(Sfp + r0 * SLD + c0, acc, SLD, nvcuda::wmma::mem_row_major);
        }
        __syncthreads();
        // dS = P * (dP - D) * scale
        for (int r = warp; r < BR; r += nwarp) {
            const int qg = qb + r;
            if (qg >= T) {
                for (int c = lane; c < BC; c += 32) dSb[r * PLD + c] = zb;
                continue;
            }
            const float dr = D_s[r];
            for (int c = lane; c < BC; c += 32) {
                float p = __bfloat162float(Pb[r * PLD + c]);
                dSb[r * PLD + c] = __float2bfloat16(p * (Sfp[r * SLD + c] - dr) * scale);
            }
        }
        __syncthreads();
        // dK += dS^T @ Q (persistent fragments)
        for (int t = warp; t < NKV; t += nwarp) {
            const int s = (t - warp) / nwarp;
            const int j0 = (t / DT_) * 16, d0 = (t % DT_) * 16;
            for (int q0 = 0; q0 < BR; q0 += 16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, bf16, nvcuda::wmma::col_major> af;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, bf16, nvcuda::wmma::row_major> bfr;
                nvcuda::wmma::load_matrix_sync(af, dSb + q0 * PLD + j0, PLD);
                nvcuda::wmma::load_matrix_sync(bfr, Qs + q0 * QLD + d0, QLD);
                nvcuda::wmma::mma_sync(dkA[s], af, bfr, dkA[s]);
            }
        }
        // dq_partial = dS @ K -> Sfp staging -> atomicAdd
        for (int t = warp; t < RT * DT_; t += nwarp) {
            const int r0 = (t / DT_) * 16, d0 = (t % DT_) * 16;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc;
            nvcuda::wmma::fill_fragment(acc, 0.f);
            for (int c0 = 0; c0 < BC; c0 += 16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, bf16, nvcuda::wmma::row_major> af;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, bf16, nvcuda::wmma::row_major> bfr;
                nvcuda::wmma::load_matrix_sync(af, dSb + r0 * PLD + c0, PLD);
                nvcuda::wmma::load_matrix_sync(bfr, Ks + c0 * QLD + d0, QLD);
                nvcuda::wmma::mma_sync(acc, af, bfr, acc);
            }
            nvcuda::wmma::store_matrix_sync(Sfp + r0 * SLD + d0, acc, SLD, nvcuda::wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < BR * DH; i += nthr) {
            int r = i / DH, d = i % DH;
            if (qb + r < T)
                atomicAdd(&dq[qoff + (long long)(qb + r) * DH + d], Sfp[r * SLD + d]);
        }
        __syncthreads();
    }
    // flush dV then dK fragments (staging via Sfp)
    for (int pass = 0; pass < 2; pass++) {
        for (int t = warp; t < NKV; t += nwarp) {
            const int s = (t - warp) / nwarp;
            const int j0 = (t / DT_) * 16, d0 = (t % DT_) * 16;
            nvcuda::wmma::store_matrix_sync(Sfp + j0 * SLD + d0, pass ? dkA[s] : dvA[s],
                                            SLD, nvcuda::wmma::mem_row_major);
        }
        __syncthreads();
        float* dst = pass ? dk : dv;
        for (int i = tid; i < BC * DH; i += nthr) {
            int j = i / DH, d = i % DH;
            if (kb + j < T)
                atomicAdd(&dst[koff + (long long)(kb + j) * DH + d], Sfp[j * SLD + d]);
        }
        __syncthreads();
    }
}

static size_t flash_bwd_smem(int BR, int BC, int DH) {
    int sld = ((BC > DH) ? BC : DH) + 4;
    return (size_t)(BR * sld + 2 * BR) * 4
         + (size_t)(2 * BC * (DH + 8) + 2 * BR * (DH + 8) + 2 * BR * (BC + 8)) * 2;
}

// ── API ─────────────────────────────────────────────────────────────────────
API int eng_init() {
    if (cublasCreate(&g_blas)) return 1;
    if (cublasSetMathMode(g_blas, CUBLAS_TF32_TENSOR_OP_MATH)) return 2;
    if (cudaMalloc(&g_ws, 64 << 20)) return 3;
    if (cublasSetWorkspace(g_blas, g_ws, 64 << 20)) return 4;
    if (cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking)) return 5;
    if (cublasSetStream(g_blas, g_stream)) return 6;    // flash kernels use >48KB dynamic shared memory — opt in here (NOT inside
    // graph capture)
    cudaFuncSetAttribute(k_flash_fwd<64, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 << 10);
    cudaFuncSetAttribute(k_flash_fwd<32, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 << 10);
    cudaFuncSetAttribute(k_flash_bwd<64, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 << 10);
    cudaFuncSetAttribute(k_flash_bwd<32, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, 100 << 10);
    for (int i = 0; i < MAX_BUFS; i++) g_bufs[i] = nullptr;
    return 0;
}

API int eng_alloc(int idx, long long nelems, int elsize) {
    if (idx < 0 || idx >= MAX_BUFS) return 1;
    if (cudaMalloc(&g_bufs[idx], nelems * elsize)) return 2;
    return 0;
}

// ── NCCL data parallel (built with -DAXIS_NCCL) ─────────────────────────
API int eng_nccl_id(void* out128) {
#ifdef AXIS_NCCL
    return (int)ncclGetUniqueId((ncclUniqueId*)out128);
#else
    (void)out128; return 103;
#endif
}

API int eng_nccl_init(int rank, int world, const void* id128) {
#ifdef AXIS_NCCL
    return (int)ncclCommInitRank(&g_comm, world, *(const ncclUniqueId*)id128, rank);
#else
    (void)rank; (void)world; (void)id128; return 103;
#endif
}

API int eng_upload(int idx, const float* host, long long nfloats) {
    return (int)cudaMemcpyAsync(g_bufs[idx], host, sizeof(float) * nfloats,
                                cudaMemcpyHostToDevice, g_stream);
}

API int eng_download(int idx, float* host, long long nfloats) {
    if (cudaMemcpyAsync(host, g_bufs[idx], sizeof(float) * nfloats,
                        cudaMemcpyDeviceToHost, g_stream)) return 1;
    return (int)cudaStreamSynchronize(g_stream);
}

static int gemm_ex(const EngOp* o) {
    // row-major via col-major swap (validated recipes), GemmStridedBatchedEx.
    // oa/ob/oc: element offsets (query tiling). beta: 0=overwrite, 1=accumulate.
    const float zero = 0.f;
    float al = o->alpha == 0.f ? 1.f : o->alpha;
    float be = o->beta;
    cudaDataType ab = (o->dt >= 1) ? CUDA_R_16BF : CUDA_R_32F;
    cudaDataType cc = (o->dt == 1) ? CUDA_R_16BF : CUDA_R_32F;
    int esa = (ab == CUDA_R_16BF) ? 2 : 4;
    int esc = (cc == CUDA_R_16BF) ? 2 : 4;
    const void* A = (const char*)g_bufs[o->a] + (long long)o->oa * esa;
    const void* B = (const char*)g_bufs[o->b] + (long long)o->ob * esa;
    void* C = (char*)g_bufs[o->c] + (long long)o->oc * esc;
    int batch = o->batch > 0 ? o->batch : 1;
    long long sa = o->sa, sb = o->sb, sc = o->sc;
    cublasOperation_t opa, opb;
    const void *M1, *M2;
    int ld1, ld2;
    long long s1, s2;
    if (o->tb == 0) {            // C = A @ B
        opa = CUBLAS_OP_N; opb = CUBLAS_OP_N;
        M1 = B; ld1 = o->n; s1 = sb;
        M2 = A; ld2 = o->k; s2 = sa;
    } else if (o->tb == 1) {     // C = A @ B^T, B row-major [n,k]
        opa = CUBLAS_OP_T; opb = CUBLAS_OP_N;
        M1 = B; ld1 = o->k; s1 = sb;
        M2 = A; ld2 = o->k; s2 = sa;
    } else {                     // tb=2: C[m,n] = A[k,m]^T @ B[k,n]
        opa = CUBLAS_OP_N; opb = CUBLAS_OP_T;
        M1 = B; ld1 = o->n; s1 = sb;
        M2 = A; ld2 = o->m; s2 = sa;
    }
    (void)zero;
    return (int)cublasGemmStridedBatchedEx(g_blas, opa, opb,
        o->n, o->m, o->k, &al,
        M1, ab, ld1, s1,
        M2, ab, ld2, s2,
        &be, C, cc, o->n, sc, batch,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

#define EL(T_) reinterpret_cast<T_*>
#define CEL(T_) reinterpret_cast<const T_*>

static int exec_op(const EngOp* op) {
    const int T = 256;
    void* A = op->a >= 0 ? g_bufs[op->a] : nullptr;
    void* B = op->b >= 0 ? g_bufs[op->b] : nullptr;
    void* C = op->c >= 0 ? g_bufs[op->c] : nullptr;
    void* D = op->d >= 0 ? g_bufs[op->d] : nullptr;
    bool h = op->dt == 1;   // bf16 storage
    switch (op->kind) {
        case OP_GEMM: case OP_GEMM_SB:
            return gemm_ex(op);
        case OP_ADD:
            if (h) k_add<<<(op->n + T - 1) / T, T, 0, g_stream>>>(CEL(bf16)(A), CEL(bf16)(B), EL(bf16)(C), op->n);
            else   k_add<<<(op->n + T - 1) / T, T, 0, g_stream>>>(CEL(float)(A), CEL(float)(B), EL(float)(C), op->n);
            break;
        case OP_MUL:
            if (h) k_mul<<<(op->n + T - 1) / T, T, 0, g_stream>>>(CEL(bf16)(A), CEL(bf16)(B), EL(bf16)(C), op->n);
            else   k_mul<<<(op->n + T - 1) / T, T, 0, g_stream>>>(CEL(float)(A), CEL(float)(B), EL(float)(C), op->n);
            break;
        case OP_SILU_MUL:
            if (h) k_silu_mul<<<(op->n + T - 1) / T, T, 0, g_stream>>>(CEL(bf16)(A), CEL(bf16)(B), EL(bf16)(C), op->n);
            else   k_silu_mul<<<(op->n + T - 1) / T, T, 0, g_stream>>>(CEL(float)(A), CEL(float)(B), EL(float)(C), op->n);
            break;
        case OP_RMSNORM:
            if (h) k_rmsnorm<<<op->m, T, T * sizeof(float), g_stream>>>(CEL(bf16)(A), CEL(bf16)(B), EL(bf16)(C), op->m, op->n, op->alpha);
            else   k_rmsnorm<<<op->m, T, T * sizeof(float), g_stream>>>(CEL(float)(A), CEL(float)(B), EL(float)(C), op->m, op->n, op->alpha);
            break;
        case OP_ADAMW: {
            const float* tb_ptr = op->tb >= 0 ? CEL(float)(g_bufs[op->tb]) : nullptr;
            bf16* pbf = op->sa > 0 ? EL(bf16)(g_bufs[op->sa]) : nullptr;   // sa = bf16 mirror
            k_adamw<<<(op->n + T - 1) / T, T, 0, g_stream>>>(EL(float)(A), CEL(float)(B), EL(float)(C), EL(float)(D),
                                                             tb_ptr, pbf, op->alpha, op->beta, op->gamma, op->n);
            break;
        }
        case OP_SCALE:
            if (h) k_scale<<<(op->n + T - 1) / T, T, 0, g_stream>>>(CEL(bf16)(A), EL(bf16)(C), op->alpha, op->n);
            else   k_scale<<<(op->n + T - 1) / T, T, 0, g_stream>>>(CEL(float)(A), EL(float)(C), op->alpha, op->n);
            break;
        case OP_COPY:
            if (h) k_copy<<<(op->n + T - 1) / T, T, 0, g_stream>>>(CEL(bf16)(A), EL(bf16)(C), op->n);
            else   k_copy<<<(op->n + T - 1) / T, T, 0, g_stream>>>(CEL(float)(A), EL(float)(C), op->n);
            break;
        case OP_PERM_0213: {
            long long total = (long long)op->m * op->n * op->k * op->batch;
            int blocks = (int)((total + T - 1) / T);
            if (h) k_perm0213<<<blocks, T, 0, g_stream>>>(CEL(bf16)(A), EL(bf16)(C), op->m, op->n, op->k, op->batch);
            else   k_perm0213<<<blocks, T, 0, g_stream>>>(CEL(float)(A), EL(float)(C), op->m, op->n, op->k, op->batch);
            break;
        }
        case OP_ROPE: {
            long long rows = (long long)op->batch * op->m;
            long long total = rows * op->n;
            int blocks = (int)((total + T - 1) / T);
            if (!op->tb) {
                if (h) k_rope<<<blocks, T, 0, g_stream>>>(CEL(bf16)(A), CEL(float)(B), CEL(float)(D), EL(bf16)(C), op->m, op->n, rows);
                else   k_rope<<<blocks, T, 0, g_stream>>>(CEL(float)(A), CEL(float)(B), CEL(float)(D), EL(float)(C), op->m, op->n, rows);
            } else {
                if (h) k_rope_inv<<<blocks, T, 0, g_stream>>>(CEL(bf16)(A), CEL(float)(B), CEL(float)(D), EL(bf16)(C), op->m, op->n, rows);
                else   k_rope_inv<<<blocks, T, 0, g_stream>>>(CEL(float)(A), CEL(float)(B), CEL(float)(D), EL(float)(C), op->m, op->n, rows);
            }
            break;
        }
        case OP_SOFTMAX_CAUSAL: {
            long long rows = (long long)op->batch * op->m;
            int width = op->n > 0 ? op->n : op->m;   // tiled: n=key range, k=query offset
            if (h) k_softmax_causal<<<(int)rows, T, T * sizeof(float), g_stream>>>(CEL(bf16)(A), EL(bf16)(C), op->m, width, op->k, rows);
            else   k_softmax_causal<<<(int)rows, T, T * sizeof(float), g_stream>>>(CEL(float)(A), EL(float)(C), op->m, width, op->k, rows);
            break;
        }
        case OP_REPEAT_KV: {
            long long total = (long long)op->batch * op->n * op->m * op->k;
            int blocks = (int)((total + T - 1) / T);
            if (h) k_repeat_kv<<<blocks, T, 0, g_stream>>>(CEL(bf16)(A), EL(bf16)(C), op->batch, op->tb, op->n, op->m, op->k);
            else   k_repeat_kv<<<blocks, T, 0, g_stream>>>(CEL(float)(A), EL(float)(C), op->batch, op->tb, op->n, op->m, op->k);
            break;
        }
        case OP_RMSNORM_BWD:
            if (h) k_rmsnorm_bwd<<<op->m, T, 2 * T * sizeof(float), g_stream>>>(CEL(bf16)(A), CEL(bf16)(B), CEL(bf16)(D), EL(bf16)(C), EL(bf16)(g_bufs[op->tb]), op->m, op->n, op->alpha);
            else   k_rmsnorm_bwd<<<op->m, T, 2 * T * sizeof(float), g_stream>>>(CEL(float)(A), CEL(float)(B), CEL(float)(D), EL(float)(C), EL(float)(g_bufs[op->tb]), op->m, op->n, op->alpha);
            break;
        case OP_COLSUM:
            if (op->dt >= 1) k_colsum<<<(op->n + T - 1) / T, T, 0, g_stream>>>(CEL(bf16)(A), EL(float)(C), op->m, op->n);
            else             k_colsum<<<(op->n + T - 1) / T, T, 0, g_stream>>>(CEL(float)(A), EL(float)(C), op->m, op->n);
            break;
        case OP_REPEAT_KV_BWD: {
            long long total = (long long)op->batch * op->tb * op->m * op->k;
            int blocks = (int)((total + T - 1) / T);
            if (h) k_repeat_kv_bwd<<<blocks, T, 0, g_stream>>>(CEL(bf16)(A), EL(bf16)(C), op->batch, op->tb, op->n, op->m, op->k);
            else   k_repeat_kv_bwd<<<blocks, T, 0, g_stream>>>(CEL(float)(A), EL(float)(C), op->batch, op->tb, op->n, op->m, op->k);
            break;
        }
        case OP_SOFTMAX_BWD: {
            long long rows = (long long)op->batch * op->m;
            int width = op->n > 0 ? op->n : op->m;
            if (h) k_softmax_bwd<<<(int)rows, T, T * sizeof(float), g_stream>>>(CEL(bf16)(A), CEL(bf16)(B), EL(bf16)(C), width, rows);
            else   k_softmax_bwd<<<(int)rows, T, T * sizeof(float), g_stream>>>(CEL(float)(A), CEL(float)(B), EL(float)(C), width, rows);
            break;
        }
        case OP_SILU_BWD:
            if (h) k_silu_bwd<<<(op->n + T - 1) / T, T, 0, g_stream>>>(CEL(bf16)(A), CEL(bf16)(B), CEL(bf16)(D), EL(bf16)(C), EL(bf16)(g_bufs[op->tb]), op->n);
            else   k_silu_bwd<<<(op->n + T - 1) / T, T, 0, g_stream>>>(CEL(float)(A), CEL(float)(B), CEL(float)(D), EL(float)(C), EL(float)(g_bufs[op->tb]), op->n);
            break;
        case OP_EMBED: {
            long long total = (long long)op->m * op->n;
            int blocks = (int)((total + T - 1) / T);
            if (h) k_embed<<<blocks, T, 0, g_stream>>>(CEL(bf16)(A), CEL(float)(B), EL(bf16)(C), op->m, op->n);
            else   k_embed<<<blocks, T, 0, g_stream>>>(CEL(float)(A), CEL(float)(B), EL(float)(C), op->m, op->n);
            break;
        }
        case OP_EMBED_BWD: {
            long long total = (long long)op->m * op->n;
            int blocks = (int)((total + T - 1) / T);
            if (op->dt >= 1) k_embed_bwd<<<blocks, T, 0, g_stream>>>(CEL(bf16)(A), CEL(float)(B), EL(float)(C), op->m, op->n);
            else             k_embed_bwd<<<blocks, T, 0, g_stream>>>(CEL(float)(A), CEL(float)(B), EL(float)(C), op->m, op->n);
            break;
        }
        case OP_CE:
            if (h) k_ce<<<op->m, T, T * sizeof(float), g_stream>>>(CEL(bf16)(A), CEL(float)(B), EL(bf16)(C), EL(float)(D), op->m, op->n);
            else   k_ce<<<op->m, T, T * sizeof(float), g_stream>>>(CEL(float)(A), CEL(float)(B), EL(float)(C), EL(float)(D), op->m, op->n);
            break;
        case OP_TICK:
            k_tick<<<1, 1, 0, g_stream>>>(EL(float)(A));
            break;
        case OP_FLASH: {
            // a=q b=k d=v c=o; m=T n=DH k=KV batch=B tb=H; sa=lse buf (0=none);
            // alpha=scale. bf16 only; DH%16==0, DH<=128.
            if (op->dt != 1 || op->n % 16 || op->n > 128) return 101;
            int T_ = op->m, DH_ = op->n;
            float* lse = op->sa > 0 ? EL(float)(g_bufs[op->sa]) : nullptr;
            if (DH_ <= 64) {
                dim3 grid((T_ + 63) / 64, op->batch * op->tb);
                k_flash_fwd<64, 64><<<grid, 256, flash_smem(64, 64, DH_), g_stream>>>(
                    CEL(bf16)(A), CEL(bf16)(B), CEL(bf16)(D), EL(bf16)(C), lse,
                    T_, DH_, op->tb, op->k, op->alpha);
            } else {
                dim3 grid((T_ + 31) / 32, op->batch * op->tb);
                k_flash_fwd<32, 64><<<grid, 256, flash_smem(32, 64, DH_), g_stream>>>(
                    CEL(bf16)(A), CEL(bf16)(B), CEL(bf16)(D), EL(bf16)(C), lse,
                    T_, DH_, op->tb, op->k, op->alpha);
            }
            break;
        }
        case OP_CAST: {
            long long n = ((long long)op->m) * (op->n > 0 ? op->n : 1);
            int blocks = (int)((n + T - 1) / T);
            if (op->tb == 0) k_cast_f2b<<<blocks, T, 0, g_stream>>>(CEL(float)(A), EL(bf16)(C), n);
            else             k_cast_b2f<<<blocks, T, 0, g_stream>>>(CEL(bf16)(A), EL(float)(C), n);
            break;
        }
        case OP_ROWDOT: {
            // c[r] = sum_d a[r,d]*b[r,d]; m=rows n=dim; bf16 in (dt>=1), fp32 out
            long long rows = op->m;
            int blocks = (int)((rows * 32 + 255) / 256);
            if (op->dt >= 1) k_rowdot<<<blocks, 256, 0, g_stream>>>(CEL(bf16)(A), CEL(bf16)(B), EL(float)(C), rows, op->n);
            else             k_rowdot<<<blocks, 256, 0, g_stream>>>(CEL(float)(A), CEL(float)(B), EL(float)(C), rows, op->n);
            break;
        }
        case OP_FLASH_BWD: {
            // a=q b=k d=v c=dO; sa=lse sb=D tb=dqf sc=dkf oa=dvf ob=H;
            // m=T n=DH k=KV batch=B; alpha=scale. dq/dk/dv fp32 (pre-zeroed).
            if (op->dt != 1 || op->n % 16 || op->n > 128) return 102;
            int T_ = op->m, DH_ = op->n, Hh = op->ob;
            float* dqf = EL(float)(g_bufs[op->tb]);
            float* dkf = EL(float)(g_bufs[op->sc]);
            float* dvf = EL(float)(g_bufs[op->oa]);
            const float* lseb = CEL(float)(g_bufs[op->sa]);
            const float* Db = CEL(float)(g_bufs[op->sb]);
            if (DH_ <= 64) {
                dim3 grid((T_ + 63) / 64, op->batch * Hh);
                k_flash_bwd<64, 64><<<grid, 256, flash_bwd_smem(64, 64, DH_), g_stream>>>(
                    CEL(bf16)(A), CEL(bf16)(B), CEL(bf16)(D), CEL(bf16)(C), lseb, Db,
                    dqf, dkf, dvf, T_, DH_, Hh, op->k, op->alpha);
            } else {
                dim3 grid((T_ + 31) / 32, op->batch * Hh);
                k_flash_bwd<32, 32><<<grid, 256, flash_bwd_smem(32, 32, DH_), g_stream>>>(
                    CEL(bf16)(A), CEL(bf16)(B), CEL(bf16)(D), CEL(bf16)(C), lseb, Db,
                    dqf, dkf, dvf, T_, DH_, Hh, op->k, op->alpha);
            }
            break;
        }
        case OP_ALLREDUCE:
            // a = fp32 buffer, n = element count; AVERAGES across ranks
#ifdef AXIS_NCCL
            if (!g_comm) return 104;
            return (int)ncclAllReduce(g_bufs[op->a], g_bufs[op->a], op->n,
                                      ncclFloat, ncclAvg, g_comm, g_stream);
#else
            return 103;
#endif
        case OP_GROUP:
            // tb=0 -> ncclGroupStart, tb=1 -> ncclGroupEnd (batches collectives)
#ifdef AXIS_NCCL
            return (int)(op->tb ? ncclGroupEnd() : ncclGroupStart());
#else
            return 103;
#endif
        default: return 100;
    }
    return 0;
}

API int eng_run_plan(const EngOp* ops, int n_ops, int sync) {
    for (int i = 0; i < n_ops; i++) {
        int rc = exec_op(&ops[i]);
        if (rc) return 1000 + i;
    }
    return sync ? (int)cudaStreamSynchronize(g_stream) : 0;
}

API int eng_capture_plan(const EngOp* ops, int n_ops) {
    if (g_graph_exec) { cudaGraphExecDestroy(g_graph_exec); g_graph_exec = nullptr; }
    if (cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeGlobal)) return 1;
    for (int i = 0; i < n_ops; i++)
        if (exec_op(&ops[i])) { cudaGraph_t junk; cudaStreamEndCapture(g_stream, &junk); return 1000 + i; }
    cudaGraph_t graph;
    if (cudaStreamEndCapture(g_stream, &graph)) return 2;
    if (cudaGraphInstantiate(&g_graph_exec, graph, nullptr, nullptr, 0)) return 3;
    cudaGraphDestroy(graph);
    return 0;
}

API int eng_replay(int times, int sync) {
    if (!g_graph_exec) return 1;
    for (int i = 0; i < times; i++)
        if (cudaGraphLaunch(g_graph_exec, g_stream)) return 2;
    return sync ? (int)cudaStreamSynchronize(g_stream) : 0;
}

API int eng_sync() { return (int)cudaStreamSynchronize(g_stream); }

API int eng_set_tf32(int on) {
    return (int)cublasSetMathMode(g_blas, on ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
}
