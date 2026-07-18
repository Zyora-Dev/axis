// Axis engine runtime — Phase 1 skeleton.
//
// A minimal, dependency-free C/CUDA runtime that executes a whole training
// step as an EXECUTION PLAN (array of op descriptors over a buffer table),
// with our own cuBLAS handle + workspace so the entire plan can be captured
// into a CUDA graph and replayed with zero Python involvement.
//
// Exposed as a plain C ABI for ctypes (no pybind, dependency-light).
//
// Build: nvcc -O3 -arch=sm_80 --shared -Xcompiler -fPIC runtime.cu -lcublas -o libaxeng.so
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define API extern "C" __attribute__((visibility("default")))
#define MAX_BUFS 4096

// ── op kinds ────────────────────────────────────────────────────────────────
enum OpKind {
    OP_GEMM = 0,       // c[m,n] = a[m,k] @ b[k,n]      (row-major)
    OP_ADD = 1,        // c = a + b                      (n elements)
    OP_MUL = 2,        // c = a * b
    OP_SILU_MUL = 3,   // c = silu(a) * b
    OP_RMSNORM = 4,    // c = a / sqrt(mean(a^2,-1)+eps) * b   (m rows, n cols)
    OP_ADAMW = 5,      // fused AdamW: p,m,v updated from g (n elements)
    OP_SCALE = 6,      // c = a * alpha
    OP_COPY = 7,       // c = a
    OP_GEMM_SB = 8,    // strided-batched: tb=0 NN, tb=1 NT (a@b^T, b=[n,k]), tb=2 TN (a[k,m]^T@b[k,n])
    OP_PERM_0213 = 9,  // [d0,d1,d2,d3] -> [d0,d2,d1,d3]  (dims m,n,k,batch = d0..d3)
    OP_ROPE = 10,      // half-split rotary on [rows, dh]; cos=b, sin=d, T=m, dh=n; tb=1 inverse
    OP_SOFTMAX_CAUSAL = 11, // rows = batch*T, row r masks cols > r%T; width n=T
    OP_REPEAT_KV = 12, // [B,KV,T,dh] -> [B,H,T,dh], grouped (h -> h/rep)
    OP_RMSNORM_BWD = 13, // a=x b=w d=g -> c=dx, tb=tmp buf (g*x*inv for dw colsum)
    OP_COLSUM = 14,    // c[n] = sum_rows a[m,n]
    OP_REPEAT_KV_BWD = 15, // group-sum: [B,H,T,dh] -> [B,KV,T,dh]
    OP_SOFTMAX_BWD = 16,   // a=p b=dp -> c = p*(dp - rowsum(dp*p)); rows=batch*m, width m
    OP_SILU_BWD = 17,  // a=g b=u d=grad -> c=dg, tb=du buffer
    OP_EMBED = 18,     // a=table[V,D] b=ids(float)[N] -> c=out[N,D]; m=N n=D
    OP_EMBED_BWD = 19, // a=g[N,D] b=ids -> atomicAdd into c=dTable; m=N n=D
    OP_CE = 20,        // a=logits[N,V] b=targets(float)[N] -> c=dlogits, d=loss(1); m=N n=V
    OP_TICK = 21,      // a: t-buffer (1 float) += 1
};

typedef struct {
    int   kind;
    int   a, b, c, d;   // buffer indices (op-specific roles)
    int   m, n, k;      // dims
    int   batch, tb;    // batch count, transpose flag / aux buffer
    int   sa, sb, sc;   // per-batch strides (elements)
    float alpha, beta, gamma;  // scalars (eps, lr, scale, ...)
} EngOp;

// ── global state ────────────────────────────────────────────────────────────
static cublasHandle_t g_blas = nullptr;
static cudaStream_t   g_stream = nullptr;
static void*          g_ws = nullptr;
static float*         g_bufs[MAX_BUFS];
static cudaGraphExec_t g_graph_exec = nullptr;

// ── kernels ─────────────────────────────────────────────────────────────────
__global__ void k_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
__global__ void k_mul(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}
__global__ void k_silu_mul(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = a[i]; c[i] = (x / (1.f + expf(-x))) * b[i]; }
}
__global__ void k_scale(const float* a, float* c, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * alpha;
}
__global__ void k_copy(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i];
}
// One block per row; fp32 accumulation; weight w broadcast over rows.
__global__ void k_rmsnorm(const float* x, const float* w, float* o,
                          int rows, int dim, float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    extern __shared__ float sh[];
    float acc = 0.f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float v = x[r * dim + j];
        acc += v * v;
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] / dim + eps);
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        o[r * dim + j] = x[r * dim + j] * inv * w[j];
}
// Fused AdamW. Two modes:
//  tbuf == nullptr: folded constants (alpha=lr*sqrt(bc2)/bc1, beta=lr*wd,
//                   gamma=eps*sqrt(bc2)) — fixed t.
//  tbuf != nullptr: t read from device buffer (OP_TICK increments it), bias
//                   correction computed ON DEVICE — exact under CUDA graphs.
//                   alpha=lr, beta=lr*wd, gamma=eps.
__global__ void k_adamw(float* p, const float* g, float* m, float* v,
                        const float* tbuf, float alpha, float beta, float gamma, int n) {
    const float b1 = 0.9f, b2 = 0.95f;
    __shared__ float s_alpha, s_gamma;
    if (threadIdx.x == 0) {
        if (tbuf) {
            float t = tbuf[0];
            float bc1 = 1.f - powf(b1, t);
            float bc2 = 1.f - powf(b2, t);
            s_alpha = alpha * sqrtf(bc2) / bc1;
            s_gamma = gamma * sqrtf(bc2);
        } else {
            s_alpha = alpha; s_gamma = gamma;
        }
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float mi = b1 * m[i] + (1.f - b1) * gi;
    float vi = b2 * v[i] + (1.f - b2) * gi * gi;
    m[i] = mi; v[i] = vi;
    float pi = p[i] * (1.f - beta);
    p[i] = pi - s_alpha * mi / (sqrtf(vi) + s_gamma);
}

__global__ void k_tick(float* t) { if (threadIdx.x == 0 && blockIdx.x == 0) t[0] += 1.f; }

// rmsnorm backward: per row two reductions (ss, gwx); dx out; tmp = g*x*inv for dw.
__global__ void k_rmsnorm_bwd(const float* x, const float* w, const float* g,
                              float* dx, float* tmp, int rows, int dim, float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    extern __shared__ float sh[];
    const float* xr = x + (long long)r * dim;
    const float* gr = g + (long long)r * dim;
    float ss = 0.f, gwx = 0.f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        ss += xr[j] * xr[j];
        gwx += gr[j] * w[j] * xr[j];
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
        float dxj = gr[j] * w[j] * inv - xr[j] * coef;
        dx[(long long)r * dim + j] = dxj;
        tmp[(long long)r * dim + j] = gr[j] * xr[j] * inv;
    }
}

__global__ void k_colsum(const float* a, float* c, int rows, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float acc = 0.f;
    for (int r = 0; r < rows; r++) acc += a[(long long)r * n + j];
    c[j] = acc;
}

__global__ void k_repeat_kv_bwd(const float* g, float* o, int B, int KV, int H,
                                int T, int dh) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)B * KV * T * dh;
    if (i >= total) return;
    int j = i % dh;
    long long r = i / dh;
    int t = r % T; r /= T;
    int kv = r % KV; r /= KV;
    int b = (int)r;
    int rep = H / KV;
    float acc = 0.f;
    for (int q = 0; q < rep; q++)
        acc += g[(((long long)b * H + kv * rep + q) * T + t) * dh + j];
    o[i] = acc;
}

__global__ void k_softmax_bwd(const float* p, const float* dp, float* ds,
                              int width, long long rows) {
    long long r = blockIdx.x;
    if (r >= rows) return;
    extern __shared__ float sh[];
    const float* pr = p + r * width;
    const float* dpr = dp + r * width;
    float dot = 0.f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) dot += dpr[j] * pr[j];
    sh[threadIdx.x] = dot; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    dot = sh[0];
    for (int j = threadIdx.x; j < width; j += blockDim.x)
        ds[r * width + j] = pr[j] * (dpr[j] - dot);
}

__global__ void k_silu_bwd(const float* g, const float* u, const float* grad,
                           float* dg, float* du, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = g[i];
    float sig = 1.f / (1.f + expf(-x));
    float silu = x * sig;
    float dsilu = sig * (1.f + x * (1.f - sig));
    dg[i] = grad[i] * u[i] * dsilu;
    du[i] = grad[i] * silu;
}

__global__ void k_embed(const float* table, const float* ids, float* o, int N, int D) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)N * D) return;
    int r = (int)(i / D), j = (int)(i % D);
    o[i] = table[(long long)((int)ids[r]) * D + j];
}

__global__ void k_embed_bwd(const float* g, const float* ids, float* dT, int N, int D) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)N * D) return;
    int r = (int)(i / D), j = (int)(i % D);
    atomicAdd(&dT[(long long)((int)ids[r]) * D + j], g[i]);
}

// Fused CE: per row (block) max + sumexp; dlogits = (softmax - onehot)/N;
// atomicAdd mean NLL into loss[0].
__global__ void k_ce(const float* logits, const float* tgt, float* dlogits,
                     float* loss, int N, int V) {
    int r = blockIdx.x;
    if (r >= N) return;
    extern __shared__ float sh[];
    const float* lr = logits + (long long)r * V;
    float mx = -1e30f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) mx = fmaxf(mx, lr[j]);
    sh[threadIdx.x] = mx; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    mx = sh[0]; __syncthreads();
    float se = 0.f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) se += expf(lr[j] - mx);
    sh[threadIdx.x] = se; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    se = sh[0];
    int t = (int)tgt[r];
    float inv = 1.f / N;
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        float p = expf(lr[j] - mx) / se;
        dlogits[(long long)r * V + j] = (p - (j == t ? 1.f : 0.f)) * inv;
    }
    if (threadIdx.x == 0)
        atomicAdd(loss, (logf(se) - (lr[t] - mx)) * inv);
}

// [d0,d1,d2,d3] -> [d0,d2,d1,d3]
__global__ void k_perm0213(const float* x, float* o, int d0, int d1, int d2, int d3) {
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

// Half-split RoPE on rows of [.., T, dh]: row index r -> t = r % T.
// cos/sin: [T, dh/2]. Matches HF rotate_half exactly.
__global__ void k_rope(const float* x, const float* cs, const float* sn,
                       float* o, int T, int dh, long long rows) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = rows * dh;
    if (i >= total) return;
    int j = i % dh;
    long long r = i / dh;
    int t = (int)(r % T);
    int half = dh / 2;
    const float* xr = x + r * dh;
    if (j < half) {
        o[i] = xr[j] * cs[t * half + j] - xr[j + half] * sn[t * half + j];
    } else {
        int jj = j - half;
        o[i] = xr[jj] * sn[t * half + jj] + xr[j] * cs[t * half + jj];
    }
}

// Inverse rotation (rope backward): dx1 = g1*c + g2*s ; dx2 = -g1*s + g2*c
__global__ void k_rope_inv(const float* g, const float* cs, const float* sn,
                           float* o, int T, int dh, long long rows) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = rows * dh;
    if (i >= total) return;
    int j = i % dh;
    long long r = i / dh;
    int t = (int)(r % T);
    int half = dh / 2;
    const float* gr = g + r * dh;
    if (j < half) {
        o[i] = gr[j] * cs[t * half + j] + gr[j + half] * sn[t * half + j];
    } else {
        int jj = j - half;
        o[i] = -gr[jj] * sn[t * half + jj] + gr[j] * cs[t * half + jj];
    }
}

// Causal row softmax: rows = batch*T, width T; row r masks cols > (r % T).
__global__ void k_softmax_causal(const float* x, float* o, int T, long long rows) {
    long long r = blockIdx.x;
    if (r >= rows) return;
    int t = (int)(r % T);
    extern __shared__ float sh[];
    const float* xr = x + r * T;
    float mx = -1e30f;
    for (int j = threadIdx.x; j <= t; j += blockDim.x) mx = fmaxf(mx, xr[j]);
    sh[threadIdx.x] = mx; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    mx = sh[0]; __syncthreads();
    float acc = 0.f;
    for (int j = threadIdx.x; j <= t; j += blockDim.x) acc += expf(xr[j] - mx);
    sh[threadIdx.x] = acc; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float denom = sh[0];
    float* orow = o + r * T;
    for (int j = threadIdx.x; j < T; j += blockDim.x)
        orow[j] = (j <= t) ? expf(xr[j] - mx) / denom : 0.f;
}

// [B,KV,T,dh] -> [B,H,T,dh], grouped repeat: out head h reads kv head h/rep.
__global__ void k_repeat_kv(const float* x, float* o, int B, int KV, int H,
                            int T, int dh) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)B * H * T * dh;
    if (i >= total) return;
    int j = i % dh;
    long long r = i / dh;
    int t = r % T; r /= T;
    int h = r % H; r /= H;
    int b = (int)r;
    int rep = H / KV;
    o[i] = x[(((long long)b * KV + h / rep) * T + t) * dh + j];
}

// ── API ─────────────────────────────────────────────────────────────────────
API int eng_init() {
    if (cublasCreate(&g_blas)) return 1;
    if (cublasSetMathMode(g_blas, CUBLAS_TF32_TENSOR_OP_MATH)) return 2;
    if (cudaMalloc(&g_ws, 64 << 20)) return 3;
    if (cublasSetWorkspace(g_blas, g_ws, 64 << 20)) return 4;
    if (cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking)) return 5;
    if (cublasSetStream(g_blas, g_stream)) return 6;
    for (int i = 0; i < MAX_BUFS; i++) g_bufs[i] = nullptr;
    return 0;
}

API int eng_alloc(int idx, long long nfloats) {
    if (idx < 0 || idx >= MAX_BUFS) return 1;
    if (cudaMalloc(&g_bufs[idx], sizeof(float) * nfloats)) return 2;
    return 0;
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

static int exec_op(const EngOp* op) {
    const int T = 256;
    float* A = g_bufs[op->a];
    float* B = op->b >= 0 ? g_bufs[op->b] : nullptr;
    float* C = op->c >= 0 ? g_bufs[op->c] : nullptr;
    float* D = op->d >= 0 ? g_bufs[op->d] : nullptr;
    switch (op->kind) {
        case OP_GEMM: {
            // row-major C[m,n] = A[m,k] @ B[k,n]  == col-major C^T = B^T A^T
            const float one = 1.f, zero = 0.f;
            return (int)cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    op->n, op->m, op->k, &one,
                                    B, op->n, A, op->k, &zero, C, op->n);
        }
        case OP_ADD:      k_add<<<(op->n + T - 1) / T, T, 0, g_stream>>>(A, B, C, op->n); break;
        case OP_MUL:      k_mul<<<(op->n + T - 1) / T, T, 0, g_stream>>>(A, B, C, op->n); break;
        case OP_SILU_MUL: k_silu_mul<<<(op->n + T - 1) / T, T, 0, g_stream>>>(A, B, C, op->n); break;
        case OP_RMSNORM:  k_rmsnorm<<<op->m, T, T * sizeof(float), g_stream>>>(A, B, C, op->m, op->n, op->alpha); break;
        case OP_ADAMW: {
            const float* tb_ptr = op->tb >= 0 ? g_bufs[op->tb] : nullptr;
            k_adamw<<<(op->n + T - 1) / T, T, 0, g_stream>>>(A, B, C, D, tb_ptr, op->alpha, op->beta, op->gamma, op->n);
            break;
        }
        case OP_SCALE:    k_scale<<<(op->n + T - 1) / T, T, 0, g_stream>>>(A, C, op->alpha, op->n); break;
        case OP_COPY:     k_copy<<<(op->n + T - 1) / T, T, 0, g_stream>>>(A, C, op->n); break;
        case OP_GEMM_SB: {
            const float zero = 0.f;
            float al = op->alpha == 0.f ? 1.f : op->alpha;
            if (op->tb == 0) {
                return (int)cublasSgemmStridedBatched(g_blas, CUBLAS_OP_N, CUBLAS_OP_N,
                    op->n, op->m, op->k, &al,
                    B, op->n, op->sb, A, op->k, op->sa, &zero, C, op->n, op->sc, op->batch);
            } else if (op->tb == 1) {   // c = a @ b^T, b row-major [n,k]
                return (int)cublasSgemmStridedBatched(g_blas, CUBLAS_OP_T, CUBLAS_OP_N,
                    op->n, op->m, op->k, &al,
                    B, op->k, op->sb, A, op->k, op->sa, &zero, C, op->n, op->sc, op->batch);
            } else {                    // tb=2: c[m,n] = a[k,m]^T @ b[k,n]
                return (int)cublasSgemmStridedBatched(g_blas, CUBLAS_OP_N, CUBLAS_OP_T,
                    op->n, op->m, op->k, &al,
                    B, op->n, op->sb, A, op->m, op->sa, &zero, C, op->n, op->sc, op->batch);
            }
        }
        case OP_PERM_0213: {
            long long total = (long long)op->m * op->n * op->k * op->batch;
            k_perm0213<<<(int)((total + T - 1) / T), T, 0, g_stream>>>(A, C, op->m, op->n, op->k, op->batch);
            break;
        }
        case OP_ROPE: {
            long long rows = (long long)op->batch * op->m;   // batch=B*H, m=T
            long long total = rows * op->n;
            if (!op->tb)
                k_rope<<<(int)((total + T - 1) / T), T, 0, g_stream>>>(A, B, D, C, op->m, op->n, rows);
            else
                k_rope_inv<<<(int)((total + T - 1) / T), T, 0, g_stream>>>(A, B, D, C, op->m, op->n, rows);
            break;
        }
        case OP_SOFTMAX_CAUSAL: {
            long long rows = (long long)op->batch * op->m;   // batch=B*H, m=T
            k_softmax_causal<<<(int)rows, T, T * sizeof(float), g_stream>>>(A, C, op->m, rows);
            break;
        }
        case OP_REPEAT_KV: {
            long long total = (long long)op->batch * op->n * op->m * op->k;  // B*H*T*dh
            k_repeat_kv<<<(int)((total + T - 1) / T), T, 0, g_stream>>>(A, C, op->batch, op->tb, op->n, op->m, op->k);
            break;
        }
        case OP_RMSNORM_BWD:
            k_rmsnorm_bwd<<<op->m, T, 2 * T * sizeof(float), g_stream>>>(A, B, D, C, g_bufs[op->tb], op->m, op->n, op->alpha);
            break;
        case OP_COLSUM:
            k_colsum<<<(op->n + T - 1) / T, T, 0, g_stream>>>(A, C, op->m, op->n);
            break;
        case OP_REPEAT_KV_BWD: {
            long long total = (long long)op->batch * op->tb * op->m * op->k;  // B*KV*T*dh
            k_repeat_kv_bwd<<<(int)((total + T - 1) / T), T, 0, g_stream>>>(A, C, op->batch, op->tb, op->n, op->m, op->k);
            break;
        }
        case OP_SOFTMAX_BWD: {
            long long rows = (long long)op->batch * op->m;
            k_softmax_bwd<<<(int)rows, T, T * sizeof(float), g_stream>>>(A, B, C, op->m, rows);
            break;
        }
        case OP_SILU_BWD:
            k_silu_bwd<<<(op->n + T - 1) / T, T, 0, g_stream>>>(A, B, D, C, g_bufs[op->tb], op->n);
            break;
        case OP_EMBED: {
            long long total = (long long)op->m * op->n;
            k_embed<<<(int)((total + T - 1) / T), T, 0, g_stream>>>(A, B, C, op->m, op->n);
            break;
        }
        case OP_EMBED_BWD: {
            long long total = (long long)op->m * op->n;
            k_embed_bwd<<<(int)((total + T - 1) / T), T, 0, g_stream>>>(A, B, C, op->m, op->n);
            break;
        }
        case OP_CE:
            k_ce<<<op->m, T, T * sizeof(float), g_stream>>>(A, B, C, D, op->m, op->n);
            break;
        case OP_TICK:
            k_tick<<<1, 1, 0, g_stream>>>(A);
            break;
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
