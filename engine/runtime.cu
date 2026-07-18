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
    OP_GEMM_SB = 8,    // strided-batched: c[i][m,n] = alpha * a[i][m,k] @ b[i][k,n or n,k^T]
    OP_PERM_0213 = 9,  // [d0,d1,d2,d3] -> [d0,d2,d1,d3]  (dims m,n,k,batch = d0..d3)
    OP_ROPE = 10,      // half-split rotary on [rows, dh]; cos=b, sin=d, T=m, dh=n
    OP_SOFTMAX_CAUSAL = 11, // rows = batch*T, row r masks cols > r%T; width n=T
    OP_REPEAT_KV = 12, // [B,KV,T,dh] -> [B,H,T,dh], grouped (h -> h/rep)
};

typedef struct {
    int   kind;
    int   a, b, c, d;   // buffer indices (op-specific roles)
    int   m, n, k;      // dims
    int   batch, tb;    // batch count, transpose-B flag
    int   sa, sb, sc;   // per-batch strides (elements)
    float alpha, beta;  // scalars (eps, lr, scale, ...)
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
// Fused AdamW: m,v moments + decoupled weight decay. alpha=lr, beta packs
// nothing fancy in phase 1 — fixed betas/eps for the skeleton.
__global__ void k_adamw(float* p, const float* g, float* m, float* v,
                        float lr, float wd, float bc1, float bc2, int n) {
    const float b1 = 0.9f, b2 = 0.95f, eps = 1e-8f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float mi = b1 * m[i] + (1.f - b1) * gi;
    float vi = b2 * v[i] + (1.f - b2) * gi * gi;
    m[i] = mi; v[i] = vi;
    float mh = mi / bc1, vh = vi / bc2;
    float pi = p[i] * (1.f - lr * wd);
    p[i] = pi - lr * mh / (sqrtf(vh) + eps);
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
        case OP_ADAMW:    k_adamw<<<(op->n + T - 1) / T, T, 0, g_stream>>>(A, B, C, D, op->alpha, op->beta, op->m / 1e6f, op->k / 1e6f, op->n); break;
        case OP_SCALE:    k_scale<<<(op->n + T - 1) / T, T, 0, g_stream>>>(A, C, op->alpha, op->n); break;
        case OP_COPY:     k_copy<<<(op->n + T - 1) / T, T, 0, g_stream>>>(A, C, op->n); break;
        case OP_GEMM_SB: {
            // row-major batched: C[i][m,n] = alpha * A[i][m,k] @ (B[i][k,n] or B[i][n,k]^T)
            const float zero = 0.f;
            float al = op->alpha == 0.f ? 1.f : op->alpha;
            if (!op->tb) {
                return (int)cublasSgemmStridedBatched(g_blas, CUBLAS_OP_N, CUBLAS_OP_N,
                    op->n, op->m, op->k, &al,
                    B, op->n, op->sb, A, op->k, op->sa, &zero, C, op->n, op->sc, op->batch);
            } else {
                // B row-major [n,k], want A @ B^T
                return (int)cublasSgemmStridedBatched(g_blas, CUBLAS_OP_T, CUBLAS_OP_N,
                    op->n, op->m, op->k, &al,
                    B, op->k, op->sb, A, op->k, op->sa, &zero, C, op->n, op->sc, op->batch);
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
            k_rope<<<(int)((total + T - 1) / T), T, 0, g_stream>>>(A, B, D, C, op->m, op->n, rows);
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
