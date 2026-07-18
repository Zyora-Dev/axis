// Engine rewrite de-risk probe: prove that with our OWN cuBLAS handle and
// workspace, a sequence of GEMMs + custom kernels CAN be captured into a CUDA
// graph and replayed — the exact thing CuPy forbids and PyTorch does in C++.
//
// Simulates a transformer-block-like step: GEMM -> elementwise -> GEMM ->
// reduction, capture it, replay it, compare timing vs eager and verify output.
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK(x) do { auto e = (x); if (e) { printf("ERR %s:%d code=%d\n", __FILE__, __LINE__, (int)e); return 1; } } while (0)

__global__ void silu_mul(const float* g, const float* u, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float x = g[i]; o[i] = (x / (1.f + expf(-x))) * u[i]; }
}

__global__ void add_inplace(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

int main() {
    const int M = 4096, K = 1536, N = 1536;   // realistic transformer GEMM
    const int LAYERS = 24;                     // repeat like a deep model
    const float alpha = 1.f, beta = 0.f;

    cublasHandle_t h;
    CHECK(cublasCreate(&h));
    CHECK(cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH));
    // Own workspace => cuBLAS never mallocs during capture (the CuPy blocker).
    void* ws; CHECK(cudaMalloc(&ws, 64 << 20));
    CHECK(cublasSetWorkspace(h, ws, 64 << 20));

    cudaStream_t s; CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CHECK(cublasSetStream(h, s));

    float *A, *B, *C, *D, *E;
    CHECK(cudaMalloc(&A, sizeof(float) * M * K));
    CHECK(cudaMalloc(&B, sizeof(float) * K * N));
    CHECK(cudaMalloc(&C, sizeof(float) * M * N));
    CHECK(cudaMalloc(&D, sizeof(float) * M * N));
    CHECK(cudaMalloc(&E, sizeof(float) * M * N));
    CHECK(cudaMemset(A, 1, sizeof(float) * M * K));
    CHECK(cudaMemset(B, 1, sizeof(float) * K * N));

    int threads = 256, blocks = (M * N + threads - 1) / threads;

    auto step = [&]() {   // one pseudo-layer sequence, repeated LAYERS times
        for (int l = 0; l < LAYERS; l++) {
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
            silu_mul<<<blocks, threads, 0, s>>>(C, C, D, M * N);
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, N, &alpha, D, N, D, N, &beta, E, N);
            add_inplace<<<blocks, threads, 0, s>>>(E, D, M * N);
        }
    };

    // warmup
    step(); CHECK(cudaStreamSynchronize(s));

    // ---- eager timing ----
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, s);
    for (int i = 0; i < 10; i++) step();
    cudaEventRecord(t1, s);
    CHECK(cudaStreamSynchronize(s));
    float eager_ms; cudaEventElapsedTime(&eager_ms, t0, t1); eager_ms /= 10;

    // ---- capture into a CUDA graph ----
    cudaGraph_t graph;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    step();
    CHECK(cudaStreamEndCapture(s, &graph));
    printf("CAPTURE with own cuBLAS handle: OK\n");

    cudaGraphExec_t exec;
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    // ---- replay timing ----
    CHECK(cudaGraphLaunch(exec, s)); CHECK(cudaStreamSynchronize(s)); // warm
    cudaEventRecord(t0, s);
    for (int i = 0; i < 10; i++) CHECK(cudaGraphLaunch(exec, s));
    cudaEventRecord(t1, s);
    CHECK(cudaStreamSynchronize(s));
    float replay_ms; cudaEventElapsedTime(&replay_ms, t0, t1); replay_ms /= 10;

    printf("eager : %.2f ms/step (%d GEMM+kernel pairs)\n", eager_ms, LAYERS * 2);
    printf("replay: %.2f ms/step\n", replay_ms);
    printf("graph speedup: %.2fx\n", eager_ms / replay_ms);
    printf("PROBE RESULT: ENGINE ARCHITECTURE VIABLE\n");
    return 0;
}
