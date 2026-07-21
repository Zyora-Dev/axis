// HIP portability layer for the Axis engine runtime (build: -DAXIS_HIP).
// CUDA names map to their HIP / hipBLAS equivalents. TF32 is NVIDIA-only, so
// math-mode calls become no-ops. Flash (WMMA) kernels are excluded on HIP v1
// — the fully-validated tiled attention path (GEMMs + block-reduction
// kernels, no warp-size assumptions) is the ROCm path until rocWMMA kernels
// are validated on real hardware.
#pragma once
#define HIPBLAS_V2                       // cuBLAS-style Ex API (hipDataType etc.)
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hipblas/hipblas.h>

// ── runtime ──────────────────────────────────────────────────────────────────
#define cudaMalloc                  hipMalloc
#define cudaMemcpyAsync             hipMemcpyAsync
#define cudaMemcpyHostToDevice      hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost      hipMemcpyDeviceToHost
#define cudaStream_t                hipStream_t
#define cudaStreamCreateWithFlags   hipStreamCreateWithFlags
#define cudaStreamNonBlocking       hipStreamNonBlocking
#define cudaStreamSynchronize       hipStreamSynchronize
#define cudaStreamBeginCapture      hipStreamBeginCapture
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamEndCapture        hipStreamEndCapture
#define cudaGraph_t                 hipGraph_t
#define cudaGraphExec_t             hipGraphExec_t
#define cudaGraphInstantiate        hipGraphInstantiate
#define cudaGraphExecDestroy        hipGraphExecDestroy
#define cudaGraphDestroy            hipGraphDestroy
#define cudaGraphLaunch             hipGraphLaunch

// ── blas ─────────────────────────────────────────────────────────────────────
#define cublasHandle_t              hipblasHandle_t
#define cublasCreate                hipblasCreate
#define cublasSetStream             hipblasSetStream
#define cublasSetWorkspace          hipblasSetWorkspace
#define cublasOperation_t           hipblasOperation_t
#define CUBLAS_OP_N                 HIPBLAS_OP_N
#define CUBLAS_OP_T                 HIPBLAS_OP_T
#define cublasGemmStridedBatchedEx  hipblasGemmStridedBatchedEx
#define cudaDataType                hipDataType
#define CUDA_R_16BF                 HIP_R_16BF
#define CUDA_R_32F                  HIP_R_32F
#define CUBLAS_COMPUTE_32F          HIPBLAS_COMPUTE_32F
#define CUBLAS_GEMM_DEFAULT         HIPBLAS_GEMM_DEFAULT

// TF32 does not exist on AMD — the calls succeed as no-ops.
#define cublasSetMathMode(h, m)     HIPBLAS_STATUS_SUCCESS
#define CUBLAS_TF32_TENSOR_OP_MATH  0
#define CUBLAS_DEFAULT_MATH         0
