#include "utils.cuh"
#include <cublas_v2.h>

void sgemm_cublas(const float* a, const float* b, float* c, int M, int N, int K) {
    // A: [M x K] row-major
    // B: [K x N] row-major  
    // C: [M x N] row-major
    // 计算目标：C = A * B
    // 等价转换到列主序接口：计算 C^T = B^T * A^T
    // 对应到 cuBLAS：
    //   op(A) = B^T  -> 传入 b, trans = T, 原始B维度 [K x N]
    //   op(B) = A^T  -> 传入 a, trans = T, 原始A维度 [M x K]
    //   C^T 维度 [N x M]

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: Failed to create handle (code: %d)\n", status);
        exit(EXIT_FAILURE);
    }

    // （可选）确保 alpha/beta 用主机指针
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

    CUDA_CHECK_PRE();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    KERNEL_TIMER_START();
    status = cublasSgemm(
        handle,
        CUBLAS_OP_N, // A := B^T (以列主序视图，故这里不再转置)
        CUBLAS_OP_N, // B := A^T (以列主序视图，故这里不再转置)
        /*m=*/N,     // rows of op(A)   = N
        /*n=*/M,     // cols of op(B)   = M
        /*k=*/K,     // cols of op(A)   = K = rows of op(B)
        &alpha,
        /*A=*/b, /*lda=*/N, // B(row K×N) 视作 col (N×K) ⇒ lda=N
        /*B=*/a, /*ldb=*/K, // A(row M×K) 视作 col (K×M) ⇒ ldb=K
        &beta,
        /*C=*/c, /*ldc=*/N // C(row M×N) 视作 col (N×M) ⇒ ldc=N
    );
    KERNEL_TIMER_STOP("sgemm_cublas");

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: sgemm failed (code: %d)\n", status);
        cublasDestroy(handle);
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK_POST_LAUNCH();
    CUDA_CHECK_POST_SYNC();

    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: Failed to destroy handle (code: %d)\n", status);
        exit(EXIT_FAILURE);
    }
}