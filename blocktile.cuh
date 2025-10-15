#include "utils.cuh"

template < 
    const int BM, 
    const int BN, 
    const int BK, 
    const int TM, 
    const int TN, 
    const int BLOCK_SIZE
>
__global__ void sgemm_block_tile_v1_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    const int r0 = blockIdx.y * BM;
    const int c0 = blockIdx.x * BN;

    A += r0 * K;
    B += c0;
    C += r0 * N + c0;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int thread_row_a = threadIdx.x / BK;
    const int thread_col_a = threadIdx.x % BK;
    const int stride_a = max(1, BLOCK_SIZE / BK);

    const int thread_row_b = threadIdx.x / BN;
    const int thread_col_b = threadIdx.x % BN;
    const int stride_b = max(1, BLOCK_SIZE / BN);

    const int thread_row_c = threadIdx.x / (BN / TN);
    const int thread_col_c = threadIdx.x % (BN / TN);

    float val[TM * TN] = {0.0f};
    float reg_a[TM] = {0.0f};
    float reg_b[TN] = {0.0f};

    for (int k = 0; k < K; k += BK) {
        const int c_a = thread_col_a; 
        for (int i = 0; i < BM; i += stride_a) { 
            const int r_a = thread_row_a + i; 
            if (r_a < BM) {
                As[r_a * BK + c_a] = (r_a < M && c_a < K) ? A[r_a * K + c_a] : 0.0f;
            }
        } 
        const int c_b = thread_col_b; 
        for (int i = 0; i < BK; i += stride_b) { 
            const int r_b = thread_row_b + i; 
            if (r_b < BK) {
                Bs[r_b * BN + c_b] = (r_b < K && c_b < N) ? B[r_b * N + c_b] : 0.0f;
            }
        } 
        __syncthreads(); 
        A += BK; 
        B += BK * N; 
        for (int kk = 0; kk < BK; kk++) { 
            for (int i = 0; i < TM; i++) { 
                reg_a[i] = As[(i * (BM / TM) + thread_row_c) * BK + kk]; 
            } 
            for (int i = 0; i < TN; i++) { 
                reg_b[i] = Bs[kk * BN + (i * (BN / TN) + thread_col_c)]; 
            } 
            for (int i = 0; i < TM; i++) { 
                for (int j = 0; j < TN; j++) { 
                    val[i * TN + j] += reg_a[i] * reg_b[j]; 
                } 
            } 
        } 
        __syncthreads(); 
    } 
    for (int i = 0; i < TM; i++) {
        const int r = i * (BM / TM) + thread_row_c; 
        for (int j = 0; j < TN; j++) {
            const int c = j * (BN / TN) + thread_col_c; 
            if (r < M && c < N) {
                C[r * N + c] = val[i * TN + j]; 
            }
        }
    }
}

template <
    const int BM,
    const int BN,
    const int BK,
    const int TM,
    const int TN,
    const int BLOCK_SIZE
>
__global__ void sgemm_block_tile_v2_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int r0 = blockIdx.y * BM;
    const int c0 = blockIdx.x * BN;

    A += r0 * K;
    B += c0;
    C += r0 * N + c0;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // thread layout c
    const int thread_row_c = threadIdx.x / (BN / TN);
    const int thread_col_c = threadIdx.x % (BN / TN);

    float val[TM * TN] = {0.0f};
    float reg_a[TM] = {0.0f};
    float reg_b[TN] = {0.0f};

    for (int k = 0; k < K; k += BK) {
        // ---- load As: [BM x BK] ----
        for (int idx = threadIdx.x; idx < BM * BK; idx += BLOCK_SIZE) {
            int r = idx / BK;     // [0, BM)
            int c = idx % BK;     // [0, BK)
            int gr = r0 + r;      // global row in A/C
            int gc = k + c;       // global col in A/K dimension
            As[idx] = (gr < M && gc < K) ? A[r * K + c] : 0.0f;
        }

        // ---- load Bs: [BK x BN] ----
        for (int idx = threadIdx.x; idx < BK * BN; idx += BLOCK_SIZE) {
            int r = idx / BN;     // [0, BK)
            int c = idx % BN;     // [0, BN)
            int gr = k + r;       // global row in B/K dimension
            int gc = c0 + c;      // global col in B/C
            Bs[idx] = (gr < K && gc < N) ? B[r * N + c] : 0.0f;
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        // As[BM * BK]
        for (int kk = 0; kk < BK; kk++) {
            for (int i = 0; i < TM; i++) {
                reg_a[i] = As[(i * (BM / TM) + thread_row_c) * BK + kk];
            }
            for (int i = 0; i < TN; i++) {
                reg_b[i] = Bs[kk * BN + (i * (BN / TN) + thread_col_c)];
            }
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    val[i * TN + j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; i++) {
        const int r = i * (BM / TM) + thread_row_c;
        for (int j = 0; j < TN; j++) {
            const int c = j * (BN / TN) + thread_col_c;
            if (r < M && c < N) {
                C[r * N + c] = val[i * TN + j];
            }
        }
    }
}

template <typename Config>
void sgemm_block_tile_v1(const float* a, const float* b, float* c, int M, int N, int K) {
    const int BM = Config::BM;
    const int BN = Config::BN;
    const int BK = Config::BK;
    const int TM = Config::TM;
    const int TN = Config::TN;
    const int BLOCK_SIZE = Config::BLOCK_SIZE;

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    CUDA_CHECK_PRE();
    KERNEL_TIMER_START();
    sgemm_block_tile_v1_kernel<BM, BN, BK, TM, TN, BLOCK_SIZE><<<grid, block>>>(a, b, c, M, N, K);
    KERNEL_TIMER_STOP("sgemm_block_tile_v1");
    CUDA_CHECK_POST_LAUNCH();
    CUDA_CHECK_POST_SYNC();
}

template <typename Config>
void sgemm_block_tile_v2(const float* a, const float* b, float* c, int M, int N, int K) {
    const int BM = Config::BM;
    const int BN = Config::BN;
    const int BK = Config::BK;
    const int TM = Config::TM;
    const int TN = Config::TN;
    const int BLOCK_SIZE = Config::BLOCK_SIZE;

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    CUDA_CHECK_PRE();
    KERNEL_TIMER_START();
    sgemm_block_tile_v2_kernel<BM, BN, BK, TM, TN, BLOCK_SIZE><<<grid, block>>>(a, b, c, M, N, K);
    KERNEL_TIMER_STOP("sgemm_block_tile_v2");
    CUDA_CHECK_POST_LAUNCH();
    CUDA_CHECK_POST_SYNC();
}