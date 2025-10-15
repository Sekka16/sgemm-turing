#include "utils.cuh"

template <
    const int BM, 
    const int BN, 
    const int BK, 
    const int TM, 
    const int TN, 
    const int BLOCK_SIZE
>
__global__ void sgemm_float4_v1_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int ROW_GROUPS = BM / TM;
    constexpr int TCOL = BLOCK_SIZE / ROW_GROUPS;

    constexpr int VEC = 4;

    constexpr int APAD = 4;
    constexpr int APITCH = BM + APAD;

    __shared__ __align__(16) float AsT[BK * APITCH];
    __shared__ __align__(16) float Bs[BK * BN];

    const int r0 = blockIdx.y * BM;
    const int c0 = blockIdx.x * BN;

    const float* A0 = A + r0 * K;
    const float* B0 = B + c0;
    float* C0 = C + r0 * N + c0;

    const int thread_row_a = threadIdx.x / (BK / VEC);
    const int thread_col_a = threadIdx.x % (BK / VEC);
    const int stride_a     = BLOCK_SIZE / (BK / VEC);

    const int thread_row_b = threadIdx.x / (BN / VEC);
    const int thread_col_b = threadIdx.x % (BN / VEC);
    const int stride_b     = BLOCK_SIZE / (BN / VEC);

    const int thread_row_c = threadIdx.x / TCOL;  // [0, ROW_GROUPS)
    const int thread_col_c = threadIdx.x % TCOL;  // [0, TCOL)

    float val[TM * TN] = { 0.0f };

    const float* A_ptr = A0;
    const float* B_ptr = B0;

    for (int kk = 0; kk < K; kk += BK) {
        const int c_a = thread_col_a; 
        for (int i = 0; i < BM; i += stride_a) {
            const int r_a = thread_row_a + i;
            if (r_a < BM) {
                const float4 a4 =
                    reinterpret_cast<const float4*>(&A_ptr[r_a * K + c_a * VEC])[0];
                const int kk0 = c_a * VEC;
                AsT[(kk0 + 0) * APITCH + r_a] = a4.x;
                AsT[(kk0 + 1) * APITCH + r_a] = a4.y;
                AsT[(kk0 + 2) * APITCH + r_a] = a4.z;
                AsT[(kk0 + 3) * APITCH + r_a] = a4.w;
            }
        }

        const int c_b = thread_col_b;
        for (int i = 0; i < BK; i += stride_b) {
            const int r_b = thread_row_b + i;
            if (r_b < BK) {
                reinterpret_cast<float4*>(&Bs[r_b * BN + c_b * VEC])[0] =
                    reinterpret_cast<const float4*>(&B_ptr[r_b * N + c_b * VEC])[0];
            }
        }

        __syncthreads();

        for (int kk = 0; kk < BK; ++kk) {
            float reg_a[TM];
            float reg_b[TN];

            const int row_base = thread_row_c * TM;
            for (int i = 0; i < TM; i += VEC) {
                reinterpret_cast<float4*>(&reg_a[i])[0] =
                    *reinterpret_cast<const float4*>(&AsT[kk * APITCH + (row_base + i)]);
            }

            for (int i = 0; i < TN; i += VEC) {
                reinterpret_cast<float4*>(&reg_b[i])[0] = 
                    *reinterpret_cast<const float4*>(&Bs[kk * BN + (i * TCOL + thread_col_c * VEC)]);
            }

            for (int i = 0; i < TM; ++i) {
                const float a = reg_a[i];
                for (int j = 0; j < TN; ++j) {
                    val[i * TN + j] += a * reg_b[j];
                }
            }
        }

        __syncthreads();
        A_ptr += BK;
        B_ptr += BK * N;
    }

    for (int i = 0; i < TM; ++i) {
        const int r = thread_row_c * TM + i;
        if (r0 + r >= M) continue;

        for (int j = 0; j < TN; j += 4) {
            const int c4 = j * TCOL + thread_col_c * VEC;
            if (c0 + c4 + 3 >= N) continue;

            reinterpret_cast<float4*>(&C0[r * N + c4])[0] = 
                *reinterpret_cast<float4*>(&val[i * TN + j]);
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
__global__ void sgemm_float4_v2_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K) {
    constexpr int ROW_GROUPS = BM / TM;
    constexpr int TCOL = BLOCK_SIZE / ROW_GROUPS;

    constexpr int VEC = 4;

    constexpr int APAD   = (BM % 32 == 0) ? 4 : ((4 - (BM % 4)) % 4);
    constexpr int APITCH = BM + APAD;

    __shared__ __align__(16) float AsT[BK * APITCH];
    __shared__ __align__(16) float Bs [BK * BN];

    const int r0 = blockIdx.y * BM;
    const int c0 = blockIdx.x * BN;

    const float* __restrict__ A0 = A + r0 * K;
    const float* __restrict__ B0 = B + c0;
    float* __restrict__ C0 = C + r0 * N + c0;

    const int thread_row_c = threadIdx.x / TCOL;
    const int thread_col_c = threadIdx.x % TCOL;

    float val[TM * TN] = {0.0f};

    const float* A_ptr = A0;
    const float* B_ptr = B0;

    constexpr int A_groups = (BM * BK) / VEC;
    constexpr int B_groups = (BK * BN) / VEC;

    for (int kk = 0; kk < K; kk += BK) {
        for (int t = threadIdx.x; t < A_groups; t += BLOCK_SIZE) {
            const int r_a = t / (BK / VEC);
            const int c4  = t % (BK / VEC);

            const float4 a4 = reinterpret_cast<const float4*>(&A_ptr[r_a * K + c4 * VEC])[0];
            const int kk0 = c4 * VEC;
            AsT[(kk0 + 0) * APITCH + r_a] = a4.x;
            AsT[(kk0 + 1) * APITCH + r_a] = a4.y;
            AsT[(kk0 + 2) * APITCH + r_a] = a4.z;
            AsT[(kk0 + 3) * APITCH + r_a] = a4.w;
        }

        for (int t = threadIdx.x; t < B_groups; t += BLOCK_SIZE) {
            const int kk  = t / (BN / VEC);
            const int c4  = t % (BN / VEC);

            reinterpret_cast<float4*>(&Bs[kk * BN + c4 * VEC])[0] = 
                reinterpret_cast<const float4*>(&B_ptr[kk * N + c4 * VEC])[0];
        }
        __syncthreads();

        for (int kk = 0; kk < BK; ++kk) {
            float reg_a[TM];
            float reg_b[TN];

            const int row_base = thread_row_c * TM;
            for (int i = 0; i < TM; i += VEC) {
                reinterpret_cast<float4*>(&reg_a[i])[0] =
                    *reinterpret_cast<const float4*>(&AsT[kk * APITCH + (row_base + i)]);
            }

            for (int i = 0; i < TN; i += VEC) {
                const int c4 = i * TCOL + thread_col_c * VEC;
                reinterpret_cast<float4*>(&reg_b[i])[0] = 
                    *reinterpret_cast<float4*>(&Bs[kk * BN + c4]);
            }

            for (int i = 0; i < TM; ++i) {
                const float a = reg_a[i];
                for (int j = 0; j < TN; ++j) {
                    val[i * TN + j] += a * reg_b[j];
                }
            }
        }

        __syncthreads();
        A_ptr += BK;
        B_ptr += BK * N;
    }

    for (int i = 0; i < TM; ++i) {
        const int r = thread_row_c * TM + i;
        if (r0 + r >= M) continue;

        for (int j = 0; j < TN; j += 4) {
            const int c4 = j * TCOL + thread_col_c * VEC;
            if (c0 + c4 + 3 >= N) continue;
            reinterpret_cast<float4*>(&C0[r * N + c4])[0] = 
                *reinterpret_cast<float4*>(&val[i * TN + j]);
        }
    }
}

template <typename Config>
void sgemm_float4_v1(const float* a, const float* b, float* c, int M, int N, int K) {
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
    sgemm_float4_v1_kernel<BM, BN, BK, TM, TN, BLOCK_SIZE><<<grid, block>>>(a, b, c, M, N, K);
    KERNEL_TIMER_STOP("sgemm_float4_v1");
    CUDA_CHECK_POST_LAUNCH();
    CUDA_CHECK_POST_SYNC();
}

template <typename Config>
void sgemm_float4_v2(const float* a, const float* b, float* c, int M, int N, int K) {
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
    sgemm_float4_v2_kernel<BM, BN, BK, TM, TN, BLOCK_SIZE><<<grid, block>>>(a, b, c, M, N, K);
    KERNEL_TIMER_STOP("sgemm_float4_v2");
    CUDA_CHECK_POST_LAUNCH();
    CUDA_CHECK_POST_SYNC();
}