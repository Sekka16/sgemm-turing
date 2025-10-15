#include "utils.cuh"

template < 
    const int BM, 
    const int BN, 
    const int BK, 
    const int TM, 
    const int TN, 
    const int BLOCK_SIZE
>
__global__ void sgemm_double_buffer_v1_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    const int r0 = blockIdx.y * BM;
    const int c0 = blockIdx.x * BN;

    A += r0 * K;
    B += c0;
    C += r0 * N + c0;

    // 双缓冲
    __shared__ float As0[BM * BK];
    __shared__ float As1[BM * BK];
    __shared__ float Bs0[BK * BN];
    __shared__ float Bs1[BK * BN];

    const int thread_row_a = threadIdx.x / BK;
    const int thread_col_a = threadIdx.x % BK;
    const int stride_a = max(1, BLOCK_SIZE / BK);

    const int thread_row_b = threadIdx.x / BN;
    const int thread_col_b = threadIdx.x % BN;
    const int stride_b = max(1, BLOCK_SIZE / BN);

    const int thread_row_c = threadIdx.x / (BN / TN);
    const int thread_col_c = threadIdx.x % (BN / TN);

    float val[TM * TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    // ===== Prologue：预取第 0 块 -> As0 / Bs0 =====
    {
        const int k0 = 0;
        const int c_a = thread_col_a;
        for (int i = 0; i < BM; i += stride_a) {
            const int r_a = thread_row_a + i;
            const int gr = r0 + r_a;
            const int gc = k0 + c_a;
            if (r_a < BM) {
                As0[r_a * BK + c_a] = (gr < M && gc < K) ? A[r_a * K + c_a] : 0.0f;
            }
        }
        const int c_b = thread_col_b;
        for (int i = 0; i < BK; i += stride_b) {
            const int r_b = thread_row_b + i;
            const int gr = k0 + r_b;
            const int gc = c0 + c_b;
            if (r_b < BK) {
                Bs0[r_b * BN + c_b] = (gr < K && gc < N) ? B[r_b * N + c_b] : 0.0f;
            }
        }
    }
    __syncthreads(); // 确保 tile0 可读

    // ===== 主循环：每轮仅一次同步（轮末） =====
    for (int k = 0, tile_id = 0; k < K; k += BK, ++tile_id) {
        // 当前/下一缓冲选择
        float* As_cur = (tile_id & 1) ? As1 : As0;
        float* Bs_cur = (tile_id & 1) ? Bs1 : Bs0;
        float* As_nxt = (tile_id & 1) ? As0 : As1;
        float* Bs_nxt = (tile_id & 1) ? Bs0 : Bs1;

        // ---- 预取下一 tile -> 空闲缓冲（无同步；轮末统一栅栏）----
        const int k_next = k + BK;
        if (k_next < K) {
            const int c_a = thread_col_a;
            for (int i = 0; i < BM; i += stride_a) {
                const int r_a = thread_row_a + i;
                const int gr = r0 + r_a;
                const int gc = k_next + c_a;
                if (r_a < BM) {
                    As_nxt[r_a * BK + c_a] = (gr < M && gc < K) ? A[r_a * K + c_a + k_next] : 0.0f;
                }
            }
            const int c_b = thread_col_b;
            for (int i = 0; i < BK; i += stride_b) {
                const int r_b = thread_row_b + i;
                const int gr = k_next + r_b;
                const int gc = c0 + c_b;
                if (r_b < BK) {
                    Bs_nxt[r_b * BN + c_b] = (gr < K && gc < N) ? B[(r_b + k_next) * N + c_b] : 0.0f;
                }
            }
        }

        // ---- 计算：消耗 As_cur / Bs_cur ----
        for (int kk = 0; kk < BK; ++kk) {
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As_cur[(i * (BM / TM) + thread_row_c) * BK + kk];
            }
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = Bs_cur[kk * BN + (j * (BN / TN) + thread_col_c)];
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    val[i * TN + j] += reg_a[i] * reg_b[j];
                }
            }
        }

        // ---- 轮末唯一一次同步：等待预取完成 + 切换缓冲的栅栏 ----
        __syncthreads();
    }

    // ===== 写回 C =====
    for (int i = 0; i < TM; ++i) {
        const int r = i * (BM / TM) + thread_row_c;
        for (int j = 0; j < TN; ++j) {
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
__global__ void sgemm_double_buffer_v2_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int r0 = blockIdx.y * BM;
    const int c0 = blockIdx.x * BN;

    A += r0 * K;
    B += c0;
    C += r0 * N + c0;

    // 双缓冲
    __shared__ float As0[BM * BK];
    __shared__ float As1[BM * BK];
    __shared__ float Bs0[BK * BN];
    __shared__ float Bs1[BK * BN];

    // thread layout c（保持原有变量名）
    const int thread_row_c = threadIdx.x / (BN / TN);
    const int thread_col_c = threadIdx.x % (BN / TN);

    float val[TM * TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    // ===== Prologue：预取第0块 -> As0/Bs0 =====
    {
        const int k0 = 0;
        for (int idx = threadIdx.x; idx < BM * BK; idx += BLOCK_SIZE) {
            int r = idx / BK;   // [0,BM)
            int c = idx % BK;   // [0,BK)
            int gr = r0 + r;
            int gc = k0 + c;
            As0[idx] = (gr < M && gc < K) ? A[r * K + c] : 0.0f;
        }
        for (int idx = threadIdx.x; idx < BK * BN; idx += BLOCK_SIZE) {
            int r = idx / BN;   // [0,BK)
            int c = idx % BN;   // [0,BN)
            int gr = k0 + r;
            int gc = c0 + c;
            Bs0[idx] = (gr < K && gc < N) ? B[r * N + c] : 0.0f;
        }
    }
    __syncthreads(); // 保证tile0可见

    // ===== 主循环（每轮仅一次同步）=====
    for (int k = 0, tile_id = 0; k < K; k += BK, ++tile_id) {
        // 当前/下一缓冲选择
        float* As_cur = (tile_id & 1) ? As1 : As0;
        float* Bs_cur = (tile_id & 1) ? Bs1 : Bs0;
        float* As_nxt = (tile_id & 1) ? As0 : As1;
        float* Bs_nxt = (tile_id & 1) ? Bs0 : Bs1;

        // --- 预取下一tile到空闲缓冲（无同步；下一轮前统一同步）---
        const int k_next = k + BK;
        if (k_next < K) {
            for (int idx = threadIdx.x; idx < BM * BK; idx += BLOCK_SIZE) {
                int r = idx / BK;
                int c = idx % BK;
                int gr = r0 + r;
                int gc = k_next + c;
                As_nxt[idx] = (gr < M && gc < K) ? A[r * K + c + k_next] : 0.0f;
            }
            for (int idx = threadIdx.x; idx < BK * BN; idx += BLOCK_SIZE) {
                int r = idx / BN;
                int c = idx % BN;
                int gr = k_next + r;
                int gc = c0 + c;
                Bs_nxt[idx] = (gr < K && gc < N) ? B[(r + k_next) * N + c] : 0.0f;
            }
        }

        // --- 计算：消费当前缓冲 ---
        for (int kk = 0; kk < BK; ++kk) {
            // A: (BM x BK) 被 thread_row_c 切成 BM/TM 个“行组”
            for (int i = 0; i < TM; ++i) {
                reg_a[i] = As_cur[(i * (BM / TM) + thread_row_c) * BK + kk];
            }
            // B: (BK x BN) 被 thread_col_c 切成 BN/TN 个“列组”
            for (int j = 0; j < TN; ++j) {
                reg_b[j] = Bs_cur[kk * BN + (j * (BN / TN) + thread_col_c)];
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    val[i * TN + j] += reg_a[i] * reg_b[j];
                }
            }
        }

        // --- 每轮唯一一次同步 ---
        // 作用1：保证本轮刚才发起的 As_nxt/Bs_nxt 预取全部完成
        // 作用2：作为内存栅栏，安全切换缓冲，防止下一轮读到未完成的数据
        __syncthreads();
    }

    // ===== 写回 =====
    for (int i = 0; i < TM; ++i) {
        const int r = i * (BM / TM) + thread_row_c;
        for (int j = 0; j < TN; ++j) {
            const int c = j * (BN / TN) + thread_col_c;
            if (r < M && c < N) {
                C[r * N + c] = val[i * TN + j];
            }
        }
    }
}


template <typename Config>
void sgemm_double_buffer_v1(const float* a, const float* b, float* c, int M, int N, int K) {
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
    sgemm_double_buffer_v1_kernel<BM, BN, BK, TM, TN, BLOCK_SIZE><<<grid, block>>>(a, b, c, M, N, K);
    KERNEL_TIMER_STOP("sgemm_double_buffer_v1");
    CUDA_CHECK_POST_LAUNCH();
    CUDA_CHECK_POST_SYNC();
}

template <typename Config>
void sgemm_double_buffer_v2(const float* a, const float* b, float* c, int M, int N, int K) {
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
    sgemm_double_buffer_v2_kernel<BM, BN, BK, TM, TN, BLOCK_SIZE><<<grid, block>>>(a, b, c, M, N, K);
    KERNEL_TIMER_STOP("sgemm_double_buffer_v2");
    CUDA_CHECK_POST_LAUNCH();
    CUDA_CHECK_POST_SYNC();
}