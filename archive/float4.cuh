template <
    const int BM, 
    const int BN, 
    const int BK, 
    const int TM, 
    const int TN, 
    const int BLOCK_SIZE
>
__global__ void sgemm_float4_v1_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    const int r0 = blockIdx.y * BM;
    const int c0 = blockIdx.x * BN;

    A += r0 * K;
    B += c0;
    C += r0 * N + c0;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int thread_row_a = threadIdx.x / (BK / 4);
    const int thread_col_a = threadIdx.x % (BK / 4);
    const int stride_a = max(1, BLOCK_SIZE / (BK / 4));     // stride=0 防呆

    const int thread_row_b = threadIdx.x / (BN / 4);
    const int thread_col_b = threadIdx.x % (BN / 4);
    const int stride_b = max(1, BLOCK_SIZE / (BN / 4));     // stride=0 防呆

    const int thread_row_c = threadIdx.x / (BN / TN);
    const int thread_col_c = threadIdx.x % (BN / TN);

    float val[TM * TN] = {0.0f};
    float reg_a[TM] = {0.0f};
    float reg_b[TN] = {0.0f};

    for (int k = 0; k < K; k += BK) {
        const int c_a = thread_col_a; 
        for (int i = 0; i < BM; i += stride_a) { 
            const int r_a = thread_row_a + i; 
            // As[r_a * BK + c_a] = (r_a < M && c_a < K) ? A[r_a * K + c_a] : 0.0f; 
            if (r_a < BM) {
                reinterpret_cast<float4*>(&As[r_a * BK + c_a * 4])[0] = reinterpret_cast<const float4*>(&A[r_a * K + c_a * 4])[0];
            }
        } 
        const int c_b = thread_col_b; 
        for (int i = 0; i < BK; i += stride_b) { 
            const int r_b = thread_row_b + i; 
            // Bs[r_b * BN + c_b] = (r_b < K && c_b < N) ? B[r_b * N + c_b] : 0.0f; 
            if (r_b < BK) {
                reinterpret_cast<float4*>(&Bs[r_b * BN + c_b * 4])[0] = reinterpret_cast<const float4*>(&B[r_b * N + c_b * 4])[0];
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
__global__ void sgemm_float4_v2_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
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
    // ---- load As: [BM x BK] ---- (float4 vectorized, 行优先展开)
    for (int idx4 = threadIdx.x; idx4 < (BM * BK) / 4; idx4 += BLOCK_SIZE) {
        int elem = idx4 * 4;      // 线性下标的 float 起点
        int r    = elem / BK;     // 行
        int c    = elem % BK;     // 列（4 连续元素起点）
        // 从全局内存一次取 4 个 float，写入 shared 的 4 个连续位置
        reinterpret_cast<float4*>(As)[idx4] =
            reinterpret_cast<const float4*>(A + r * K + c)[0];
    }

    // ---- load Bs: [BK x BN] ---- (float4 vectorized, 行优先展开)
    for (int idx4 = threadIdx.x; idx4 < (BK * BN) / 4; idx4 += BLOCK_SIZE) {
        int elem = idx4 * 4;
        int r    = elem / BN;     // 行（K 维）
        int c    = elem % BN;     // 列（C/N 维）
        reinterpret_cast<float4*>(Bs)[idx4] =
            reinterpret_cast<const float4*>(B + r * N + c)[0];
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

// template <
//     const int BM,
//     const int BN,
//     const int BK,
//     const int TM,
//     const int TN,
//     const int BLOCK_SIZE
// >
// __global__ void sgemm_float4_v5_kernel(const float* __restrict__ A,
//                                        const float* __restrict__ B,
//                                        float* __restrict__ C,
//                                        int M, int N, int K) {
//     // ===== 约束 =====
//     static_assert(BM % TM == 0, "BM % TM == 0");
//     static_assert(BN % TN == 0, "BN % TN == 0");
//     static_assert(TN % 4 == 0 && TM % 4 == 0, "TM,TN must be multiples of 4");
//     static_assert(BK % 4 == 0, "BK must be multiple of 4 for float4 loads");

//     constexpr int ROW_GROUPS = BM / TM;
//     static_assert(BLOCK_SIZE % ROW_GROUPS == 0, "BLOCK_SIZE % (BM/TM) == 0");
//     constexpr int TCOL = BLOCK_SIZE / ROW_GROUPS;

//     constexpr int VEC = 4;
//     constexpr int G   = TN / VEC;
//     static_assert(TCOL * VEC * G == BN, "thread mapping mismatch");

//     // ===== AsT/Bs 布局（含 padding）=====
//     constexpr int APAD   = (BM % 32 == 0) ? 4 : ((4 - (BM % 4)) % 4);
//     constexpr int APITCH = BM + APAD;
//     static_assert(APITCH % 4 == 0, "APITCH must be multiple of 4");

//     // ===== 显式静态共享内存：双缓冲开两个面板 =====
//     __shared__ __align__(16) float AsT[2][BK * APITCH]; // [buf][kk*APITCH + row]
//     __shared__ __align__(16) float Bs [2][BK * BN];     // [buf][kk*BN + col]

//     // block tile 起点
//     const int r0 = blockIdx.y * BM;
//     const int c0 = blockIdx.x * BN;
//     const float* __restrict__ A0 = A + r0 * K;
//     const float* __restrict__ B0 = B + c0;
//     float* __restrict__ C0 = C + r0 * N + c0;

//     // ===== 线程划分（与 v3 相同）=====
//     const int thread_row_a = threadIdx.x / (BK / VEC);
//     const int thread_col_a = threadIdx.x % (BK / VEC);
//     const int stride_a     = max(1, BLOCK_SIZE / (BK / VEC));

//     const int thread_row_b = threadIdx.x / (BN / VEC);
//     const int thread_col_b = threadIdx.x % (BN / VEC);
//     const int stride_b     = max(1, BLOCK_SIZE / (BN / VEC));

//     const int thread_row_c = threadIdx.x / TCOL;
//     const int thread_col_c = threadIdx.x % TCOL;

//     // ===== 累加寄存器 =====
//     float val[TM * TN];
//     #pragma unroll
//     for (int t = 0; t < TM * TN; ++t) val[t] = 0.0f;

//     const int tiles = (K + BK - 1) / BK;

//     // -------- Prologue：把第 0 块放到 buf=0 --------
//     {
//         const int c_a = thread_col_a;
//         for (int i = 0; i < BM; i += stride_a) {
//             const int r_a = thread_row_a + i;
//             if (r_a < BM) {
//                 const float4 a4 = reinterpret_cast<const float4*>(&A0[r_a * K + c_a * 4])[0];
//                 const int kk0 = c_a * 4;
//                 AsT[0][(kk0 + 0) * APITCH + r_a] = a4.x;
//                 AsT[0][(kk0 + 1) * APITCH + r_a] = a4.y;
//                 AsT[0][(kk0 + 2) * APITCH + r_a] = a4.z;
//                 AsT[0][(kk0 + 3) * APITCH + r_a] = a4.w;
//             }
//         }
//         const int c_b = thread_col_b;
//         for (int i = 0; i < BK; i += stride_b) {
//             const int r_b = thread_row_b + i;
//             if (r_b < BK) {
//                 reinterpret_cast<float4*>(&Bs[0][r_b * BN + c_b * 4])[0] =
//                     reinterpret_cast<const float4*>(&B0[r_b * N + c_b * 4])[0];
//             }
//         }
//     }
//     __syncthreads(); // tile0 就绪

//     int buf = 0; // 当前读面板

//     // ================= 主循环（每轮仅一次 __syncthreads） =================
//     for (int t = 0; t < tiles; ++t) {
//         const int cur = buf;          // 读
//         const int nxt = buf ^ 1;      // 写

//         // ---- 计算：读 AsT[cur] / Bs[cur] ----
//         #pragma unroll
//         for (int kk = 0; kk < BK; ++kk) {
//             float reg_a[TM];
//             float reg_b[TN];

//             const int row_base = thread_row_c * TM;

//             #pragma unroll
//             for (int i = 0; i < TM; i += 4) {
//                 reinterpret_cast<float4*>(&reg_a[i])[0] =
//                     *reinterpret_cast<const float4*>(&AsT[cur][kk * APITCH + (row_base + i)]);
//             }

//             #pragma unroll
//             for (int g = 0; g < G; ++g) {
//                 const int c4 = (g * TCOL + thread_col_c) * 4;
//                 const float4 b4 =
//                     reinterpret_cast<const float4*>(&Bs[cur][kk * BN + c4])[0];
//                 reg_b[g * 4 + 0] = b4.x;
//                 reg_b[g * 4 + 1] = b4.y;
//                 reg_b[g * 4 + 2] = b4.z;
//                 reg_b[g * 4 + 3] = b4.w;
//             }

//             #pragma unroll
//             for (int i = 0; i < TM; ++i) {
//                 const float a = reg_a[i];
//                 #pragma unroll
//                 for (int j = 0; j < TN; ++j) {
//                     val[i * TN + j] += a * reg_b[j];
//                 }
//             }
//         }

//         // ---- 预取下一块到 AsT[nxt]/Bs[nxt]（与上面计算重叠；统一在回环处同步）----
//         if (t + 1 < tiles) {
//             const int kbase = (t + 1) * BK;

//             const int c_a = thread_col_a;
//             for (int i = 0; i < BM; i += stride_a) {
//                 const int r_a = thread_row_a + i;
//                 if (r_a < BM) {
//                     const float4 a4 = reinterpret_cast<const float4*>(
//                         &A0[r_a * K + (kbase + c_a * 4)]
//                     )[0];
//                     const int kk0 = c_a * 4;
//                     AsT[nxt][(kk0 + 0) * APITCH + r_a] = a4.x;
//                     AsT[nxt][(kk0 + 1) * APITCH + r_a] = a4.y;
//                     AsT[nxt][(kk0 + 2) * APITCH + r_a] = a4.z;
//                     AsT[nxt][(kk0 + 3) * APITCH + r_a] = a4.w;
//                 }
//             }

//             const int c_b = thread_col_b;
//             for (int i = 0; i < BK; i += stride_b) {
//                 const int r_b = thread_row_b + i;
//                 if (r_b < BK) {
//                     reinterpret_cast<float4*>(&Bs[nxt][r_b * BN + c_b * 4])[0] =
//                         reinterpret_cast<const float4*>(
//                             &B0[(kbase + r_b) * N + c_b * 4]
//                         )[0];
//                 }
//             }
//         }

//         __syncthreads(); // 唯一同步：结束本轮计算并确保 nxt 完成
//         buf ^= 1;        // 仅 1 条 XOR 指令，切换面板
//     }

//     // ===== 写回（float4）=====
//     #pragma unroll
//     for (int i = 0; i < TM; ++i) {
//         const int r = thread_row_c * TM + i;
//         if (r0 + r >= M) continue;

//         #pragma unroll
//         for (int g = 0; g < G; ++g) {
//             const int c4 = (g * TCOL + thread_col_c) * 4;
//             if (c0 + c4 + 3 >= N) continue;

//             float4 v = {
//                 val[i * TN + g * 4 + 0],
//                 val[i * TN + g * 4 + 1],
//                 val[i * TN + g * 4 + 2],
//                 val[i * TN + g * 4 + 3]
//             };
//             reinterpret_cast<float4*>(&C0[r * N + c4])[0] = v;
//         }
//     }
// }