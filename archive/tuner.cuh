#pragma once
#include "test.cuh"

static inline void print_cfg(int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE) {
    printf("[Launch] BM=%d BN=%d BK=%d TM=%d TN=%d BLOCK=%d\n",
           BM, BN, BK, TM, TN, BLOCK_SIZE);
    fflush(stdout);
}

// template<int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE>
// inline void run_one() {
//     using Cfg = SGEMMConfigConst<BM, BN, BK, TM, TN, BLOCK_SIZE>;

//     // ===== 编译期开关 =====
//     constexpr bool USE_A_TRANSPOSE = true;
//     constexpr bool USE_A_PADDING   = true;

//     // 是否“必须”用 float4（若为 false，则有备选的标量版本可跑）
//     constexpr bool REQUIRE_FLOAT4  = true;

//     // 是否启用双/单缓冲
//     constexpr bool ENABLE_DB = true;
//     constexpr bool ENABLE_SB = true;

//     // ===== 基础约束 =====
//     constexpr bool base_ok =
//         ((BM % TM) == 0) &&
//         ((BN % TN) == 0) &&
//         ((BM * BN) % (TM * TN) == 0) &&
//         (((BM * BN) / (TM * TN)) == BLOCK_SIZE) &&
//         (((BM / TM) * (BN / TN)) <= 1024);

//     // v1（2D 映射）需要“列铺满”
//     constexpr bool col_full_B = (BN <= BLOCK_SIZE);
//     constexpr bool col_full_A = (BK <= BLOCK_SIZE);
//     constexpr bool valid_v1   = base_ok && col_full_A && col_full_B;
//     // v2（拉直）通用
//     constexpr bool valid_v2   = base_ok;

//     // ===== float4 判定（拆成载入/计算两类）=====
//     // 载入只关心列方向与维度对齐（A: K, B: N；smem: BK/BN）
//     // 你前面说“我确定总是对齐”，那可以把 K%4==0/N%4==0 当作前置成立。
//     constexpr bool vec_load_ok =
//         (BK % 4 == 0) && (BN % 4 == 0);  // K%4/N%4 由数据构造保证

//     // 计算端若想也用 float4（可选）：通常需要 TM/TN 也为 4 的倍数
//     constexpr bool vec_math_ok =
//         (TM % 4 == 0) && (TN % 4 == 0) && (BK % 4 == 0);

//     // 只要“必须 float4”就以载入为准；若不强制，也可退化到标量
//     constexpr bool vec_ok =
//         (!REQUIRE_FLOAT4) ? true : vec_load_ok;

//     // ===== 共享内存占用 =====
//     constexpr int  A_PAD   = (USE_A_TRANSPOSE && USE_A_PADDING)
//                              ? ((BM % 32 == 0) ? 4 : ((4 - (BM % 4)) % 4))
//                              : 0;
//     constexpr int  A_PITCH = USE_A_TRANSPOSE ? (BM + A_PAD) : BM;

//     // 注意：As 用 [BK x A_PITCH]，Bs 用 [BK x BN]
//     constexpr size_t SmemFloatsSingle = size_t(BK) * size_t(A_PITCH)
//                                       + size_t(BK) * size_t(BN);
//     constexpr size_t SmemBytesSingle  = SmemFloatsSingle * sizeof(float);
//     constexpr size_t SmemBytesDouble  = 2ull * SmemBytesSingle;

//     constexpr size_t SmemCapBytes     = 64ull * 1024ull; // 64KB

//     constexpr bool sb_fit_64k = (SmemBytesSingle <= SmemCapBytes);
//     constexpr bool db_fit_64k = (SmemBytesDouble <= SmemCapBytes);

//     // ===== Kernel 可运行性 =====
//     // 双缓冲（先）
//     constexpr bool K_DB_V1_OK = ENABLE_DB && db_fit_64k && valid_v1 && vec_ok;
//     constexpr bool K_DB_V2_OK = ENABLE_DB && db_fit_64k && valid_v2 && vec_ok;

//     // 单缓冲（后）
//     constexpr bool K_SB_V1_OK = ENABLE_SB && sb_fit_64k && valid_v1 && vec_ok;
//     constexpr bool K_SB_V2_OK = ENABLE_SB && sb_fit_64k && valid_v2 && vec_ok;

//     // 强制让 float4_v3 走“通用单缓冲”支路（不被 BN>BLOCK_SIZE 卡住）
//     constexpr bool K_SB_F4_V3_OK = ENABLE_SB && sb_fit_64k && valid_v2 && vec_ok;

//     // 若至少有一个能跑，再打印
//     if constexpr (K_DB_V1_OK || K_DB_V2_OK || K_SB_V1_OK || K_SB_V2_OK || K_SB_F4_V3_OK) {
//         print_cfg(BM, BN, BK, TM, TN, BLOCK_SIZE);
//     }

//     // ===== 先 Double Buffer =====
//     if constexpr (K_DB_V1_OK) { 
//         // test_sgemm(sgemm_double_buffer_v1<Cfg>);
//         // test_sgemm(sgemm_float4_v4<Cfg>);
//     }
//     if constexpr (K_DB_V2_OK) { 
//         // test_sgemm(sgemm_double_buffer_v2<Cfg>); 
//     }

//     // ===== 再 Single Buffer =====
//     if constexpr (K_SB_V1_OK) { 
//         // test_sgemm(sgemm_block_tile_v2<Cfg>); 
//     }
//     if constexpr (K_SB_V2_OK) { 
//         // test_sgemm(sgemm_block_tile_v3<Cfg>); 
//     }

//     // 单缓冲 float4（通用）
//     if constexpr (K_SB_F4_V3_OK) {
//         // 若你的 v3 就是“单缓冲 + 拉直 + float4 载入”的实现
//         test_sgemm(sgemm_float4_v3<Cfg>);
//         test_sgemm(sgemm_float4_v4<Cfg>);
//         // compare_sgemm(sgemm_cublas, sgemm_float4_v4<Cfg>);
//     }

//     // 全部不满足则静默跳过
// }


// // ---- 最后一层：根据 BLOCK_SIZE 分发 ----
// template<int BM, int BN, int BK, int TM, int TN>
// inline void dispatch_block_size(int block_size) {
//     switch (block_size) {
//         case 128: run_one<BM, BN, BK, TM, TN, 128>(); break;
//         case 256: run_one<BM, BN, BK, TM, TN, 256>(); break;
//         default: /* 非法 BLOCK_SIZE，忽略 */ break;
//     }
// }

// // ---- 第五层：TN 分发 ----
// template<int BM, int BN, int BK, int TM>
// inline void dispatch_tn(int TN, int block_size) {
//     switch (TN) {
//         case 4:  dispatch_block_size<BM, BN, BK, TM, 4 >(block_size); break;
//         case 8:  dispatch_block_size<BM, BN, BK, TM, 8 >(block_size); break;
//         case 16: dispatch_block_size<BM, BN, BK, TM, 16>(block_size); break;
//         case 32: dispatch_block_size<BM, BN, BK, TM, 32>(block_size); break;
//         default: break;
//     }
// }

// // ---- 第四层：TM 分发 ----
// template<int BM, int BN, int BK>
// inline void dispatch_tm(int TM, int TN, int block_size) {
//     switch (TM) {
//         case 4:  dispatch_tn<BM, BN, BK, 4 >(TN, block_size); break;
//         case 8:  dispatch_tn<BM, BN, BK, 8 >(TN, block_size); break;
//         case 16: dispatch_tn<BM, BN, BK, 16>(TN, block_size); break;
//         case 32: dispatch_tn<BM, BN, BK, 32>(TN, block_size); break;
//         default: break;
//     }
// }

// // ---- 第三层：BK 分发 ----
// template<int BM, int BN>
// inline void dispatch_bk(int BK, int TM, int TN, int block_size) {
//     switch (BK) {
//         case 4:  dispatch_tm<BM, BN, 4 >(TM, TN, block_size); break;
//         case 8:  dispatch_tm<BM, BN, 8 >(TM, TN, block_size); break;
//         case 16: dispatch_tm<BM, BN, 16>(TM, TN, block_size); break;
//         case 32: dispatch_tm<BM, BN, 32>(TM, TN, block_size); break;
//         default: break;
//     }
// }

// // ---- 第二层：BN 分发 ----
// template<int BM>
// inline void dispatch_bn(int BN, int BK, int TM, int TN, int block_size) {
//     switch (BN) {
//         case 64:  dispatch_bk<BM, 64 >(BK, TM, TN, block_size); break;
//         case 128: dispatch_bk<BM, 128>(BK, TM, TN, block_size); break;
//         case 256: dispatch_bk<BM, 256>(BK, TM, TN, block_size); break;
//         default: break;
//     }
// }

// // ---- 第一层：BM 分发 ----
// inline void dispatch_bm(int BM, int BN, int BK, int TM, int TN, int block_size) {
//     switch (BM) {
//         case 64:  dispatch_bn<64 >(BN, BK, TM, TN, block_size); break;
//         case 128: dispatch_bn<128>(BN, BK, TM, TN, block_size); break;
//         case 256: dispatch_bn<256>(BN, BK, TM, TN, block_size); break;
//         default: break;
//     }
// }

// void run_tuning() {
//     const int BMs[] = {64, 128, 256};
//     const int BNs[] = {64, 128, 256};
//     const int BKs[] = {8, 16, 32};
//     const int TMs[] = {8, 16, 32};
//     const int TNs[] = {8, 16, 32};

//     for (int BM : BMs) {
//         for (int BN : BNs) {
//             for (int BK : BKs) {
//                 for (int TM : TMs) {
//                     for (int TN : TNs) {
//                         // 约束：可整除 & 线程不超限
//                         if (BM % TM != 0 || BN % TN != 0) continue;

//                         const int threads = (BM / TM) * (BN / TN);
//                         if (threads > 1024) continue;

//                         // 计算 BLOCK_SIZE，并限定在 {128, 256}
//                         if ((BM * BN) % (TM * TN) != 0) continue;
//                         int BLOCK_SIZE = (BM * BN) / (TM * TN);
//                         if (!(BLOCK_SIZE == 128 || BLOCK_SIZE == 256)) continue;

//                         // 进入模板分发（在里面用 switch 选择编译期实例）
//                         dispatch_bm(BM, BN, BK, TM, TN, BLOCK_SIZE);
//                     }
//                 }
//             }
//         }
//     }
// }
