#pragma once
#include <cstdint>
#include "test.cuh"

// ===== 你项目里的前置声明（链接到已有实现） =====
template<int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE>
struct SGEMMConfigConst;

void print_cfg(int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE);

template<typename Kernel>
void test_sgemm(Kernel);

template<typename Cfg> void sgemm_double_buffer_v1(const float*, const float*, float*, int, int, int);
template<typename Cfg> void sgemm_double_buffer_v2(const float*, const float*, float*, int, int, int);

// 单个 block 动态共享内存上限（按需改 48KB/64KB）
#ifndef DB_PER_BLOCK_SMEM_CAP_BYTES
#define DB_PER_BLOCK_SMEM_CAP_BYTES (64ull * 1024ull)
#endif

// ===== 单配置的基础检查（双缓冲：SMEM 容量 ×2） =====
template<int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE>
struct DBCheck {
    // 线程/映射基础
    static constexpr bool div_ok   = (BM % TM == 0) && (BN % TN == 0);
    static constexpr int  TPB      = (BM / TM) * (BN / TN);
    static constexpr bool tpb_ok   = (TPB <= 1024);

    // BLOCK_SIZE 必须等于 (BM*BN)/(TM*TN)
    static constexpr bool blk_ok   = ((BM * BN) % (TM * TN) == 0) &&
                                     (BLOCK_SIZE == (BM * BN) / (TM * TN));

    // 双缓冲共享内存：As0/As1 + Bs0/Bs1
    static constexpr uint64_t smem_floats_single =
        uint64_t(BM) * uint64_t(BK) + uint64_t(BK) * uint64_t(BN);
    static constexpr uint64_t smem_bytes_db =
        2ull * smem_floats_single * sizeof(float);
    static constexpr bool smem_ok = (smem_bytes_db <= DB_PER_BLOCK_SMEM_CAP_BYTES);

    // v1 额外：列“铺满”（与 block-tile v1 同口径）
    static constexpr bool col_full_ok = (BN <= BLOCK_SIZE) && (BK <= BLOCK_SIZE);
};

// ===== Runner：类模板 + 静态 exec() =====

// v1：列铺满版本（BN<=BLOCK_SIZE && BK<=BLOCK_SIZE）
template<int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE>
struct DB_RunV1 {
    static inline void exec() {
        using Cfg = SGEMMConfigConst<BM, BN, BK, TM, TN, BLOCK_SIZE>;
        using Chk = DBCheck<BM, BN, BK, TM, TN, BLOCK_SIZE>;
        if constexpr (Chk::div_ok && Chk::tpb_ok && Chk::blk_ok && Chk::smem_ok && Chk::col_full_ok) {
            print_cfg(BM, BN, BK, TM, TN, BLOCK_SIZE);
            test_sgemm(sgemm_double_buffer_v1<Cfg>);
        }
    }
};

// v2：拉直通用版本（无列铺满限制）
template<int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE>
struct DB_RunV2 {
    static inline void exec() {
        using Cfg = SGEMMConfigConst<BM, BN, BK, TM, TN, BLOCK_SIZE>;
        using Chk = DBCheck<BM, BN, BK, TM, TN, BLOCK_SIZE>;
        if constexpr (Chk::div_ok && Chk::tpb_ok && Chk::blk_ok && Chk::smem_ok) {
            print_cfg(BM, BN, BK, TM, TN, BLOCK_SIZE);
            test_sgemm(sgemm_double_buffer_v2<Cfg>);
        }
    }
};

// ===== 6 层分发（和 block-tile 保持一致的风格） =====
template<template<int,int,int,int,int,int> class Runner, int BM, int BN, int BK, int TM, int TN>
inline void db_dispatch_block_size(int block_size) {
    switch (block_size) {
        case 128: Runner<BM, BN, BK, TM, TN, 128>::exec(); break;
        case 256: Runner<BM, BN, BK, TM, TN, 256>::exec(); break;
        default: break;
    }
}

template<template<int,int,int,int,int,int> class Runner, int BM, int BN, int BK, int TM>
inline void db_dispatch_tn(int TN, int block_size) {
    switch (TN) {
        case 4:  db_dispatch_block_size<Runner, BM, BN, BK, TM, 4 >(block_size); break;
        case 8:  db_dispatch_block_size<Runner, BM, BN, BK, TM, 8 >(block_size); break;
        case 16: db_dispatch_block_size<Runner, BM, BN, BK, TM, 16>(block_size); break;
        case 32: db_dispatch_block_size<Runner, BM, BN, BK, TM, 32>(block_size); break;
        default: break;
    }
}

template<template<int,int,int,int,int,int> class Runner, int BM, int BN, int BK>
inline void db_dispatch_tm(int TM, int TN, int block_size) {
    switch (TM) {
        case 4:  db_dispatch_tn<Runner, BM, BN, BK, 4 >(TN, block_size); break;
        case 8:  db_dispatch_tn<Runner, BM, BN, BK, 8 >(TN, block_size); break;
        case 16: db_dispatch_tn<Runner, BM, BN, BK, 16>(TN, block_size); break;
        case 32: db_dispatch_tn<Runner, BM, BN, BK, 32>(TN, block_size); break;
        default: break;
    }
}

template<template<int,int,int,int,int,int> class Runner, int BM, int BN>
inline void db_dispatch_bk(int BK, int TM, int TN, int block_size) {
    switch (BK) {
        case 4:  db_dispatch_tm<Runner, BM, BN, 4 >(TM, TN, block_size); break;
        case 8:  db_dispatch_tm<Runner, BM, BN, 8 >(TM, TN, block_size); break;
        case 16: db_dispatch_tm<Runner, BM, BN, 16>(TM, TN, block_size); break;
        case 32: db_dispatch_tm<Runner, BM, BN, 32>(TM, TN, block_size); break;
        default: break;
    }
}

template<template<int,int,int,int,int,int> class Runner, int BM>
inline void db_dispatch_bn(int BN, int BK, int TM, int TN, int block_size) {
    switch (BN) {
        case 64:  db_dispatch_bk<Runner, BM, 64 >(BK, TM, TN, block_size); break;
        case 128: db_dispatch_bk<Runner, BM, 128>(BK, TM, TN, block_size); break;
        case 256: db_dispatch_bk<Runner, BM, 256>(BK, TM, TN, block_size); break;
        default: break;
    }
}

template<template<int,int,int,int,int,int> class Runner>
inline void db_dispatch_bm(int BM, int BN, int BK, int TM, int TN, int block_size) {
    switch (BM) {
        case 64:  db_dispatch_bn<Runner, 64 >(BN, BK, TM, TN, block_size); break;
        case 128: db_dispatch_bn<Runner, 128>(BN, BK, TM, TN, block_size); break;
        case 256: db_dispatch_bn<Runner, 256>(BN, BK, TM, TN, block_size); break;
        default: break;
    }
}

// ===== 两个对外入口：分别跑 v1 / v2 =====
inline void run_double_buffer_v1_tuning() {
    const int BMs[] = {64, 128, 256};
    const int BNs[] = {64, 128, 256};
    const int BKs[] = {8, 16, 32};
    const int TMs[] = {4, 8, 16, 32};
    const int TNs[] = {4, 8, 16, 32};

    for (int BM : BMs) for (int BN : BNs) for (int BK : BKs)
    for (int TM : TMs) for (int TN : TNs) {
        if (BM % TM != 0 || BN % TN != 0) continue;
        const int threads = (BM / TM) * (BN / TN);
        if (threads > 1024) continue;
        if ((BM * BN) % (TM * TN) != 0) continue;
        if (TM * TN >= 256) continue;

        const int BLOCKSZ = (BM * BN) / (TM * TN);
        if (!(BLOCKSZ == 128 || BLOCKSZ == 256)) continue;

        // v1 额外：列“铺满”
        if (BN > BLOCKSZ || BK > BLOCKSZ) continue;

        // 双缓冲共享内存上限
        const uint64_t smem_bytes_db =
            2ull * ( (uint64_t)BM * BK + (uint64_t)BK * BN ) * sizeof(float);
        if (smem_bytes_db > DB_PER_BLOCK_SMEM_CAP_BYTES) continue;

        db_dispatch_bm<DB_RunV1>(BM, BN, BK, TM, TN, BLOCKSZ);
    }
}

inline void run_double_buffer_v2_tuning() {
    const int BMs[] = {64, 128, 256};
    const int BNs[] = {64, 128, 256};
    const int BKs[] = {8, 16, 32};
    const int TMs[] = {4, 8, 16, 32};
    const int TNs[] = {4, 8, 16, 32};

    for (int BM : BMs) for (int BN : BNs) for (int BK : BKs)
    for (int TM : TMs) for (int TN : TNs) {
        if (BM % TM != 0 || BN % TN != 0) continue;
        const int threads = (BM / TM) * (BN / TN);
        if (threads > 1024) continue;
        if ((BM * BN) % (TM * TN) != 0) continue;
        if (TM * TN >= 256) continue;

        const int BLOCKSZ = (BM * BN) / (TM * TN);
        if (!(BLOCKSZ == 128 || BLOCKSZ == 256)) continue;

        const uint64_t smem_bytes_db =
            2ull * ( (uint64_t)BM * BK + (uint64_t)BK * BN ) * sizeof(float);
        if (smem_bytes_db > DB_PER_BLOCK_SMEM_CAP_BYTES) continue;

        db_dispatch_bm<DB_RunV2>(BM, BN, BK, TM, TN, BLOCKSZ);
    }
}

// 想两个版本都跑
inline void run_double_buffer_tuning() {
    run_double_buffer_v1_tuning();
    run_double_buffer_v2_tuning();
}
