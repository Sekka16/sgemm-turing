#pragma once
#include <cstdint>
#include "test.cuh"

// ========= 你项目里的前置声明（链接到已有实现） =========
template<int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE>
struct SGEMMConfigConst;

void print_cfg(int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE);

template<typename Kernel>
void test_sgemm(Kernel);

template<typename Cfg> void sgemm_float4_v1(const float*, const float*, float*, int, int, int);
template<typename Cfg> void sgemm_float4_v2(const float*, const float*, float*, int, int, int);

// 单 block 动态共享内存上限（按需改 48KB/64KB）
#ifndef F4_PER_BLOCK_SMEM_CAP_BYTES
#define F4_PER_BLOCK_SMEM_CAP_BYTES (64ull * 1024ull)
#endif

// ========= float4 约束聚合（镜像 kernel 里的 static_assert + 共享内存计算） =========
template<int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE>
struct F4Check {
    // 1) 基本映射
    static constexpr bool map_div_ok  = (BM % TM == 0) && (BN % TN == 0);
    static constexpr int  TPB         = (BM / TM) * (BN / TN);
    static constexpr bool tpb_ok      = (TPB <= 1024);
    static constexpr bool block_sz_ok = ((BM * BN) % (TM * TN) == 0) &&
                                        (BLOCK_SIZE == (BM * BN) / (TM * TN));

    // 2) 向量化必要条件
    static constexpr bool vec_align_ok = (TM % 4 == 0) && (TN % 4 == 0) && (BK % 4 == 0);

    // 3) 行组/列线程组、覆盖 BN
    static constexpr int  ROW_GROUPS   = BM / TM;                // >0 因为 BM%TM==0
    static constexpr bool row_groups_ok= (ROW_GROUPS > 0) && (BLOCK_SIZE % ROW_GROUPS == 0);
    static constexpr int  TCOL         = row_groups_ok ? (BLOCK_SIZE / ROW_GROUPS) : 0;
    static constexpr int  VEC          = 4;
    static constexpr int  G            = TN / 4;
    static constexpr bool cover_bn_ok  = row_groups_ok && (TCOL * VEC * G == BN);

    // 4) APITCH 对齐（AsT 为 BK x APITCH）
    static constexpr int  APAD         = (BM % 32 == 0) ? 4 : ((4 - (BM % 4)) % 4);
    static constexpr int  APITCH       = BM + APAD;
    static constexpr bool apitch_ok    = (APITCH % 4 == 0);

    // 5) 单缓冲共享内存容量 —— 注意 AsT 用 APITCH
    static constexpr uint64_t smem_floats =
        uint64_t(BK) * uint64_t(APITCH) + uint64_t(BK) * uint64_t(BN);
    static constexpr uint64_t smem_bytes  = smem_floats * sizeof(float);
    static constexpr bool smem_ok         = (smem_bytes <= F4_PER_BLOCK_SMEM_CAP_BYTES);

    // —— 总条件 ——（必须全部满足才允许实例化内核）
    static constexpr bool ok = map_div_ok && tpb_ok && block_sz_ok
                             && vec_align_ok
                             && cover_bn_ok
                             && apitch_ok
                             && smem_ok;
};

// ========= SFINAE Invoker：OK=false 不实例化内核，OK=true 才真正调用 =========
template<bool OK, typename Cfg, int V /* 1 or 2 */>
struct F4Invoker { static inline void run() {} };

template<typename Cfg>
struct F4Invoker<true, Cfg, 1> {
    static inline void run() { 
        // test_sgemm(sgemm_float4_v1<Cfg>); 
        compare_sgemm(sgemm_cublas, sgemm_float4_v1<Cfg>);
    }
};
template<typename Cfg>
struct F4Invoker<true, Cfg, 2> {
    static inline void run() { 
        // test_sgemm(sgemm_float4_v2<Cfg>);
        compare_sgemm(sgemm_cublas, sgemm_float4_v2<Cfg>);
    }
};

// ========= Runner：类模板 + 静态 exec() =========
template<int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE>
struct F4_RunV1 {
    static inline void exec() {
        using Cfg = SGEMMConfigConst<BM, BN, BK, TM, TN, BLOCK_SIZE>;
        using Chk = F4Check<BM, BN, BK, TM, TN, BLOCK_SIZE>;
        if constexpr (Chk::ok) { print_cfg(BM, BN, BK, TM, TN, BLOCK_SIZE); }
        F4Invoker<Chk::ok, Cfg, 1>::run();
    }
};

template<int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE>
struct F4_RunV2 {
    static inline void exec() {
        using Cfg = SGEMMConfigConst<BM, BN, BK, TM, TN, BLOCK_SIZE>;
        using Chk = F4Check<BM, BN, BK, TM, TN, BLOCK_SIZE>;
        if constexpr (Chk::ok) { print_cfg(BM, BN, BK, TM, TN, BLOCK_SIZE); }
        F4Invoker<Chk::ok, Cfg, 2>::run();
    }
};

// ========= 6 层分发（和你其它 tuner 一样的写法，前缀 f4_） =========
template<template<int,int,int,int,int,int> class Runner, int BM, int BN, int BK, int TM, int TN>
inline void f4_dispatch_block_size(int block_size) {
    switch (block_size) {
        case 128: Runner<BM, BN, BK, TM, TN, 128>::exec(); break;
        case 256: Runner<BM, BN, BK, TM, TN, 256>::exec(); break;
        default: break;
    }
}
template<template<int,int,int,int,int,int> class Runner, int BM, int BN, int BK, int TM>
inline void f4_dispatch_tn(int TN, int block_size) {
    switch (TN) {
        case 4:  f4_dispatch_block_size<Runner, BM, BN, BK, TM, 4 >(block_size); break;
        case 8:  f4_dispatch_block_size<Runner, BM, BN, BK, TM, 8 >(block_size); break;
        case 16: f4_dispatch_block_size<Runner, BM, BN, BK, TM, 16>(block_size); break;
        case 32: f4_dispatch_block_size<Runner, BM, BN, BK, TM, 32>(block_size); break;
        default: break;
    }
}
template<template<int,int,int,int,int,int> class Runner, int BM, int BN, int BK>
inline void f4_dispatch_tm(int TM, int TN, int block_size) {
    switch (TM) {
        case 4:  f4_dispatch_tn<Runner, BM, BN, BK, 4 >(TN, block_size); break;
        case 8:  f4_dispatch_tn<Runner, BM, BN, BK, 8 >(TN, block_size); break;
        case 16: f4_dispatch_tn<Runner, BM, BN, BK, 16>(TN, block_size); break;
        case 32: f4_dispatch_tn<Runner, BM, BN, BK, 32>(TN, block_size); break;
        default: break;
    }
}
template<template<int,int,int,int,int,int> class Runner, int BM, int BN>
inline void f4_dispatch_bk(int BK, int TM, int TN, int block_size) {
    switch (BK) {
        case 4:  f4_dispatch_tm<Runner, BM, BN, 4 >(TM, TN, block_size); break;
        case 8:  f4_dispatch_tm<Runner, BM, BN, 8 >(TM, TN, block_size); break;
        case 16: f4_dispatch_tm<Runner, BM, BN, 16>(TM, TN, block_size); break;
        case 32: f4_dispatch_tm<Runner, BM, BN, 32>(TM, TN, block_size); break;
        default: break;
    }
}
template<template<int,int,int,int,int,int> class Runner, int BM>
inline void f4_dispatch_bn(int BN, int BK, int TM, int TN, int block_size) {
    switch (BN) {
        case 64:  f4_dispatch_bk<Runner, BM, 64 >(BK, TM, TN, block_size); break;
        case 128: f4_dispatch_bk<Runner, BM, 128>(BK, TM, TN, block_size); break;
        case 256: f4_dispatch_bk<Runner, BM, 256>(BK, TM, TN, block_size); break;
        default: break;
    }
}
template<template<int,int,int,int,int,int> class Runner>
inline void f4_dispatch_bm(int BM, int BN, int BK, int TM, int TN, int block_size) {
    switch (BM) {
        case 64:  f4_dispatch_bn<Runner, 64 >(BN, BK, TM, TN, block_size); break;
        case 128: f4_dispatch_bn<Runner, 128>(BN, BK, TM, TN, block_size); break;
        case 256: f4_dispatch_bn<Runner, 256>(BN, BK, TM, TN, block_size); break;
        default: break;
    }
}

// ========= 两个对外入口（分别跑 v1 / v2） =========
inline void run_float4_v1_tuning() {
    const int BMs[] = {64, 128, 256};
    const int BNs[] = {64, 128, 256};
    const int BKs[] = {8, 16, 32};
    const int TMs[] = {4, 8, 16, 32};
    const int TNs[] = {4, 8, 16, 32};

    for (int BM : BMs) for (int BN : BNs) for (int BK : BKs)
    for (int TM : TMs) for (int TN : TNs) {
        // 运行时快速过滤（减少模板实例）
        if (BM % TM != 0 || BN % TN != 0) continue;
        const int threads = (BM / TM) * (BN / TN);
        if (threads > 1024) continue;
        if ((BM * BN) % (TM * TN) != 0) continue;
        if (TM * TN >= 256) continue;

        const int BLOCKSZ = (BM * BN) / (TM * TN);
        if (!(BLOCKSZ == 128 || BLOCKSZ == 256)) continue;

        // 仅把候选交给分发（是否真正实例化，由编译期 F4Check 决定）
        f4_dispatch_bm<F4_RunV1>(BM, BN, BK, TM, TN, BLOCKSZ);
    }
}

inline void run_float4_v2_tuning() {
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

        f4_dispatch_bm<F4_RunV2>(BM, BN, BK, TM, TN, BLOCKSZ);
    }
}

// 想两个版本都跑
inline void run_float4_tuning() {
    run_float4_v1_tuning();
    run_float4_v2_tuning();
}
