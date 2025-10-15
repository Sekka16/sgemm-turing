#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib> // 新增：用于std::exit函数
#include <iostream>

constexpr int M = 256;
constexpr int N = 2048;
constexpr int K = 2048;

template <int BM_, int BN_, int BK_, int TM_, int TN_, int BLOCK_SIZE_>
struct SGEMMConfigConst {
    static constexpr int BM = BM_;
    static constexpr int BN = BN_;
    static constexpr int BK = BK_;
    static constexpr int TM = TM_;
    static constexpr int TN = TN_;
    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;
    static_assert((BM * BN) / (TM * TN) == BLOCK_SIZE_ && "(BM * BN) / (TM * TN) != BLOCK_SIZE_");
};

using SGEMMfunc = void(*)(const float* a, const float* b, float* c, int M, int N, int K);

void compare_sgemm(SGEMMfunc ref_impl, SGEMMfunc test_impl) {
    float* a = new float[M * K];
    float* b = new float[K * N];
    float* c_ref = new float[M * N];
    float* c_test = new float[M * N];

    // 初始化输入
    for (int i = 0; i < M * K; ++i) a[i] = static_cast<float>(i % 100) * 0.02f;
    for (int i = 0; i < K * N; ++i) b[i] = static_cast<float>((i % 100) - 50) * 0.04f;
    std::fill(c_ref, c_ref + M * N, 0.0f);
    std::fill(c_test, c_test + M * N, 0.0f);

    // 分配设备内存
    float *d_a, *d_b, *d_c_ref, *d_c_test;
    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_c_ref, M * N * sizeof(float));
    cudaMalloc(&d_c_test, M * N * sizeof(float));

    cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_ref, c_ref, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_test, c_test, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // 调用参考实现
    ref_impl(d_a, d_b, d_c_ref, M, N, K);
    // 调用测试实现
    test_impl(d_a, d_b, d_c_test, M, N, K);

    cudaMemcpy(c_ref, d_c_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(c_test, d_c_test, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 比较输出
    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::fabs(c_ref[i] - c_test[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-3f) {
            if (++errors < 10)
                std::cerr << "Mismatch at index " << i / N << "," << i % N << ": ref = " << c_ref[i] << ", test = " << c_test[i] << ", diff = " << diff << "\n";
        }
    }

    std::cout << "Compare complete: errors = " << errors << ", max_diff = " << max_diff << std::endl;

    // 清理
    delete[] a;
    delete[] b;
    delete[] c_ref;
    delete[] c_test;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_ref);
    cudaFree(d_c_test);
}

void test_sgemm(SGEMMfunc sgemm_impl) {
    float* a = new float[M * K];
    float* b = new float[K * N];
    float* c = new float[M * N];

    for (int i = 0; i < M * K; ++i) {
        a[i] = i;
    }
    for (int i = 0; i < K * N; ++i) {
        b[i] = i;
    }

    float* d_a, *d_b, *d_c;
    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, M * N * sizeof(float), cudaMemcpyHostToDevice);

    sgemm_impl(d_a, d_b, d_c, M, N, K);

    cudaMemcpy(c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] a;
    delete[] b;
    delete[] c;
}

static inline void print_cfg(int BM, int BN, int BK, int TM, int TN, int BLOCK_SIZE) {
    printf("[Launch] BM=%d BN=%d BK=%d TM=%d TN=%d BLOCK=%d\n",
           BM, BN, BK, TM, TN, BLOCK_SIZE);
    fflush(stdout);
}