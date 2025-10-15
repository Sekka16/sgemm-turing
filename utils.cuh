#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cstdlib>  // 新增：用于std::exit函数

#define CUDA_CHECK_PRE() \
    do { \
        cudaError_t err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[Pre-launch] CUDA error: %s\n", cudaGetErrorString(err)); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUDA_CHECK_POST_LAUNCH() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[Post-launch] CUDA kernel launch failed: %s\n", cudaGetErrorString(err)); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUDA_CHECK_POST_SYNC() \
    do { \
        cudaError_t err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[Post-sync] CUDA kernel execution failed: %s\n", cudaGetErrorString(err)); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

#define KERNEL_TIMER_START() \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop);  \
    cudaEventRecord(start);

#define KERNEL_TIMER_STOP(kernel_name)                                  \
    cudaEventRecord(stop);                                              \
    cudaEventSynchronize(stop);                                         \
    float elapsed_time_ms = 0.0f;                                       \
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);                \
    printf("[Timing] %s took %.3f ms\n", kernel_name, elapsed_time_ms); \
    cudaEventDestroy(start);                                            \
    cudaEventDestroy(stop);
