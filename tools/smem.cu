// smem_limits_min.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(x)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        auto e = (x);                                                                              \
        if (e != cudaSuccess)                                                                      \
        {                                                                                          \
            fprintf(stderr, "CUDA error %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

__global__ void dummy_kernel() {}

int main()
{
    int dev = 0;
    CHECK(cudaGetDevice(&dev));

    // 必要的上限（都应该支持）
    int per_sm = 0, per_block_default = 0, per_block_optin = 0;
    CHECK(cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    CHECK(cudaDeviceGetAttribute(&per_block_default, cudaDevAttrMaxSharedMemoryPerBlock, dev));
    // 旧驱动可能不支持 optin 查询：做兼容处理
    cudaError_t e = cudaDeviceGetAttribute(&per_block_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (e != cudaSuccess)
    {
        per_block_optin = per_block_default; // 退化为默认上限
    }

    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Shared mem per SM (hard cap): %d bytes\n", per_sm);
    printf("Per-block shared mem default : %d bytes\n", per_block_default);
    printf("Per-block shared mem opt-in  : %d bytes\n", per_block_optin);

    // 读取 kernel 的动态共享内存默认上限
    cudaFuncAttributes attr{};
    CHECK(cudaFuncGetAttributes(&attr, dummy_kernel));
    printf("\n[kernel] default max dynamic smem: %d bytes\n", attr.maxDynamicSharedSizeBytes);

    // 尝试把该 kernel 的动态共享内存上限提升到 opt-in 值（若不支持会返回错误）
    if (per_block_optin > per_block_default)
    {
        e = cudaFuncSetAttribute(dummy_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, per_block_optin);
        printf("opt-in set to %d bytes: %s\n", per_block_optin,
               (e == cudaSuccess ? "OK" : cudaGetErrorString(e)));
        // 再读一遍看看是否生效
        CHECK(cudaFuncGetAttributes(&attr, dummy_kernel));
        printf("[kernel] after opt-in, max dynamic smem: %d bytes\n", attr.maxDynamicSharedSizeBytes);
    }

    // 小验证：用“当前 kernel 动态上限”发一次
    size_t dyn = (size_t)attr.maxDynamicSharedSizeBytes;
    printf("\nLaunching with dynamic smem = %zu ...\n", dyn);
    dummy_kernel<<<1, 32, dyn>>>();
    e = cudaDeviceSynchronize();
    printf("%s\n", (e == cudaSuccess ? "Launch OK" : cudaGetErrorString(e)));
    return 0;
}
