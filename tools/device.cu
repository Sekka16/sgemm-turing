#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "SM count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Clock rate (MHz): " << prop.clockRate / 1000.0 << std::endl;
    std::cout << "Memory Clock rate (MHz): " << prop.memoryClockRate / 1000.0 << std::endl;
    std::cout << "Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;

    double bandwidthGBs =
        2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    std::cout << "Theoretical Bandwidth (GB/s): " << bandwidthGBs << std::endl;

    return 0;
}