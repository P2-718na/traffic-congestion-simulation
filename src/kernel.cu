#include "kernel.h"

#include <cstdio>

void __global__ print()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::printf("%d\n", idx);
}

void f()
{
    int count{};
    const auto error = cudaGetDeviceCount(&count);

    std::printf("Found %i GPUs", count);
    if (error != 0) {
        exit(1);
    }

    print<<<1, 10>>>();
    cudaDeviceSynchronize();
}