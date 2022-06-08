#include "device_launch_parameters.h"
#include "hashmap.cuh"

#include <iostream>

#include <stdio.h>

int main()
{
    HashMap<float, float, HashFunc<float>> map;

    // Add vectors in parallel.
    cudaError_t cudaStatus = cudaSetDevice(0);
    map.Put(2.0f, 2.0f);
    map.Put(3.0f, 100.0f);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    std::cout << map[3.0f] << std::endl;

    map.~HashMap();

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}