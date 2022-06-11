#include "device_launch_parameters.h"
#include "hashmap.cuh"

#include <iostream>

#include <stdio.h>

__global__ void TestKernel(float* result, HashMap<float, float, HashFunc<float>>& map)
{
    float output = 1;
    result = &output;
}

int main()
{
    HashMap<float, float, HashFunc<float>> map;

    cudaError_t cuda_status = cudaSetDevice(0);

    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "CudaSetDevice failed!");
        return 1;
    }

    map.Put(2.0f, 2.0f);
    map.Put(3.0f, 100.0f);
    map.Put(3.0f, 120.0f);

    float* result;
    cudaMalloc(&result, sizeof(float));

    TestKernel<<<1, 1>>> (result, map);


    function<cudaError_t()> error_func = []() { return cudaGetLastError(); };
    cuda_status = WrapperFunction(error_func, "Main", "GPUAccessLastError", cuda_status, "");

    function<cudaError_t()> sync_func = []() { return cudaDeviceSynchronize(); };
    cuda_status = WrapperFunction(sync_func, "Main", "GPUAccessSyncFunc", cuda_status, "");

    map.~HashMap();

    cuda_status = cudaDeviceReset();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}