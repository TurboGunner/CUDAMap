#include "device_launch_parameters.h"
#include "hashmap.cuh"

#include <iostream>

#include <stdio.h>

__global__ void TestKernel(HashMap<float, float, HashFunc<float>>* map)
{
    printf("%f", map->Get(4.0f));
}

int main()
{
    cudaError_t cuda_status = cudaSuccess;

    HashMap<float, float, HashFunc<float>>* map = new HashMap<float, float, HashFunc<float>>();

    function<cudaError_t()> set_device_func = []() { return cudaSetDevice(0); };
    cuda_status = WrapperFunction(set_device_func, "Main", "GPUSetDevice", cuda_status, "");

    map->Put(2.0f, 2.0f);
    map->Put(3.0f, 100.0f);
    map->Put(4.0f, 120.0f);
    map->Put(1.0f, 1220.0f);

    TestKernel<<<1, 1>>> (map);

    function<cudaError_t()> error_func = []() { return cudaGetLastError(); };
    cuda_status = WrapperFunction(error_func, "Main", "GPUAccessLastError", cuda_status, "");

    function<cudaError_t()> sync_func = []() { return cudaDeviceSynchronize(); };
    cuda_status = WrapperFunction(sync_func, "Main", "GPUAccessSyncFunc", cuda_status, "");

    delete map;

    function<cudaError_t()> reset_func = []() { return cudaDeviceReset(); };
    cuda_status = WrapperFunction(reset_func, "Main", "GPUDeviceReset", cuda_status, "");

    return 0;
}
