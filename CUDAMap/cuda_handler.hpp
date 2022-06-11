#pragma once

#include "cuda_runtime.h"

#include <stdexcept>
#include <vector>
#include <functional>
#include <string>
#include <iostream>

#include <math.h>

using std::string;
using std::vector;
using std::function;

//Consolidates calls and error handling in the main or calling kernel functions
inline cudaError_t WrapperFunction(function<cudaError_t()> func, string operation_name, string method_name, cudaError_t error, string optional_args) {
    cudaError_t cuda_status = error;
    if (cuda_status != cudaSuccess) {
        return cuda_status;
    }
    cuda_status = func();
    if (cuda_status != cudaSuccess) {
        std::cout << operation_name << " returned error code " << cuda_status << " after launching " << method_name << "\n" << std::endl;
        std::cout << "Error Stacktrace: " << cudaGetErrorString(cuda_status) << "\n" << std::endl;
        if (optional_args.size() > 0) {
            std::cout << "Additional Stacktrace: " << optional_args << std::endl;
        }
    }
    return cuda_status;
}
