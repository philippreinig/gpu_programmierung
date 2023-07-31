/// @file
////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Copyright (C) 2016/17      Christian Lessig, Otto-von-Guericke Universitaet Magdeburg
///
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  module     : Exercise 1
///
///  author     : lessig@isg.cs.ovgu.de
///
///  project    : GPU Programming
///
///  description: Cuda utility functions
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_

// includes, system
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>


template <typename T>
void checkError(T result, char const *const func, const char *const file, int const line)
{

    if (result)
    {
        auto errName = cudaGetErrorName(result);
        auto errString = cudaGetErrorString(result);
        std::cerr << "CUDA error at " << file << "::" << line << " with error code "
                  << static_cast<int>(result) << "(" << errName << " - " << errString << ") for " << func << "()." << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define checkErrorsCuda(val) checkError((val), #val, __FILE__, __LINE__)

inline void
checkLastCudaErrorFunc(const char *errorMessage, const char *file, const int line)
{

    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        std::cout << "Exiting program because of CUDA error" << std::endl;
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define checkLastCudaError(msg) checkLastCudaErrorFunc(msg, __FILE__, __LINE__)

////////////////////////////////////////////////////////////////////////////////////////////////////
//! Print device properties
////////////////////////////////////////////////////////////////////////////////////////////////////
void printDeviceProps(const cudaDeviceProp &devProp)
{

    printf("Major revision number:         %d\n", devProp.major);
    printf("Minor revision number:         %d\n", devProp.minor);
    printf("Name:                          %s\n", devProp.name);
    printf("Total global memory:           %iMB\n", (int)(devProp.totalGlobalMem / 1048576));
    printf("Total shared memory per block: %i\n", (int)devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n", devProp.regsPerBlock);
    printf("Warp size:                     %d\n", devProp.warpSize);
    printf("Maximum memory pitch:          %i\n", (int)devProp.memPitch);
    printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    {
        printf("Maximum block dimension %d:  %d\n", i, devProp.maxThreadsDim[i]);
    }
    for (int i = 0; i < 3; ++i)
    {
        printf("Maximum grid dimension %d:   %d\n", i, devProp.maxGridSize[i]);
    }
    printf("Clock rate:                    %d\n", devProp.clockRate);
    printf("Total constant memory:         %i\n", (int)devProp.totalConstMem);
    printf("Texture alignment:             %i\n", (int)devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
}

void initDevice(int &device_handle, int &max_threads_per_block, int &deviceCount)
{
    // std::cout << ">>> Initalizing the device" << std::endl;

    cudaGetDeviceCount(&deviceCount);
    checkLastCudaError("kernel invocation failed: reduction1");

    if (0 == deviceCount)
    {
        device_handle = -1;
        max_threads_per_block = 0;
    }
    else
    {
        device_handle = 0;

        cudaSetDevice(device_handle);

        cudaDeviceProp cudaDeviceProps;
        cudaGetDeviceProperties(&cudaDeviceProps, device_handle);

        checkLastCudaError("cudaGetDiveProperties failed");
        // std::cout << cudaDeviceProps.clockRate << std::endl;
        // std::cout << cudaDeviceProps.name << std::endl;

        // printDeviceProps(cudaDeviceProps);

        max_threads_per_block = cudaDeviceProps.maxThreadsPerBlock;
    }
}
#endif // _CUDA_UTIL_H_
