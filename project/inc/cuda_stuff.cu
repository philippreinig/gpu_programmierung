#include <stddef.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>


#include "util.hpp"
#include "cuda_utils.cu"
#include "matrix.hpp"



__global__
void multiply_parallel_worker(double* M_1, uint m_1_rows, uint m_1_cols,
                              double* M_2, uint m_2_rows, uint m_2_cols,
                              double* M_res)
{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < m_1_rows * m_2_cols){
        int row = tid % m_1_rows;

        int col = tid % m_2_cols;

        for (uint k = 0; k < m_1_cols; ++k)
        {
            M_res[row*m_2_cols+col] += M_1[row*m_1_cols+k] * M_2[k*m_2_cols+col];
        }
    }
}

void multiply_parallel(double* M_1, uint m_1_rows, uint m_1_cols,
                        double* M_2, uint m_2_rows, uint m_2_cols,
                        double* M_res)
{
    int device_handle;
    int device_count;
    int max_threads_per_block;

    initDevice(device_handle, max_threads_per_block, device_count);

    cudaDeviceProp cudaDeviceProps;
    cudaGetDeviceProperties(&cudaDeviceProps, device_handle);

    // printDeviceProps(cudaDeviceProps);

    // std::cout << "Initalizing the device memory" << std::endl;

    double* M_1_device = nullptr;
    double* M_2_device = nullptr;
    double* M_res_device = nullptr;

    int m_1_size = sizeof(double) * m_1_rows * m_1_cols;
    int m_2_size = sizeof(double) * m_2_rows * m_2_cols;
    int m_res_size = sizeof(double) * m_1_rows * m_2_cols;

    // std::cout << "M_1: " << m_1_rows << "x" << m_1_cols << "=" << m_1_size << std::endl;
    // std::cout << "M_2: " << m_2_rows << "x" << m_2_cols << "=" << m_2_size << std::endl;
    // std::cout << "M_res: " << m_1_rows << "x" << m_2_cols << "=" << m_res_size << std::endl;


    cudaMalloc(&M_1_device, m_1_size);
    checkLastCudaError("cudaMalloc M_1_device");

    cudaMalloc(&M_2_device, m_2_size);
    checkLastCudaError("cudaMalloc M_2_device");

    cudaMalloc(&M_res_device, m_res_size);
    checkLastCudaError("cudaMalloc M_res_device");

    // std::cout << "M_1:" << M_1 << "-" << M_1+m_1_size << std::endl;
    // std::cout << "M_2:" << M_2 << "-" << M_2+m_2_size << std::endl;
    // std::cout << "M_res:" << M_res << "-" << M_res+m_res_size << std::endl;


    cudaMemcpy((void*)M_1_device, (void *)M_1, m_1_size, cudaMemcpyHostToDevice);
    checkLastCudaError("cudaMemcpy M_1_device");

    cudaMemcpy((void *)M_2_device, (void *)M_2, m_2_size, cudaMemcpyHostToDevice);
    checkLastCudaError("cudaMemcpy M_2_device");

    cudaMemcpy((void*) M_res_device, (void*) M_res, m_res_size, cudaMemcpyHostToDevice);
    checkLastCudaError("cudaMemcpy M_res_device");

    std::cout << "Device memory initialization successful" << std::endl;

    int amount_threads = m_1_rows * m_2_cols;

    int amount_blocks = (amount_threads % max_threads_per_block == 0) ? amount_threads / max_threads_per_block : amount_threads / max_threads_per_block + 1;

    // printf("Using %i threads across %i blocks, to calculate result matrix of size %ix%i\n", amount_threads, amount_blocks, m_1_rows, m_2_cols);

    multiply_parallel_worker<<<amount_blocks, max_threads_per_block>>>(M_1_device, m_1_rows, m_1_cols,
                                                                        M_2_device, m_2_rows, m_2_cols,
                                                                        M_res_device);
    cudaDeviceSynchronize();
    checkLastCudaError("kernel invocation failed: multiply_parallel_worker");

    
    checkLastCudaError("cudaDeviceSynchronize:");

    cudaMemcpy((void *)M_res, (void *)M_res_device, m_res_size, cudaMemcpyDeviceToHost);
    checkLastCudaError("cudaMemCpy: device -> host");

    cudaFree(M_1_device);
    checkLastCudaError("cudaFree M_1_device");

    cudaFree(M_2_device);
    checkLastCudaError("cudaFree M_2_device");

    cudaFree(M_res_device);
    checkLastCudaError("cudaFree M_res_device");
}
