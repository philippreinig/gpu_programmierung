#ifndef _A2_HPP_
#define _A2_HPP_

#include <cuda_runtime.h>
#include <cuda.h>

namespace a2{
/** @brief initializes the device to use.

  checks how many CUDA gpus are on the machine. If multiple are found it sets the first one to be used. It also querries how many threads per block can be used. If no gpu is found the deviceCount should be 0, the device_handle -1, and max_threads_per_block 0.
*/
void initDevice(int& device_handle, int& max_threads_per_block, int& deviceCount);

/** @brief allocates and initializes memory on the device

  allocates memory on the device and saves the pointer in data_dev. Size determines the number of integers to be reserved. The memory will be initialized to the value of the memory pointed to from data_host. data_host is assumed to be at least the size of data_dev.
*/
void initDeviceMemory(const int* data_host, int** data_dev, const int size);

/** @brief computes the reference solution in serial.

  computes the complete reduction of data_host single threaded on the CPU and writes the result to res. Returns the time spent in the reduction excluding any kind of memory transfer (none is needed).
*/
float reference(const int* data_host, const int size, int& res);


/** @brief the GPU kernel for the unoptimized, naive approch

  computes only one step of a reduction tree. No internal synchronization is performed. The stride is used to determine the step width of the tree.
  Example:
  data = {d1, d2, d3, d4, d5, d6, d7, d8}
  res = {d1, d2, d3, d4, d5, d6, d7, d8}
  reduction1<<<...>>>(data, res, 8, 1)
  -> res == {d1+d2, d2, d3+d4, d4, d5+d6, d6, d7+d8, d8}
  swap(data, res)
  reduction<<<...>>>(data, res, 8, 2)
  -> res == {d1+d2+d3+d4, d2, d3+d4, d4, d5+d6+d7+d8, d6, d7+d8, d8}
  swap(data, res)
  reduction<<<...>>>(data, res, 8, 4)
  -> res == {d1+d2+d3+d4+d5+d6+d7+d8, d2, d3+d4, d4, d5+d6+d7+d8, d6, d7+d8, d8}
*/
__global__
void reduction1(const int* data_dev, int* res, const int size, const int stride);

/** @brief computes the complete reduction of data using reduction1

  allocates memory on the GPU, copies the input data to the GPU, runs all the needed kernels (reduction1) (up until it is sufficient to finish up on CPU), writes the result to res and finaly cleans up the memory. The return value is the time spent in the reduction computation (excluding memory transfer) in seconds.
*/
float version1(const int* data_host, const int size, int& res);


/** @brief the GPU kernel for the naive shared memory version

  this computes a complete reduction within one (or two) threadblocks. It uses shared memory and includes synchronization on the GPU (on threadblock level). It does NOT avoid branch divergences. The result of one thread block is written at the address blockIdx.x.
  Example
  data = {d1, d2, d3, d4, d5, d6, d7, d8}
  res = {d1, d2, d3, d4, d5, d6, d7, d8}
  reduction2<<<2,4>>>(data, res, 8, 1)
  -> res == {d1+d2+d3+d4, d5+d6+d7+d8, _, _, _, _, _, _}
*/
__global__
void reduction2(const int* data_dev, int* res, const int size);

/** @brief computes the complete reduction of data using reduction2

  allocates memory on the GPU, copies the input data to the GPU, runs all the needed kernels (reduction2) (up until it is sufficient to finish up on CPU), writes the result to res and finaly cleans up the memory. The return value is the time spent in the reduction computation (excluding memory transfer) in seconds.
*/
float version2(const int* data_host, const int size, int& res);


/** @brief the GPU kernel for the shared memory version where branch divergence is handled.

  this function computes the same result as reduction2, however it avoids diverging branches, and thus, should be faster.
*/
__global__
void reduction3(const int* data_dev, int* res, const int size);
/** @brief computes the complete reduction of data using reduction2

  allocates memory on the GPU, copies the input data to the GPU, runs all the needed kernels (reduction3) (up until it is sufficient to finish up on CPU), writes the result to res and finaly cleans up the memory. The return value is the time spent in the reduction computation (excluding memory transfer) in seconds.
*/
float version3(const int* data_host, const int size, int& res);

};


#endif /* end of include guard: _A2_HPP_ */
