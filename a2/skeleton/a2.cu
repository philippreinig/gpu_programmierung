#include <a2.hpp>
#include <cuda_util.h>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace a2
{
	static int MAX_THREADS_PER_BLOCK = 0;

	/** @brief initializes the device to use

	  checks how many CUDA gpus are on the machine.
	  If multiple are found it sets the first one to be used.
	  It also querries how many threads per block can be used.
	  If no gpu is found the deviceCount should be 0, the device_handle -1, and max_threads_per_block 0.
	*/
	void initDevice(int &device_handle, int &max_threads_per_block, int &deviceCount)
	{
		// std::cout << ">>> Initalizing the device" << std::endl;

		cudaGetDeviceCount(&deviceCount);
		checkLastCudaErrorFunc("kernel invocation failed: reduction1", "a2.cu", -1);
		
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
			
			checkLastCudaErrorFunc("cudaGetDiveProperties failed", "a2.cu", -1);
				// std::cout << cudaDeviceProps.clockRate << std::endl;
				// std::cout << cudaDeviceProps.name << std::endl;
				
			// printDeviceProps(cudaDeviceProps);

			max_threads_per_block = cudaDeviceProps.maxThreadsPerBlock;
			MAX_THREADS_PER_BLOCK = max_threads_per_block;
		}

		// std::cout << "<<< Device initialization successful" << std::endl;
		
	}

	/** @brief allocates and initializes memory on the device

	  allocates memory on the device and saves the pointer in data_dev.
	  Size determines the number of integers to be reserved.
	  The memory will be initialized to the value of the memory pointed to from data_host.
	  Data_host is assumed to be at least the size of data_dev.
	*/
	void initDeviceMemory(const int* data_host, int **data_dev, const int size)
	{
		// std::cout << "Initalizing the device memory" << std::endl;

		cudaMalloc(data_dev, sizeof(int) * size);
		
		checkLastCudaErrorFunc("cudaMalloc", "a2.cu", -1);

		cudaMemcpy((void*) *data_dev, (void*) data_host, sizeof(int) * size, cudaMemcpyHostToDevice);

		checkLastCudaErrorFunc("cudaMemcpy", "a2.cu", -1);

		// std::cout << "Device memory initialization successful" << std::endl;

	}

	/** @brief computes the reference solution in serial

	  computes the complete reduction of data_host single threaded on the CPU and writes the result to res.
	  Returns the time spent in the reduction excluding any kind of memory transfer (none is needed).
	*/
	float reference(const int* data_host, const int size, int &res)
	{
		auto start = std::chrono::high_resolution_clock::now();

		int sum = 0;
		for (int i = 0; i < size; ++i)
		{
			sum += data_host[i];
		}

		res = sum;

		auto end = std::chrono::high_resolution_clock::now();

		auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		auto duration_sec = duration_ms / 1000.0f;

		return duration_sec;
	}

	__global__
	void reduction1(const int *data_dev, int *res, const int size, const int stride)
	{
		int tid = blockDim.x * blockIdx.x  + threadIdx.x;
		res[tid] = data_dev[2*tid] + data_dev[2*tid+1];
		// printf("%d: %d + %d = %d\n", tid, data_dev[2*tid], data_dev[2*tid+1], res[tid]);
	}

	/** @brief computes the complete reduction of data using reduction1

	  allocates memory on the GPU,
	  copies the input data to the GPU,
	  runs all the needed kernels (reduction1) (up until it is sufficient to finish up on CPU),
	  writes the result to res and finaly cleans up the memory.
	  The return value is the time spent in the reduction computation (excluding memory transfer) in seconds.
	*/
	float version1(const int* data_host, const int size, int &res)
	{
		int device_handle;
		int device_count;
		int max_threads_per_block;
		initDevice(device_handle, max_threads_per_block, device_count);

		int size_r = size;
		int* data_r = new int[size_r];
		std::memcpy((void*) data_r, (void*) data_host, sizeof(int) * size);
		
		auto start = std::chrono::high_resolution_clock::now();
		
		for(unsigned int iter = 1; size_r > 1; ++iter, size_r/=2){
			// printf("Iteration %u\n", iter);

			int* data_dev = nullptr;

			initDeviceMemory(data_r, &data_dev, size_r);

			int amount_threads = size_r/2;

			int amount_blocks = max(1, amount_threads / max_threads_per_block);

			int threads_per_block = amount_threads / amount_blocks;

			// std::cout << "Calculating sum with " << amount_threads << " threads in " << amount_blocks << " blocks ->"
			// << threads_per_block << " threads per block" << std::endl;

			int* result_device;
			int* result = new int[amount_threads];

			cudaMalloc(&result_device, sizeof(int) * amount_threads);

			checkLastCudaErrorFunc("cudaMalloc", "a2.cu", -1);

			std::cout << "array_size: " << size << "(" << (size % 2 == 0) << ")" <<
						 "blocks: " << amount_blocks << ", " <<
						 "threads per block: " << threads_per_block << std::endl;


			reduction1<<<amount_blocks,threads_per_block>>>(data_dev, result_device, size, 0);

			checkLastCudaErrorFunc("kernel invocation failed: reduction1", "a2.cu", -1);

			cudaMemcpy((void*) result, (void*) result_device, sizeof(int) * amount_threads, cudaMemcpyDeviceToHost);
			
			checkLastCudaErrorFunc("cudaMemCpy", "a2.cu", -1);

			cudaFree(data_dev);
			checkLastCudaErrorFunc("cudaFree result_device", "a2.cu", -1);

			cudaFree(result_device);
			checkLastCudaErrorFunc("cudaFree result_device", "a2.cu", -1);

			free(data_r);

			data_r = new int[size_r/2];

			std::memcpy((void*) data_r, (void*) result, sizeof(int) * amount_threads);

			free(result);
		}

		auto end = std::chrono::high_resolution_clock::now();

		auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		auto duration_sec = duration_ms / 1000.0f;

		res = data_r[0];

		return duration_sec;

		

		// for(int i = 0; i < size/2; ++i){
		// 	std::cout << i << ": " << result[i] << std::endl;
		// }



		return 0;
	}

	/** @brief the GPU kernel for the naive shared memory version

	  this computes a complete reduction within one (or two) threadblocks. It uses shared memory and includes synchronization on the GPU (on threadblock level). It does NOT avoid branch divergences. The result of one thread block is written at the address blockIdx.x.
	  Example
	  data = {d1, d2, d3, d4, d5, d6, d7, d8}
	  res = {d1, d2, d3, d4, d5, d6, d7, d8}
	  reduction2<<<2,4>>>(data, res, 8, 1)
	  -> res == {d1+d2+d3+d4, d5+d6+d7+d8, _, _, _, _, _, _}
	*/
	__global__ void reduction2(const int *data_dev, int *res, const int size)
	{
	}

	/** @brief computes the complete reduction of data using reduction2

	  allocates memory on the GPU, copies the input data to the GPU, runs all the needed kernels (reduction2) (up until it is sufficient to finish up on CPU), writes the result to res and finaly cleans up the memory. The return value is the time spent in the reduction computation (excluding memory transfer) in seconds.
	*/
	float version2(const int *data_host, const int size, int &res)
	{
		return 0;
	}

	/** @brief the GPU kernel for the shared memory version where branch divergence is handled. Also increases the arithmetic density

	  this function computes the same result as reduction2, however it avoids diverging branches and increases the arithmetic density (aka one threadblock works on more than 1024 elements e.g. double the amount).
	*/
	__global__ void reduction3(const int *data_dev, int *res, const int size)
	{
	}
	/** @brief computes the complete reduction of data using reduction3

	  allocates memory on the GPU, copies the input data to the GPU, runs all the needed kernels (reduction3) (up until it is sufficient to finish up on CPU), writes the result to res and finaly cleans up the memory. The return value is the time spent in the reduction computation (excluding memory transfer) in seconds.
	*/
	float version3(const int *data_host, const int size, int &res)
	{
		return 0;
	}

};
