#include "BlellochScan.h"
#include <iostream>

__global__ void fixup(T* d_out, T* d_aux, int size) {
	int tx = threadIdx.x;
	int tid = tx + blockDim.x * blockIdx.x;

	// map access pattern
	if (blockIdx.x > 0) {
		d_out[2 * tid] += d_aux[blockIdx.x];	
		d_out[2 * tid + 1] += d_aux[blockIdx.x];	
	}
}

__global__ void blellochScanGpuKernel(T* d_in, T* d_aux, int size) {
	extern __shared__ T sh_mem [];
	int tx = threadIdx.x;
	int tid = tx + blockIdx.x * blockDim.x;

	if (tid < size / 2 + 1) {
		// populate shared memory
		sh_mem[2 * tx] = d_in[2 * tid];
		sh_mem[2 * tx + 1] = d_in[2 * tid + 1];
		__syncthreads();

		// part 1 - reduce (up-sweep)
		int offset = 1;
		for (int s = blockDim.x; s > 0; s >>= 1) {
			if (tx < s) {
				int a = offset * (2 * tx + 1) - 1;
				int b = offset * (2 * tx + 2) - 1;
				sh_mem[b] += sh_mem[a];
			}
			offset <<= 1;
			__syncthreads();
		}

		// part 2 - store the value (the total sum of block i) to an auxiliary array
		if (tx == 0 && d_aux != 0) {
			d_aux[blockIdx.x] = sh_mem[blockDim.x * 2 - 1];
		}
				
		// part 3 - downsweep
		// clear last element
		if (tx == 0) {
			sh_mem[blockDim.x * 2 - 1] = 0;
		}
		__syncthreads(); //wait for thread 0

		if (d_aux == 0) {
			printf("\nThread %d: at %d value: %d, at %d value: %d\n", tid, 2 * tx,
			sh_mem[2 * tx], 2 * tx + 1, sh_mem[2 * tx + 1]);
		}

		for (int s = 1; s < blockDim.x * 2; s <<= 1){
			offset >>= 1;
			if (tx < s)
			{
				int a = offset * (2 * tx + 1) - 1;
				int b = offset * (2 * tx + 2) - 1;
				
				T temp = sh_mem[a];
				sh_mem[a] = sh_mem[b];
				sh_mem[b] += temp; 
			}
			__syncthreads();
		}

		// save back results 

		// printf("Thread %d: at %d value: %d, at %d value: %d\n", tid, 2 * tx,
		// sh_mem[2 * tx], 2 * tx + 1, sh_mem[2 * tx + 1]);

		d_in[2 * tid] = sh_mem[2 * tx];
		d_in[2 * tid + 1] = sh_mem[2 * tx + 1];
	}
}

void blellochScanGpu(std::vector<T> &h_in, std::vector<T> &h_out) {
	const int in_size = h_in.size();
	const int aux_size = (in_size - 1) / (2 * TILE_WIDTH);
	std::cout << "Aux size: " << aux_size << "\n";

	T* d_in;
	T* d_aux;

	const int IN_BYTE_SIZE = in_size * sizeof(T);

	cudaMalloc((void**)& d_in, IN_BYTE_SIZE);
	if (aux_size > 0) {
		cudaMalloc((void**)& d_aux, IN_BYTE_SIZE);
		cudaMemset(d_aux, 0, IN_BYTE_SIZE);
	}

	cudaMemcpy(d_in, h_in.data(), IN_BYTE_SIZE, cudaMemcpyHostToDevice);

	dim3 grid((in_size / 2 - 1) / TILE_WIDTH + 1);
	dim3 block(TILE_WIDTH);
	int sh_mem = TILE_WIDTH * 2 * sizeof(T);
	
	if (aux_size == 0) {
		blellochScanGpuKernel<<<grid, block, sh_mem>>>(d_in, 0, d_out, in_size);
		cudaDeviceSynchronize();
	}
	else {
		blellochScanGpuKernel<<<grid, block, sh_mem>>>(d_in, d_aux, in_size);
		cudaDeviceSynchronize();
		blellochScanGpuKernel<<<1, block, sh_mem>>>(d_aux, 0, TILE_WIDTH * 2);
		cudaDeviceSynchronize();
		fixup<<<grid, block>>>(d_in, d_aux, in_size);
		cudaDeviceSynchronize();
	}

	cudaMemcpy(h_out.data(), d_in, IN_BYTE_SIZE, cudaMemcpyDeviceToHost);

	cudaFree(d_aux);
	cudaFree(d_in);
}

void exclusiveScanCpu(std::vector<T> &h_in, std::vector<T> &h_out) {
	T acc = 0.0;

	for (unsigned int i = 0; i != h_in.size(); ++i) {
		h_out[i] = acc;
		acc += h_in[i];
	}
}