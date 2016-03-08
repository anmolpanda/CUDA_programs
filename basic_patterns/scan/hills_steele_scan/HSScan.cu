#include "HSScan.h"
#include <iostream>

__global__ void hillsSteeleScanGpuKernel(int* d_in, int* d_out, int size) {
	extern __shared__ int sh_mem [];
	int tx = threadIdx.x;
	int tid = tx + blockIdx.x * blockDim.x;

	if (tid < size) {
		sh_mem[tx] = d_in[tid];
		__syncthreads();

		int offset = 1;
		for (int s = blockDim.x - offset; s >= blockDim.x / 2; s = blockDim.x - offset) {
			if (tx < s) {
				int b = offset + tx;
				sh_mem[b] += sh_mem[tx];
				printf("Thread %d val: %d\n", tid, sh_mem[tx]);
			}
			offset <<= 1;
			__syncthreads();
		}

		d_out[tid] = sh_mem[tx];
	}
}

void hillsSteeleScanGpu (std::vector<int>& h_in, std::vector<int>& h_out) {
	int* d_in;
	int* d_out;	
	const int IN_BYTE_SIZE = h_in.size() * sizeof(int);
	const int SH_MEM_BYTE_SIZE = TILE_WIDTH * sizeof(int);

	cudaMalloc((void**)& d_in, IN_BYTE_SIZE);
	cudaMalloc((void**)& d_out, IN_BYTE_SIZE);

	cudaMemcpy(d_in, h_in.data(), IN_BYTE_SIZE, cudaMemcpyHostToDevice);


	hillsSteeleScanGpuKernel<<<1, TILE_WIDTH, SH_MEM_BYTE_SIZE>>>(d_in, d_out, h_in.size());

	cudaMemcpy(h_out.data(), d_out, IN_BYTE_SIZE, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);
}

void inclusiveScanCpu(std::vector<int> &h_in, std::vector<int> &h_out) {
	int acc = 0;

	for (unsigned int i = 0; i != h_in.size(); ++i) {
		acc += h_in[i];
		h_out[i] = acc;
	}
}