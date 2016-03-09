#include "HSScan.h"
#include <iostream>

__global__ void fixup(int* d_in, int* d_aux, int size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (blockIdx.x > 0 && tid < size) {
		//if(threadIdx.x == 0) printf("d_aux[%d] = %d\n", blockIdx.x - 1, d_aux[blockIdx.x - 1]);
		d_in[tid] += d_aux[blockIdx.x - 1];
	}
}

__global__ void hillsSteeleScanGpuKernel(int *d_in, int *d_aux, int size) {
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
			}
			offset <<= 1;
			__syncthreads();
		}
		d_in[tid] = sh_mem[tx];

		if (d_aux != NULL && tx == 0) {
			d_aux[blockIdx.x] = sh_mem[blockDim.x - 1];
		}
	}
}

void hillsSteeleScanGpu(std::vector<int>& h_in, std::vector<int>& h_out) {
	int *d_in;
	int *d_aux;

	const int AUX_SIZE = (h_in.size() - 1) / TILE_WIDTH;
	std::cout << AUX_SIZE << "\n";
	const int IN_BYTE_SIZE = h_in.size() * sizeof(int);
	const int SH_MEM_BYTE_SIZE = TILE_WIDTH * sizeof(int);

	cudaMalloc((void**)& d_in, IN_BYTE_SIZE);

	cudaMemcpy(d_in, h_in.data(), IN_BYTE_SIZE, cudaMemcpyHostToDevice);

	dim3 grid(AUX_SIZE + 1);
	dim3 block(TILE_WIDTH);

	if (AUX_SIZE == 0) {
		hillsSteeleScanGpuKernel<<<grid, block, SH_MEM_BYTE_SIZE>>>(d_in, NULL, h_in.size());
	}
	else {
		const int AUX_BYTE_SIZE = AUX_SIZE * sizeof(int);
		cudaMalloc((void**)& d_aux, AUX_BYTE_SIZE);
		hillsSteeleScanGpuKernel<<<grid, block, SH_MEM_BYTE_SIZE>>>(d_in, d_aux, h_in.size());
		cudaDeviceSynchronize();
		const int blockSize = AUX_SIZE + 1;
		hillsSteeleScanGpuKernel<<<1, blockSize, blockSize * sizeof(int)>>>(d_aux, NULL, blockSize);
		cudaDeviceSynchronize();
		fixup<<<grid, block>>>(d_in, d_aux, h_in.size());
		cudaFree(d_aux);
	}

	cudaMemcpy(h_out.data(), d_in, IN_BYTE_SIZE, cudaMemcpyDeviceToHost);
	cudaFree(d_in);
}

void inclusiveScanCpu(std::vector<int> &h_in, std::vector<int> &h_out) {
	int acc = 0;

	for (unsigned int i = 0; i != h_in.size(); ++i) {
		acc += h_in[i];
		h_out[i] = acc;
	}
}