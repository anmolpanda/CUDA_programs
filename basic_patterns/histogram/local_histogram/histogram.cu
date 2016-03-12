#include "histogram.h"

__global__ void localHistogramGpuKernel(int* d_in, int* d_out, int size){
	extern __shared__ int sh_mem[];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	const int n = (size - 1) / TILE_WIDTH + 1;
	for (int i = 0; i != n; ++i) {
		sh_mem[thr]
	}	
	atomicAdd(&(d_out[item]), 1);
}

void localHistogramGpu(std::vector<int>& h_in, std::vector<int>& h_out){
	const int IN_BYTE_SIZE = h_in.size() * sizeof(int);
	const int OUT_BYTE_SIZE = h_out.size() * sizeof(int);

	int* d_in;
	int* d_out;

	cudaMalloc((void**)& d_in, IN_BYTE_SIZE);
	cudaMalloc((void**)& d_out, OUT_BYTE_SIZE);

	cudaMemcpy(d_in, h_in.data(), IN_BYTE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, h_out.data(), IN_BYTE_SIZE, cudaMemcpyHostToDevice);

	dim3 grid(1);
	dim3 block(TILE_WIDTH);
	const int SH_MEM_BYTE_SIZE = h_out.size() * TILE_WIDTH * sizeof(int);
	histogramGpuKernel<<<grid, block, SH_MEM_BYTE_SIZE>>>(d_in, d_out, h_in.size());

	cudaMemcpy(h_out.data(), d_out, OUT_BYTE_SIZE, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);
}

void histogramCpu(std::vector<int> &input, std::vector<int> &bins) {
	for(auto element: input){
		bins[element]++;
	}
}