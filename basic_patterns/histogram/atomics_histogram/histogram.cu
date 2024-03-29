#include "histogram.h"

__global__ void histogramGpuKernel(int* d_in, int* d_out){
	int thread = threadIdx.x + blockIdx.x * blockDim.x;
	int item = d_in[thread];
	atomicAdd(&(d_out[item]), 1);
}

void histogramGpu(std::vector<int>& h_in, std::vector<int>& h_out){
	const int IN_BYTE_SIZE = h_in.size() * sizeof(int);
	const int OUT_BYTE_SIZE = h_out.size() * sizeof(int);

	int* d_in;
	int* d_out;

	cudaMalloc((void**)& d_in, IN_BYTE_SIZE);
	cudaMalloc((void**)& d_out, OUT_BYTE_SIZE);

	cudaMemcpy(d_in, h_in.data(), IN_BYTE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, h_out.data(), IN_BYTE_SIZE, cudaMemcpyHostToDevice);

	dim3 grid((h_in.size() - 1) / TILE_WIDTH + 1);
	dim3 block(TILE_WIDTH);
	histogramGpuKernel<<<grid, block>>>(d_in, d_out);

	cudaMemcpy(h_out.data(), d_out, OUT_BYTE_SIZE, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);
}

void histogramCpu(std::vector<int> &input, std::vector<int> &bins) {
	for(auto element: input){
		bins[element]++;
	}
}