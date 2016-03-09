#include "reduce.h"

__global__ void reduceGpuKernel(int* d_in, int* d_out) {
	extern __shared__ int sh_mem[];

	int tx = threadIdx.x;
	int t = blockDim.x * blockIdx.x + tx;
	
	sh_mem[tx] = d_in[t];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if(tx < s) sh_mem[tx] += sh_mem[tx + s];
		__syncthreads();
	}
	if (tx == 0) {
		d_out[blockIdx.x] = sh_mem[tx];
	}
}

void reduceGpu(std::vector<int>& h_in, int* h_out) {
	int* d_in;
	int* d_aux;
	int* d_out;

	const int AUX_SIZE = (h_in.size() - 1) / TILE_WIDTH;
	std::cout << AUX_SIZE << "\n";
	const int IN_BYTE_SIZE = h_in.size() * sizeof(int);
	const int SH_MEM_BYTE_SIZE = TILE_WIDTH * sizeof(int);

	cudaMalloc((void**)& d_in, IN_BYTE_SIZE);
	cudaMalloc((void**)& d_out, sizeof(int));

	cudaMemcpy(d_in, h_in.data(), IN_BYTE_SIZE, cudaMemcpyHostToDevice);

	dim3 grid(AUX_SIZE + 1);
	dim3 block(TILE_WIDTH);
	if (AUX_SIZE == 0) {
		reduceGpuKernel<<<grid, block, SH_MEM_BYTE_SIZE>>>(d_in, d_out);
	}
	else {
		const int AUX_BYTE_SIZE = AUX_SIZE * sizeof(int);
		cudaMalloc((void**)& d_aux, AUX_BYTE_SIZE);
		reduceGpuKernel<<<grid, block, SH_MEM_BYTE_SIZE>>>(d_in, d_aux);
		cudaDeviceSynchronize();
		reduceGpuKernel<<<1, AUX_SIZE + 1, AUX_BYTE_SIZE>>>(d_aux, d_out);
		cudaFree(d_aux);
	}

	cudaMemcpy(h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
}

void reduceCpu(std::vector<int> &h_in, int *h_out) {
	*h_out = std::accumulate(h_in.begin(), h_in.end(), 0);
}