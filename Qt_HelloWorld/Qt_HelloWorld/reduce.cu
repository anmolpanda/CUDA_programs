#include "reduce.h"

// power of 2
#define TILE_WIDTH 1024

__global__ void reduce_add_global_mem(float* d_in, float* d_out){
	int tx = threadIdx.x;
	int t = blockDim.x * blockIdx.x + tx;

	// reduction in global memory
	for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
		if(tx < s) d_in[t] += d_in[t + s];
		__syncthreads();
	}
	if(tx == 0) d_out[blockIdx.x] = d_in[t];
}

__global__ void reduce_add_shared_mem(float* d_in, float* d_out){
	int tx = threadIdx.x;
	int t = blockDim.x * blockIdx.x + tx;
	extern __shared__ float sh_mem[];
	sh_mem[tx] = d_in[t];
	__syncthreads();

	//reduction in shared memory
	for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
		if(tx < s) sh_mem[tx] += sh_mem[tx + s];
		__syncthreads();
	}
	if(tx == 0) d_out[blockIdx.x] = sh_mem[tx];
}

void reduce_add(std::vector<float>& h_in, float* h_out, bool use_shared_mem){
	float* d_in;
	float* d_out;
	float* d_final;

	const int IN_BYTE_SIZE = h_in.size() * sizeof(float);
	const int OUT_BYTE_SIZE = h_in.size() / TILE_WIDTH * sizeof(float);

	cudaMalloc((void**)& d_in, IN_BYTE_SIZE);
	cudaMalloc((void**)& d_out, OUT_BYTE_SIZE);
	cudaMalloc((void**)& d_final, sizeof(float));

	cudaMemcpy(d_in, h_in.data(), IN_BYTE_SIZE, cudaMemcpyHostToDevice);

	// int gridSize = h_in.size() / TILE_WIDTH;
	// int blockSize = TILE_WIDTH;

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	if(use_shared_mem){
        //printf("Running reduce with shared mem\n");
		reduce_add_shared_mem<<<1024, 1024, 1024 * sizeof(float)>>>(d_in, d_out);
		// gridSize = TILE_WIDTH;
		// blockSize = 1;
		reduce_add_shared_mem<<<1, 1024, 1024 * sizeof(float)>>>(d_out, d_final);
	}
	else{  
        //printf("Running global reduce\n");
		reduce_add_global_mem<<<1024, 1024>>>(d_in, d_out);
		// gridSize = TILE_WIDTH;
		// blockSize = 1;
		reduce_add_global_mem<<<1, 1024>>>(d_out, d_final);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
    float elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);    

	cudaMemcpy(h_out, d_final, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_final);

	std::cout << "GPU result: " << *h_out << " in " << elapsedTime << " ms.\n";
}
