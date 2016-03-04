#include "scan.h"
#include <iostream>

__global__ void belloch_scan_gpu(float* d_in, float* d_out, int size)
{
	extern __shared__ float sh_mem [];
	int tx = threadIdx.x;
	int tid = tx + blockIdx.x * blockDim.x;

	if(tid < size){
		// populate shared memory
		sh_mem[2 * tx] = d_in[2 * tid];
		sh_mem[2 * tx + 1] = d_in[2 * tid + 1];
		__syncthreads();

		// part 1 - reduce
		int offset = 1;
		for(int s = blockDim.x; s > 0; s >>= 1){
			if(tx < s){
				int a = offset * (2 * tx + 1) - 1;
				int b = offset * (2 * tx + 2) - 1;
				sh_mem[b] += sh_mem[a];
			}
			offset <<= 1;
			__syncthreads();
		}

		printf("Thread %d, pos %d, value %f \nThread %d, pos %d, value %f \n",
			tx, 2*tx, sh_mem[2 * tx], tx, 2 * tx + 1, sh_mem[2 * tx + 1]);
		// clear last element
		sh_mem[blockDim.x * 2 - 1] = 0;
		
		// part 2 - downsweep
		for(int s = 1; s < blockDim.x * 2; s <<= 1){
			offset >>= 1;
			if(tx < s)
			{
				int a = offset * (2 * tx + 1) - 1;
				int b = offset * (2 * tx + 2) - 1;
				
				float temp = sh_mem[a];
				sh_mem[a] = sh_mem[b];
				sh_mem[b] += temp; 
			}
			__syncthreads();
		}

		// save back results 
		d_out[2 * tid] = sh_mem[2 * tx];
		d_out[2 * tid + 1] = sh_mem[2 * tx + 1];
	}
}

__global__ void hills_steele_scan_gpu(float* d_in, float* d_out, int size)
{
	extern __shared__ float sh_mem [];
	int tx = threadIdx.x;
	int tid = tx + blockIdx.x * blockDim.x;

	if(tid < size){
		// // populate shared memory
		// sh_mem[2 * tx] = d_in[2 * tid];
		// sh_mem[2 * tx + 1] = d_in[2 * tid + 1];
		// __syncthreads();

		// // part 1 - reduce
		// int offset = 1;
		// for(unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
		// 	if(tx < s){
		// 		int a = offset * (2 * tx + 1) - 1;
		// 		int b = offset * (2 * tx + 2) - 1;
		// 		sh_mem[b] += sh_mem[a];
		// 	}
		// 	offset <<= 1;
		// 	__syncthreads();
		// }

		// // clear last element
		// sh_mem[blockDim.x - 1] = 0;
		
		// // part 2 - downsweep
		// for(unsigned int s = 1; s < blockDim.x; s <<= 1){
		// 	if(tx < s)
		// 	{
		// 		int a = offset * (2 * tx + 1) - 1;
		// 		int b = offset * (2 * tx + 2) - 1;
				
		// 		float temp = sh_mem[a];
		// 		sh_mem[a] = sh_mem[b];
		// 		sh_mem[b] += temp; 
		// 	}
		// 	__syncthreads();
		// }

		// // save back results 
		// d_out[2 * tid] = sh_mem[2 * tx];
		// d_out[2 * tid + 1] = sh_mem[2 * tx + 1];
	}
}

void belloch_scan(std::vector<float>& h_in, std::vector<float>& h_out)
{
	float* d_in;
	float* d_out;	
	const int IN_BYTE_SIZE = h_in.size() * sizeof(float);
	cudaMalloc((void**)& d_in, IN_BYTE_SIZE);
	cudaMalloc((void**)& d_out, IN_BYTE_SIZE);

	cudaMemcpy(d_in, h_in.data(), IN_BYTE_SIZE, cudaMemcpyHostToDevice);

	const int sh_mem = TILE_WIDTH * 2 * sizeof(float);
	belloch_scan_gpu<<<1, TILE_WIDTH, sh_mem>>>(d_in, d_out, h_in.size());

	cudaMemcpy(h_out.data(), d_out, IN_BYTE_SIZE, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);
}

void hills_steele_scan(std::vector<float>& h_in, std::vector<float>& h_out){
	float* d_in;
	float* d_out;	
	const int IN_BYTE_SIZE = h_in.size() * sizeof(float);
	cudaMalloc((void**)& d_in, IN_BYTE_SIZE);
	cudaMalloc((void**)& d_out, IN_BYTE_SIZE);

	cudaMemcpy(d_in, h_in.data(), IN_BYTE_SIZE, cudaMemcpyHostToDevice);

	hills_steele_scan_gpu<<<1, TILE_WIDTH>>>(d_in, d_out, h_in.size());

	cudaMemcpy(h_out.data(), d_out, IN_BYTE_SIZE, cudaMemcpyDeviceToHost);

	cudaFree(d_out);
	cudaFree(d_in);
}

