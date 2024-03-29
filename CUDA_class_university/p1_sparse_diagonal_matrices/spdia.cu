#include "spdia.h"

__global__ void spMVdiaGPUKernel(const int *d_diagonals, 
								 const int *d_offsets, 
								 const int *d_inVec, 
								 int *d_outVec, 
								 const int vecSize, 
								 const int numOfDiags) 
{
	int tx = threadIdx.x;
	int tid = tx + blockIdx.x * blockDim.x;

	if (tid < vecSize) {
		int dot = 0;

		for (int i = 0; i != numOfDiags; ++i){
			int diagInd = d_offsets[i]; 
			int col = tid + diagInd;

			if(0 <= col	&& col < vecSize) {
				dot += d_diagonals[i * vecSize + tid] * d_inVec[col];
			}
		}
		d_outVec[tid] = dot;
	}
}

void spMVdiaGPU(std::vector<int> &h_diagonals, 
				std::vector<int> &h_offsets, 
				std::vector<int> &h_inVec,	
				std::vector<int> &h_outVec)
{
	const int vecSize = h_inVec.size();
	int *d_diagonals;
	int *d_offsets;
	int *d_inVec;
	int *d_outVec;

	const int DIAGS_BYTE_SIZE = h_diagonals.size() * sizeof(int);
	const int VEC_BYTE_SIZE = h_inVec.size() * sizeof(int);
	const int OFFSETS_BYTE_SIZE = h_offsets.size() * sizeof(int);
	
	cudaMalloc((void**)& d_diagonals, DIAGS_BYTE_SIZE);
	cudaMalloc((void**)& d_inVec, VEC_BYTE_SIZE);
	cudaMalloc((void**)& d_outVec, VEC_BYTE_SIZE);
	cudaMalloc((void**)& d_offsets, OFFSETS_BYTE_SIZE);
	
	cudaMemcpy(d_diagonals, h_diagonals.data(), DIAGS_BYTE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inVec, h_inVec.data(), VEC_BYTE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_offsets, h_offsets.data(), OFFSETS_BYTE_SIZE, cudaMemcpyHostToDevice);

	dim3 grid((vecSize - 1) / TILE_WIDTH + 1);
	dim3 block(TILE_WIDTH);
	std::cout << "Threads per block: " << TILE_WIDTH << " blocks per grid: " 
			  <<  (vecSize - 1) / TILE_WIDTH + 1 << "\n";
	spMVdiaGPUKernel<<<grid, block>>>(d_diagonals,
				d_offsets, d_inVec, d_outVec, vecSize, h_offsets.size());

	cudaMemcpy(h_outVec.data(), d_outVec, VEC_BYTE_SIZE, cudaMemcpyDeviceToHost);

	cudaFree(d_diagonals);
	cudaFree(d_inVec);
	cudaFree(d_outVec);
	cudaFree(d_offsets);
}