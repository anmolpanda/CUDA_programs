#include "spdia.h"

__global__ void spMVdiaGPUKernel(const int *d_diagonals, 
								 const int *d_offsets, 
								 const int *d_inVec, 
								 int *d_outVec, 
								 const int vecSize, 
								 const int numOfDiags) 
{
	extern __shared__ int shared[];
  	int *ds_inVec = shared;
  	//int *ds_outVec = &shared[blockDim.x];

	int tx = threadIdx.x;
	int tid = tx + blockIdx.x * blockDim.x;

	// each thread computes dot product
	int dot = 0;

	for(int tileIndex = 0; tileIndex != gridDim.x; ++tileIndex) {
		int pos = tx + tileIndex * TILE_WIDTH;
		ds_inVec[tx] = pos < vecSize ? d_inVec[pos] : 0; // padding
		__syncthreads();

		if (tid < vecSize) {
			for (int i = 0; i != numOfDiags; ++i){
				int col = tid + d_offsets[i]; 
	
				if(TILE_WIDTH * tileIndex <= col && col < TILE_WIDTH * (tileIndex + 1)) {
					dot += d_diagonals[i * vecSize + tid] * ds_inVec[col - tileIndex * TILE_WIDTH];
					// if(tid == 999936) {
					// 	printf("tid: %d, tile: %d, offset: %d, col: %d, ds_inVec[tx]: %d, d_inVec[tid]: %d, dot: %d\n",
					// 		tid, tileIndex, d_offsets[i], col, ds_inVec[col - tileIndex * TILE_WIDTH], d_inVec[col], dot);
					// }
				}
			}
			__syncthreads();
		}
	}
	if (tid < vecSize) {
		d_outVec[tid] = dot;
	}
}

void spMVdiaGPU(std::vector<int> &h_diagonals, 
				std::vector<int> &h_offsets, 
				std::vector<int> &h_inVec,	
				std::vector<int> &h_outVec)
{
	int vecSize = h_inVec.size();
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
	const int SH_MEM_BYTE_SIZE = TILE_WIDTH * sizeof(int);
	spMVdiaGPUKernel<<<grid, block, SH_MEM_BYTE_SIZE>>>(d_diagonals,
				d_offsets, d_inVec, d_outVec, vecSize, h_offsets.size());

	cudaMemcpy(h_outVec.data(), d_outVec, VEC_BYTE_SIZE, cudaMemcpyDeviceToHost);

	cudaFree(d_diagonals);
	cudaFree(d_inVec);
	cudaFree(d_outVec);
	cudaFree(d_offsets);
}