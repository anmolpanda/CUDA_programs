#include "spdia.h"


__global__ void sparseDiagMatrixMultKernel(int *d_SDM, int *d_indices, int *d_inVec, 
							int *d_outVec, const int vecSize, const int indicesSize) {
	// syntactic sugar - splitting shared memory block into 
	// 					 2 shared memory arrays
	extern __shared__ int shared[];
  	int *ds_inVec = shared;
  	int *ds_outVec = &shared[blockDim.x];

	int tx = threadIdx.x;
	int tid = tx + blockIdx.x * blockDim.x;


	if (tid < vecSize) {
		ds_outVec[tx] = 0;
		for(int tileIndex = 0; tileIndex != (vecSize - 1) / TILE_WIDTH + 1; ++tileIndex) {
			ds_inVec[tx] = d_inVec[tx + tileIndex * TILE_WIDTH];
			__syncthreads();
			printf("Thread %d, tileIndex:: %d,  val: %d\n", 
					tid, tileIndex, ds_inVec[tx]);

			for (int i = 0; i != indicesSize; ++i){ // oops -> complexity
				//printf("Thread %d, i: %d,  val: %d\n", tx, i, d_SDM[tx + i * vecSize]);
				int diagInd = d_indices[i]; 
				if(-tid + TILE_WIDTH * tileIndex <= diagInd 
					&& diagInd < TILE_WIDTH * (tileIndex + 1) - tid) {
					printf("Thread: %d, tileIndex: %d, diagInd: %d, ds_inVec[diagInd + tid]: %d, d_SDM[i * vecSize + tid]: %d\n", 
							tid, tileIndex, diagInd, ds_inVec[diagInd + tid - tileIndex * TILE_WIDTH], 
							d_SDM[i * vecSize + tid]);
					ds_outVec[tx] += (d_SDM[i * vecSize + tid] * ds_inVec[diagInd + tid - tileIndex * TILE_WIDTH]);
				}
			}
			__syncthreads();
		}



		d_outVec[tid] = ds_outVec[tx];
	}
}

void sparseDiagMatrixMult(std::vector<int> &h_SDM, std::vector<int> &h_indices, 
	std::vector<int> &h_inVec, std::vector<int> &h_outVec) 
{
	const int vecSize = h_inVec.size();
	int *d_SDM;
	int *d_indices;
	int *d_inVec;
	int *d_outVec;

	const int SDM_BYTE_SIZE = h_SDM.size() * sizeof(int);
	const int VEC_BYTE_SIZE = h_inVec.size() * sizeof(int);
	const int VEC_INDICES_BYTE_SIZE = h_indices.size() * sizeof(int);
	
	cudaMalloc((void**)& d_SDM, SDM_BYTE_SIZE);
	cudaMalloc((void**)& d_inVec, VEC_BYTE_SIZE);
	cudaMalloc((void**)& d_outVec, VEC_BYTE_SIZE);
	cudaMalloc((void**)& d_indices, VEC_INDICES_BYTE_SIZE);
	
	cudaMemcpy(d_SDM, h_SDM.data(), SDM_BYTE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inVec, h_inVec.data(), VEC_BYTE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, h_indices.data(), VEC_INDICES_BYTE_SIZE, cudaMemcpyHostToDevice);

	dim3 grid((vecSize - 1) / TILE_WIDTH + 1);
	dim3 block(TILE_WIDTH);
	std::cout << "Threads per block: " << TILE_WIDTH << " blocks per grid: " 
			  <<  (vecSize - 1) / TILE_WIDTH + 1 << "\n";
	const int SH_MEM_BYTE_SIZE = 2 * TILE_WIDTH * sizeof(int);
	sparseDiagMatrixMultKernel<<<grid, block, SH_MEM_BYTE_SIZE>>>(d_SDM,
				d_indices, d_inVec, d_outVec, vecSize, h_indices.size());

	cudaMemcpy(h_outVec.data(), d_outVec, VEC_BYTE_SIZE, cudaMemcpyDeviceToHost);

	cudaFree(d_SDM);
	cudaFree(d_inVec);
	cudaFree(d_outVec);
	cudaFree(d_indices);
}