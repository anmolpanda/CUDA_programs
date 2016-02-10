#include "mm.h"
#include <stdio.h>

#define TILE_WIDTH 16

typedef struct { 
	int width;
	int height;
	float* elements;
} Matrix;

__global__ void MatMultKernel(int m, int n, int k, int* d_A, int* d_B, int* d_C){
	__shared__ int ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ int ds_B[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	int Row = by * blockDim.y + ty;
	int Col = bx * blockDim.x + tx;
	int CValue = 0;

	for(int tileIndex = 0; tileIndex < n/TILE_WIDTH; ++tileIndex){
		ds_A[ty][tx] = d_A[Row * n + tileIndex * TILE_WIDTH + tx];
		ds_B[ty][tx] = d_B[(tileIndex * TILE_WIDTH + ty) * k + Col];
		__syncthreads();
		for(int i = 0; i < TILE_WIDTH; ++i){
			CValue += ds_A[ty][i] * ds_B[i][tx];
		}
		__syncthreads();
	}
	d_C[Row * k + Col] = CValue;
}

void MatMult(int A_size, int* h_A, int B_size, int* h_B, int C_size, int* h_C){
	int* d_A;
	int* d_B;
	int* d_C;
	cudaMalloc((void**) &d_A, A_size * A_size * sizeof(int));
	cudaMalloc((void**) &d_B, B_size * B_size * sizeof(int));
	cudaMalloc((void**) &d_C, C_size * C_size * sizeof(int));
	cudaMemcpy(d_A, h_A, A_size * A_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, B_size * B_size * sizeof(int), cudaMemcpyHostToDevice);

	const dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
  	const dim3 gridSize(1, 1, 1);
	MatMultKernel<<<gridSize, blockSize>>>(16, 16, 16, d_A, d_B, d_C);

	cudaMemcpy(h_C, d_C, C_size * C_size * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}