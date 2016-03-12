#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdio.h>

#define TILE_WIDTH 64

__global__ void sparseDiagMatrixMultKernel( int *d_SDM, int *d_inVec, 
											int *d_outVec, const int size);

void sparseDiagMatrixMult(std::vector<int> &h_SDM, std::vector<int> &h_indices, 
						std::vector<int> &h_inVec,	std::vector<int> &h_outVec);