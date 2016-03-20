#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdio.h>

#define TILE_WIDTH 256

__global__ void spMVdiaGPUKernel(const int *d_diagonals, 
								 const int *d_offsets, 
								 const int *d_inVec, 
								 int *d_outVec, 
								 const int vecSize, 
								 const int numOfDiags);

void spMVdiaGPU(std::vector<int> &h_diagonals, 
				std::vector<int> &h_offsets, 
				std::vector<int> &h_inVec,	
				std::vector<int> &h_outVec);