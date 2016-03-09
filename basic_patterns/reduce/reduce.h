#include "cuda_runtime.h"

#include <vector>
#include <iostream>
#include <algorithm>

#define TILE_WIDTH 1024

__global__ void reduceGpuKernel(int* d_in, int* d_out);

void reduceGpu(std::vector<int>& data_vec, int* h_out);

void reduceCpu(std::vector<int>& data_vec, int* h_out);