#include "cuda_runtime.h"
#include <stdio.h>

#include <vector>

#define TILE_WIDTH 128

// inclusive scan - Hills/Steele implementation. 
// O(log n) step complexity, O(n * log n) work complexity
__global__ void hillsSteeleScanGpuKernel(int* d_in, int* d_out, int size);

void hillsSteeleScanGpu(std::vector<int>& h_in, std::vector<int>& h_out);

void inclusiveScanCpu(std::vector<int> &h_in, std::vector<int> &h_out);