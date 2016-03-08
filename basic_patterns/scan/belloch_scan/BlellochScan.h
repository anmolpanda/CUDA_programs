#include "cuda_runtime.h"
#include <stdio.h>

#include <vector>

#define TILE_WIDTH 4

typedef int T;

// exclusive scan - Blelloch implementation. 
// O(log n) step complexity, O(n) work complexity
__global__ void blellochScanGpuKernel(T* d_in, T* d_aux, T* d_out, int size);

void blellochScanGpu(std::vector<T>& h_in, std::vector<T>& h_out);

void exclusiveScanCpu(std::vector<T> &h_in, std::vector<T> &h_out);