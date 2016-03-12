#include "cuda_runtime.h"

#include <vector>

#define TILE_WIDTH 1024

__global__ void localHistogramGpuKernel(int* d_in, int* d_out, int size);

void localHistogramGpu(std::vector<int>& h_in, std::vector<int>& h_out);

void histogramCpu(std::vector<int> &input, std::vector<int> &bins);