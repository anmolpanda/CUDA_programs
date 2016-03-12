#include "cuda_runtime.h"

#include <vector>

#define TILE_WIDTH 1024

__global__ void simple_histogram(float* d_in, float* d_out);

void simple_histogram(std::vector<int>& h_in, std::vector<int>& h_out);

void histogramCpu(std::vector<int> &input, std::vector<int> &bins);