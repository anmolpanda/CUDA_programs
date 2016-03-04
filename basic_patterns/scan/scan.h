#include "cuda_runtime.h"
#include <stdio.h>

#include <vector>

#define TILE_WIDTH 4

// exclusive scan - Belloch implementation. 
// O(log n) step complexity, O(n) work complexity
__global__ void belloch_scan_gpu(float* d_in, float* d_out);

void belloch_scan(std::vector<float>& h_in, std::vector<float>& h_out);

// inclusive scan - Hills/Steele implementation. 
// O(log n) step complexity, O(n * log n) work complexity
__global__ void hills_steele_scan_gpu(float* d_in, float* d_out);

void hills_steele_scan(std::vector<float>& h_in, std::vector<float>& h_out);