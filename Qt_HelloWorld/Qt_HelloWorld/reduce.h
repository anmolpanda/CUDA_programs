#include "cuda_runtime.h"

#include <vector>
#include <iostream>

__global__ void reduce_add_global_mem(float* d_in, float* d_out);

__global__ void reduce_add_shared_mem(float* d_in, float* d_out);

void reduce_add(std::vector<float>& data_vec, float* h_out, bool use_shared_mem);