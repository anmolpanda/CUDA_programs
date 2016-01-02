#include <cuda_runtime.h>
#include <iostream>

#define ARRAY_SIZE 10
#define THREADS 100000

__global__ void increment_naive(int* array){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	i = i % 10;
	array[i] = array[i] + 1;
}

__global__ void increment_atomic(int* array){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	i = i % 10;
	atomicAdd(&array[i], 1);
}

void print(int* ptr){
	std::cout << "\n\nincrement result:\n[ ";
	for(int i = 0; i != ARRAY_SIZE; ++i) std::cout << ptr[i] << " ";
	std::cout << " ]\n";
}

int main(){
	// =======================NAIVE====================================
	//cuda event timer
	cudaEvent_t start1, stop1;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);


	int h_array[ARRAY_SIZE];
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

	int* d_array;
	cudaMalloc((void**) &d_array, ARRAY_BYTES);
	cudaMemset((void*) d_array, 0, ARRAY_BYTES);

	cudaEventRecord(start1, 0);
	increment_naive<<<THREADS / 1000, 1000>>>(d_array);
	cudaEventRecord(stop1, 0);

	cudaEventSynchronize(stop1);
	float msec1 = 0;
	cudaEventElapsedTime(&msec1, start1, stop1);

	cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(d_array);

	std::cout << "Naive add: \n";
	print(h_array);
	std::cout << "Elapsed time = " << msec1 << " ms\n\n";
	

	//====================ATOMICS=======================================
	cudaEvent_t start2, stop2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);

	cudaMalloc((void**) &d_array, ARRAY_BYTES);
	cudaMemset((void*) d_array, 0, ARRAY_BYTES);

	cudaEventRecord(start2, 0);
	increment_atomic<<<THREADS / 1000, 1000>>>(d_array);
	cudaEventRecord(stop2, 0);

	cudaEventSynchronize(stop2);
	float msec2 = 0;
	cudaEventElapsedTime(&msec2, start2, stop2);

	cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	cudaFree(d_array);

	std::cout << "Atomic add: \n";
	print(h_array);
	std::cout << "Elapsed time = " << msec2 << " ms\n\n";
}