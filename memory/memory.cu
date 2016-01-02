#include <cuda_runtime.h>

__global__ void use_local_memory_GPU(float in){
	float f; // "f" is in local memory and private to each thread
	f = in;  // "in" is in local memory and private to each thread
}

__global__ void use_global_memory_GPU(float* array){
	array[threadIdx.x] = 2.0f * (float) threadIdx.x;
}


// trivial, doesn't do anything special, just shared mem demo
__global__ void use_shared_memory_GPU(float* array){
	int i = threadIdx.x;
	int index = threadIdx.x;
	float average = 0.0f;
	float sum = 0.0f;

	// shared memory, visible for all threads in thread block
	// it has same lifetime as the thread block
	__shared__ float sh_arr[128]; 

	// each thread reads 1 elemtn from global memroy
	// and write 1 element to the shared memory 
	sh_arr[index] = array[index];

	__syncthreads();

	// compute sum across all prev elements
	for(i = 0; i < index; ++i) sum += sh_arr[i];

	// compute average of all prev elements
	average = sum / (index + 1.0f);

	if(sh_arr[index] > average) array[index] = average;

	// this code has no effect. 
	// it modifes memory but it is never copied back to global memory
	// end when thread block terminates
	__syncthreads();
	sh_arr[index] = 3.14;
}


int main(){
	// LOCAL
	use_local_memory_GPU<<<1,128>>>(2.0f);


	// GLOBAL
	float h_arr[128];
	float* d_arr;
	cudaMalloc((void**) &d_arr, sizeof(float)*128);
	cudaMemcpy((void*) d_arr, (void*)h_arr, sizeof(float)*128, cudaMemcpyHostToDevice);
	use_global_memory_GPU<<<1,128>>>(d_arr);
	cudaMemcpy((void*) h_arr, (void*)d_arr, sizeof(float)*128, cudaMemcpyDeviceToHost); 


	// SHARED

	use_shared_memory_GPU<<<1, 128>>>(d_arr);
	cudaMemcpy((void*) h_arr, (void*)d_arr, sizeof(float)*128, cudaMemcpyDeviceToHost); 

	cudaFree(d_arr);
}