#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

__global__ void print(){
	int idx = blockIdx.x;
	printf("This is thread %d\n", idx); 
}

int main(){
	std::cout << "GPU manages in what order block will execute (GPU assigns blocks to SMs).\n"
			  << "This is an example where there are ~ 21 trilion "
			  << "(16 factorial) ways to execute the kernel\n";
 	dim3 GridSize(16,1,1);
 	dim3 BlockSize(1,1,1);
 	print<<<GridSize, BlockSize>>>();

 	// force the printf()s to flush
 	cudaDeviceSynchronize();
}