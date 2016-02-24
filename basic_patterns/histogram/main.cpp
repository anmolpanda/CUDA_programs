#include <iostream> 
#include <vector> 
#include <random>
#include <iomanip>

#include "histogram.h"

int main(){
	std::default_random_engine dre;
	std::uniform_int_distribution<int> di(0, 99);

	int deviceCount {};
	cudaGetDeviceCount(&deviceCount);
	if(deviceCount == 0){
		std::cerr << "Error: no devices supporting CUDA.\n";
		return -1;
	}

	for(int dev = 0; dev < deviceCount; ++dev){
		cudaSetDevice(dev);
		cudaDeviceProp devProps;
		if(cudaGetDeviceProperties(&devProps, dev) == 0){
			std::cout << "Device       " << dev << "\n";
			std::cout << "GPU name:    " << devProps.name << "\n";
			std::cout << "global mem:  " << (unsigned int)devProps.totalGlobalMem/(1024*1024) << " MBytes" << "\n";
			std::cout << "compute cap: " << (int)devProps.major << "." << (int)devProps.minor << "\n";
			std::cout << "clock:       " << (int)devProps.clockRate << " kHz\n";
		}
	}

	std::vector<int> input(1024 * 1024, 0);
	std::vector<int> bins_cpu(128, 0);
	std::vector<int> bins_gpu(128, 0);
	for(auto& ele: input) ele = di(dre);

	// serial histogram
	for(unsigned int i = 0; i != input.size(); ++i){
		bins_cpu[input[i]]++;
	}

	// print cpu result
	std::cout << "CPU histogram:\n";
	for(unsigned int i = 0; i != bins_cpu.size(); ++i){
		std::cout << "Bin " << i << ": " << bins_cpu[i] << "\n";
	}

	simple_histogram(input, bins_gpu);

	//print gpu result
	std::cout << "GPU histogram:\n";
	for(unsigned int i = 0; i != bins_gpu.size(); ++i){
		std::cout << "Bin " << i << ": " << bins_gpu[i] << "\n";
	}

	return 0;
}