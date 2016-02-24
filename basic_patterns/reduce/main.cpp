#include <iostream> 
#include <vector> 
#include <random>
#include <iomanip>

#include "reduce.h"

int main(){
	std::default_random_engine dre;
	std::uniform_real_distribution<float> di(-1.1503654,1.15236543645);

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
			std::cout << "Device " << dev << "\n";
			std::cout << "GPU name:    "  << devProps.name << "\n";
			std::cout << "global mem:  " << (unsigned int)devProps.totalGlobalMem/(1024*1024) << " MBytes" << "\n";
			std::cout << "compute cap: " << (int)devProps.major << "." << (int)devProps.minor << "\n";
			std::cout << "clock:       " << (int)devProps.clockRate << " kHz\n";
		}
	}

	float sum {0.0};
	// vector of float of size 2^20
	std::vector<float> data_vec(1024 * 1024);
	for(auto& ele: data_vec){
		ele = di(dre);
		sum += ele;
	}
	std::cout << "CPU result: " << std::fixed <<  std::setprecision(10) << sum << "\n";

	float gpu_sum {};
	reduce_add(data_vec, &gpu_sum, true);
	reduce_add(data_vec, &gpu_sum, false);

	return 0;
}