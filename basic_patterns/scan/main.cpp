#include <iostream> 
#include <vector> 
#include <random>
#include <iomanip>

#include "scan.h"

int main(){
	std::default_random_engine dre;
	std::uniform_real_distribution<float> di(0, 1.15236543645);

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

	// vector of float of size 2^20
	std::vector<float> scan_inputs(1024 * 1024, 0.0);
	std::vector<float> scan_outputs(1024 * 1024, 0.0);
	for(auto& ele: scan_inputs){
		ele = di(dre);
		sum += ele;
	}

	// serial inclusive scan, op = (+)
	float acc = 0.0;
	for(unsigned int i = 0; i != scan_inputs.size(); ++i){
		acc += scan_inputs[i];
		scan_outputs[i] = acc;
	}

	// serial exclusive scan, op = (+)
	float acc = 0.0;
	for(unsigned int i = 0; i != scan_inputs.size(); ++i){
		scan_outputs[i] = acc;
		acc += scan_inputs[i];
	}

	scan(scan_inputs, scan_outputs);

	return 0;
}