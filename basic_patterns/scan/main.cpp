#include <iostream> 
#include <vector> 
#include <random>
#include <iomanip>
#include <limits>

#include "scan.h"

bool verify_scan(std::vector<float>& host_ref, std::vector<float>& gpu_ref){
	bool if_ok = true;;
	float epsilon = std::numeric_limits<float>::epsilon();
	epsilon *= 100;
	for(unsigned int i = 0; i != host_ref.size(); ++i){
		std::cout << std::scientific << host_ref[i] - gpu_ref[i] << " ";
		if(std::abs(host_ref[i] - gpu_ref[i]) > epsilon){
			std::cout << "Fail at: " << i << " value: " << gpu_ref[i] << "\n";
			if_ok = false;
			break;
		}
	}
	std::cout << "\n";
	return if_ok;
}	


int main(){
	std::default_random_engine dre;
	std::uniform_real_distribution<float> di(0, 1.0);

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
	std::vector<float> scan_inputs(2 * TILE_WIDTH, 0.0);
	std::vector<float> scan_outputs_cpu(2 * TILE_WIDTH, 0.0);
	std::vector<float> scan_outputs_gpu(2 * TILE_WIDTH, 0.0);
	std::cout << "Data:\n";
	for(auto& ele: scan_inputs){
		ele = di(dre);
		std::cout << ele << " ";
	}
	std::cout << "\n";

	// // serial inclusive scan, op = (+)
	// float acc = 0.0;
	// for(unsigned int i = 0; i != scan_inputs.size(); ++i){
	// 	acc += scan_inputs[i];
	// 	scan_outputs[i] = acc;
	// }

	// serial exclusive scan, op = (+)
	std::cout << "Results - serial scan\n";
	float acc = 0.0;
	for(unsigned int i = 0; i != scan_inputs.size(); ++i){
		scan_outputs_cpu[i] = acc;
		std::cout << scan_outputs_cpu[i] << " ";
		acc += scan_inputs[i];
	}
	std::cout << "\n";

	// parallel exclusive scan, op = (+)
	belloch_scan(scan_inputs, scan_outputs_gpu);

	// verify
	if(verify_scan(scan_outputs_cpu, scan_outputs_gpu)){
		std::cout << "OK. CPU result matches GPU result\n";
	}
	else std::cout << "FAIL. CPU result doesn't match GPU result\n";

	return 0;
}