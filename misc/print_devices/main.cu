#include <iostream>

#include <cuda.h>

int main() {
	int deviceCount {};
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		std::cerr << "Error: no devices supporting CUDA.\n";
		return -1;
	}

	for (int dev = 0; dev < deviceCount; ++dev) {
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
	return 0;
}