#include <iostream> 
#include <vector> 

#include "BlellochScan.h"
#include "aux.h"

#define DATA_SIZE 35

int main() {
	std::vector<T> scanInputs(DATA_SIZE);
	std::vector<T> scanOutputsCpu(DATA_SIZE);
	std::vector<T> scanOutputsGpu(DATA_SIZE);

	generateRandomData(scanInputs);

	blellochScanGpu(scanInputs, scanOutputsGpu);
	exclusiveScanCpu(scanInputs, scanOutputsCpu);

	verifyGpuScan(scanOutputsCpu, scanOutputsGpu);

	return 0;
}