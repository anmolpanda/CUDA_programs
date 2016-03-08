#include <iostream> 
#include <vector> 

#include "HSScan.h"
#include "aux.h"

#define DATA_SIZE 128

int main() {
	std::vector<int> scanInputs(DATA_SIZE);
	std::vector<int> scanOutputsCpu(DATA_SIZE);
	std::vector<int> scanOutputsGpu(DATA_SIZE);

	generateRandomData(scanInputs);

	hillsSteeleScanGpu(scanInputs, scanOutputsGpu);
	inclusiveScanCpu(scanInputs, scanOutputsCpu);

	printVector(scanInputs);
	printVector(scanOutputsCpu);
	printVector(scanOutputsGpu);

	verifyGpuScan(scanOutputsCpu, scanOutputsGpu);

	return 0;
}