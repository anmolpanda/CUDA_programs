#include <iostream> 
#include <vector> 
#include <random>
#include <iomanip>

#include "histogram.h"
#include "aux.h"

#define DATA_SIZE 1<<11
#define BINS 32

int main(){
	std::vector<int> histogramInputs(DATA_SIZE);
	std::vector<int> histogtamOutputsCpu(BINS);
	std::vector<int> histogtamOutputsGpu(BINS);

	// values 0..9 
	generateRandomData(histogtamInputs);

	localhistogramGpu(histogtamInputs, histogtamOutputsGpu);
	histogramCpu(histogtamInputs, histogtamOutputsCpu);

	//printVector(scanInputs);
	//printVector(scanOutputsCpu);
	//printVector(scanOutputsGpu);

	return 0;
}