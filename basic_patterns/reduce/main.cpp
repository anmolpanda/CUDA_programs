#include <iostream> 
#include <vector> 
#include <random>
#include <iomanip>

#include "reduce.h"
#include "aux.h"

#define DATA_SIZE 1<<21

int main() {
	std::vector<int> reduceInputs(DATA_SIZE);
	std::vector<int> scanOutputsCpu(DATA_SIZE);
	std::vector<int> scanOutputsGpu(DATA_SIZE);
	int reduceResultCpu;
	int reduceResultGpu;

	generateRandomData(reduceInputs);

	reduceGpu(reduceInputs, &reduceResultGpu);
	reduceCpu(reduceInputs, &reduceResultCpu);

	if (reduceResultCpu == reduceResultGpu) {
		std::cout << "PASS. GPU result matches CPU result.\n";
	}
	else {
		std::cout << "FAIL. GPU result matches CPU result.\n";
	}
	return 0;
}