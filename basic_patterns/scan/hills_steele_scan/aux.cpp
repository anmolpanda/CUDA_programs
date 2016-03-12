#include "aux.h"

void verifyGpuHistogram(std::vector<int> &host_ref, std::vector<int> &gpu_ref) {
	bool isCorrect = true;
	float epsilon = std::numeric_limits<int>::epsilon();
	epsilon *= 10000;

	for(unsigned int i = 0; i != host_ref.size(); ++i) {
		if(std::abs(host_ref[i] - gpu_ref[i]) > epsilon) {
			std::cout << "Fail at: " << i << " value: " << gpu_ref[i] << "\n";
			isCorrect = false;
			break;
		}
	}

	if (isCorrect) {
		std::cout << "OK. GPU result matches CPU result\n";
	}
	else {
		std::cout << "FAIL. GPU result doesn't match CPU result\n";
	}
}	

void printVector(std::vector<int> &vec) {
	for (auto element: vec) {
		std::cout << element << " ";
	}
	std::cout << "\n";
}

void generateRandomData(std::vector<int> &vec) {
	std::default_random_engine dre;
	std::uniform_int_distribution<int> di(0, 9);
	for (auto &element: vec) {
		element = di(dre);
	}
}