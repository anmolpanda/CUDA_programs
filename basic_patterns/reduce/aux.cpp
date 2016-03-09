#include "aux.h"

void printVector(std::vector<int> &vec) {
	for (auto element: vec) {
		std::cout << element << " ";
	}
	std::cout << "\n";
}

void generateRandomData(std::vector<int> &vec) {
	std::default_random_engine dre;
	std::uniform_int_distribution<int> di(0, 1);
	for (auto &element: vec) {
		element = di(dre);
	}
}