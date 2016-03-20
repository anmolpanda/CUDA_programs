#include <iostream>
#include <vector>
#include <assert.h>

#include "spdia.h"
#include "aux.h"

int main() {
	int numOfRows;
	int numOfDiagonals;
	std::cin >> numOfRows >> numOfDiagonals;
	std::cout << "numOfRows: " << numOfRows << 
				" and numOfDiagonals: " << numOfDiagonals << "\n";

	int diagMatrixSize = numOfDiagonals * numOfRows;
	assert(numOfRows > 0 && numOfRows <= 1000000000); //10^9
	assert(diagMatrixSize > 0 && diagMatrixSize <= 1000000000); //10^9

	std::vector<int> offsets(numOfDiagonals, 0);
	std::vector<int> diagonals(numOfDiagonals * numOfRows, 0);
	std::vector<int> inputVector(numOfRows, 0);
	std::vector<int> outputVectorGpu(numOfRows, 0);
	std::vector<int> outputVectorCpu(numOfRows, 0);

	for (int i = 0; i != numOfDiagonals; ++i) {
		int diagIndex;
		std::cin >> diagIndex;
		offsets[i] = diagIndex;
		for (int j = 0; j != numOfRows; ++j) {
			int entry;
			std::cin >> entry;
			diagonals[j + i * numOfRows] = entry;
		}
	}

	for (int i = 0; i != numOfRows; ++i) {
		int entry;
		std::cin >> entry;
		inputVector[i] = entry;
	}

	spMVdiaGPU(diagonals, offsets, inputVector, outputVectorGpu);
	spMVdiaCPU(diagonals, offsets, inputVector, outputVectorCpu);

	// std::cout << "Print diagonals: \n";
	// printMatrix(diagonals, numOfRows, numOfDiagonals);

	std::cout << "Print offsets: ";
	printVector(offsets);

	// std::cout << "Print input vector: ";
	// printVector(inputVector);

	// std::cout << "Print output vector GPU: ";
	// printVector(outputVectorGpu);

	// std::cout << "Print output vector CPU: ";
	// printVector(outputVectorCpu);

	std::cout << "Verify: ";
	verify(outputVectorCpu, outputVectorGpu);
}