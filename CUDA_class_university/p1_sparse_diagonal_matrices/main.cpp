#include <iostream>
#include <vector>

#include "spdia.h"
#include "aux.h"

int main() {
	const int size = 4;
	std::vector<int> indices {-3, 0, 2};
	std::vector<int> sparseDiagonalMatrix { 0, 0, 0, 5,
										    1, 1, 4, 2,
										    2, 3, 0, 0};
	std::vector<int> inputVector {1, 2, 3, 4};
	std::vector<int> outputVector(inputVector.size());

	sparseDiagMatrixMult(sparseDiagonalMatrix, indices, inputVector, 
						outputVector);

	std::cout << "Print input vector: ";
	printVector(inputVector);

	std::cout << "Print output vector: ";
	printVector(outputVector);

}