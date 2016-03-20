#include "aux.h"

void verify(std::vector<int> &host_ref, std::vector<int> &gpu_ref) {
	bool isCorrect = true;
	float epsilon = std::numeric_limits<int>::epsilon();
	epsilon *= 10000;

	for(unsigned int i = 0; i != host_ref.size(); ++i) {
		if(std::abs(host_ref[i] - gpu_ref[i]) > epsilon) {
			std::cout << "Fail at: " << i << " gpu value: " << gpu_ref[i] << "\n" << " cpu value: " << host_ref[i] << "\n";
			isCorrect = false;
			break;
		}
	}

	if (isCorrect) {
		std::cout << "OK. CPU result matches GPU result\n";
	}
	else {
		std::cout << "FAIL. CPU result doesn't match GPU result\n";
	}
}	


void spMVdiaCPU(std::vector<int> &h_diagonals, 
				std::vector<int> &h_offsets, 
				std::vector<int> &h_inVec,	
				std::vector<int> &h_outVec)
{
	int vectorSize = h_outVec.size();
	for (int i = 0; i != vectorSize; ++i) {
		int dot = 0;
		for (int j = 0; j != h_offsets.size(); ++j) { 
			int diagIdx = h_offsets[j];
			int col = i + diagIdx;
			int row = h_offsets.size() - j - 1;

			if (0 <= col && col < vectorSize) {
				dot += h_diagonals[i + j * vectorSize] * h_inVec[col];
			}
		}
		h_outVec[i] = dot;
	}
}

void printVector(std::vector<int> &vec) {
	for (auto element: vec) {
		std::cout << element << " ";
	}
	std::cout << "\n";
}

void printMatrix(std::vector<int> &vec, int cols, int rows) {
	for (int i = 0; i != rows; ++i) {
		for (int j = 0; j != cols; ++j) {
			std::cout << vec[i * cols + j] << " ";
		}
		std::cout << "\n";
	}
}

