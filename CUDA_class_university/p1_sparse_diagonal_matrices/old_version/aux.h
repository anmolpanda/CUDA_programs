#include <vector>
#include <iostream>
#include <limits>

void verify(std::vector<int> &host_ref, 
			std::vector<int> &gpu_ref);

void spMVdiaCPU(std::vector<int> &h_diagonals, 
				std::vector<int> &h_offsets, 
				std::vector<int> &h_inVec,	
				std::vector<int> &h_outVec);

void printVector(std::vector<int> &vec);

void printMatrix(std::vector<int> &vec, int cols, int rows);
