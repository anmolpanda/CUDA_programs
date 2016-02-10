#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

#include "mm.h"

void print_matrix(std::vector<int>& vec, int h, int w){
	std::cout << "\nmatrix output:\n";
	for(int i = 0; i != h; ++i){
		for(int j = 0; j!= w; ++j){
			std::cout << std::setw(12) << vec[i*w + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void print_matrix(int* vec, int h, int w){
	std::cout << "\nmatrix output:\n";
	for(int i = 0; i != h; ++i){
		for(int j = 0; j!= w; ++j){
			std::cout << std::setw(12) << vec[i*w + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int main(){
	const int A_height = 16;
	const int A_width = 16;
	const int B_height = 16;
	const int B_width = 16;
	const int C_height = 16;
	const int C_width = 16;
	std::vector<int> A(A_height * A_width, 0); // matrix A
	std::vector<int> B(B_height * B_width, 0); // matrix B
	std::vector<int> C(C_height * C_width, 0); // matrix C
	
	std::default_random_engine dre;
	std::uniform_int_distribution<int> di(0,1);

	// generate uniformly rondom number [0,1]
	for(int i = 0; i != A.size(); ++i) A.at(i) = di(dre); 
	
	// generate uniformly rondom number [0,1]
	for(int i = 0; i != B.size(); ++i) B.at(i) = di(dre);

	MatMult(16, A.data(), 16, B.data(), 16, C.data());

	print_matrix(A, A_height, A_width);
	print_matrix(B, B_height, B_width);
	print_matrix(C, C_height, C_width);
}