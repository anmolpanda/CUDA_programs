#include <vector>
#include <iostream>
#include <limits>
#include <random>

typedef int T;

void verifyGpuScan(std::vector<T> &host_ref, std::vector<T> &gpu_ref);

void printVector(std::vector<T> &vec);

void generateRandomData(std::vector<T> &vec);