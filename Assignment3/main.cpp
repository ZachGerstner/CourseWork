#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <string>
#include <chrono>
#include <cmath>
#include "utils.h"
#include "partial_sum.h"
using namespace std::chrono;

void linearPrefix(int *in, int *out, int size)
{
	int tmp[size];
	tmp[0] = in[0];
	for (int i = 1; i < size; ++i)
	{
		tmp[i] = tmp[i-1] + in[i];
	}
	out = tmp;
}

int main(int argc, char const **argv)
{
	int *d_out, *d_in;
	int h_in[SIZE], h_out[SIZE];
	int s_in[SIZE], s_out[SIZE];

	//populate randInt
	for (int i = 0; i < SIZE; ++i)
	{
		h_in[i] = rand() % 100 + 1;
	}
	for (int j = 0; j < SIZE; ++j)
	{
		s_in[j] = rand() % 100 + 1;
	}

	//Malloc 
	cudaMalloc((void **) &d_in, sizeof(int)*SIZE);
	cudaMalloc((void **) &d_out, sizeof(int)*SIZE);
	
	//Run and time linear prefix summation
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	linearPrefix(s_in, s_out, SIZE);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto computeTime = duration_cast<nanoseconds>( t2 - t1 ).count();
	std::cout << "Compute Time for Linear Prefix Sum : " << computeTime << "\n";
	
	//Memcpy to device
	checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(int)*SIZE, cudaMemcpyHostToDevice));
	
	//parallel prefix sum
	my_prefix_sum(d_in, d_out);
	
	//Memcpy to host
	checkCudaErrors(cudaMemcpy(&h_out, d_out, sizeof(int)*SIZE, cudaMemcpyDeviceToHost));
	
	//Free Memory
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}
