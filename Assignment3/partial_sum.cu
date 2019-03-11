#include "./partial_sum.h"

//Function to perform out[i] = out[i-1] + A[i]
/*__global__ void InefficientPrefixSum(int *d_in, int *d_out)
{
	//Initialize index.
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int thread = threadIdx.x;
	__shared__ int tmp[SIZE];
	
	if(index < SIZE)
	{
		tmp[thread] += d_in[index];
	
		for(unsigned int strd = 1; strd <= thread; strd *= 2)
		{
			__syncthreads();
			int in1 = tmp[thread - strd];
			tmp[thread] += in1;
		}
		__syncthreads();
		if(index<SIZE)
		{
			d_out[index] = tmp[thread];
		}
	}
	printf("%i\n", d_out);
}
*/
__global__ void EfficientPrefixSum(int *d_in, int *d_out)
{
	//Initialize index
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int thread = threadIdx.x;

	//Shared memory
	__shared__ int s_mem[BLOCK*2]; 
	
	//Load shared memory
	if(index < (SIZE/2))
	{
		s_mem[thread * 2] = d_in[thread * 2];
		s_mem[thread * 2 + 1] = d_in[thread * 2 + 1];
	}
	
	//Reduction pass
	for(unsigned int strd = 1; strd <= BLOCK; strd *= 2)
	{
		int r_index = (thread + 1) * strd * 2 - 1;
		if (r_index < 2 * BLOCK)
		{
			s_mem[r_index] += s_mem[r_index - strd];
		}
			__syncthreads(); 
	}
		
	//Post reduction reverse pass
	for (unsigned int strd2 = BLOCK/2; strd2 > 0; strd2 /=2)
	{
		__syncthreads();
		int idx = (thread + 1) * strd2 * 2 - 1;
		if(idx + strd2 < 2 * BLOCK)
		{
			s_mem[idx + strd2] += s_mem[idx];
		}
	}
	__syncthreads();
	//Write result from shared to output
	if (index < SIZE)
	{
		d_out[index] = s_mem[thread];
	}
}




void my_prefix_sum(int *d_in, int *d_out)
{
	dim3 blockSize(BLOCK,1,1);
	dim3 gridSize((SIZE/BLOCK*2),1,1);
	//InefficientPrefixSum <<< gridSize, blockSize >>> (d_in, d_out);
	//cudaDeviceSynchronize();
	//checkCudaErrors(cudaGetLastError());
	EfficientPrefixSum <<< gridSize, blockSize >>> (d_in, d_out);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}
