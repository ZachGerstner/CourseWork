#include "im2Gray.h"

#define BLOCK 32



/*
 
  Given an input image d_in, perform the grayscale operation 
  using the luminance formula i.e. 
  o[i] = 0.224f*r + 0.587f*g + 0.111*b; 
  
  Your kernel needs to check for boundary conditions 
  and write the output pixels in gray scale format. 

  you may vary the BLOCK parameter.
 
 */
__global__ 
void im2Gray(uchar4 *d_in, unsigned char *d_grey, int numRows, int numCols)
{
	//declare shared mem
	extern __shared__ uchar4 pixel[];
	//set indexes
	int width = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;

	//check for bounds
	if(width < numCols && height < numRows)
	{
		//calculate gray scale 1D coordinate
		int greypoint = height * numCols + width;
		//calculate rgb positions
		uchar4 pixel = d_in[greypoint];
		//rescale and store 
		d_grey[greypoint] = 0.224f * pixel.x + 0.587f * pixel.y + 0.111 * pixel.z;
		__syncthreads();
	}
}




void launch_im2gray(uchar4 *d_in, unsigned char* d_grey, size_t numRows, size_t numCols){
    // configure launch params here 
    dim3 block(32,32,1);
    dim3 grid((numCols/32),(numRows/32),1);

    im2Gray<<<grid,block, numCols*sizeof(uchar4)>>>(d_in, d_grey, numRows, numCols);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    
}





