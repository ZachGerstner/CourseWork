#include "./gaussian_kernel.h" 

/*
The actual gaussian blur kernel to be implemented by 
you. Keep in mind that the kernel operates on a 
single channel.
 
 */
__global__ 
void gaussianBlur(unsigned char *d_in, unsigned char *d_out, 
        const int rows, const int cols, float *d_filter, int fWidth)
{
	//Establish indexes
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	//Ensure calculations are in bounds.
	if(column < cols && row < rows)
	{
		float pixelPlace = 0;
		int in_start_col = column - (fWidth/2);
		int in_start_row = row - (fWidth/2);
		
		for(int i = 0; i < fWidth; ++i)
		{
			for(int j = 0; j < fWidth; ++j)
			{
				int current_c = in_start_col + i;
				int current_r = in_start_row + j;
				if(current_r > -1 && current_r < rows && current_c > -1 && current_c < cols)
				{
					pixelPlace += (float)d_in[current_r * cols + current_c] * d_filter[i * fWidth + j];
				}
			}
		}
		//Write ouput.
		d_out[row * cols + column] = (unsigned char)pixelPlace;
	}
} 



/*
  Given an input RGBA image separate 
  that into appropriate rgba channels.
 */
__global__ 
void separateChannels(uchar4 *d_imrgba, unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, size_t rows, size_t cols)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(y < rows && x < cols)
	{
		int r = y * cols + x;
		d_r[r] = d_imrgba[r].x;
		d_g[r] = d_imrgba[r].y;
		d_b[r] = d_imrgba[r].z;	
	}
} 
 

/*
  Given input channels combine them 
  into a single uchar4 channel. 

  You can use some handy constructors provided by the 
  cuda library i.e. 
  make_int2(x, y) -> creates a vector of type int2 having x,y components 
  make_uchar4(x,y,z,255) -> creates a vector of uchar4 type x,y,z components 
  the last argument being the transperency value. 
 */
__global__ 
void recombineChannels(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, uchar4 *d_orgba, size_t rows, size_t cols)
{
	int index2= blockIdx.y * blockDim.y + threadIdx.y;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index2 < rows && index < cols)
	{
		int index3 = index2 * cols + index;
		d_orgba[index3].x = d_b[index3];
		d_orgba[index3].y = d_g[index3];
		d_orgba[index3].z = d_r[index3];
		d_orgba[index3].w = 255;
	}
} 

__global__ void gaussian_blur_seperable_row(unsigned char *d_in, unsigned char *d_out, const int rows, float *d_filter, int fWidth)
{
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(y < rows)
	{
		int current = y - (fWidth/2);
		float pixelVal = 0;
		for(int i = 0; i < rows; ++i)
		{
			if(current > -1 && current < rows)
			{
				pixelVal += (float)d_in[i * rows + current] * d_filter[i];
			}
		}
		d_out[y * current] = (unsigned char)pixelVal;
	}
}

__global__ void gaussian_blur_seperable_col(unsigned char *d_in, unsigned char *d_out, const int cols, float *d_filter, int fWidth)
{
	//Strided memory access 
	__extern__ __shared__ d_s_col[][]
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if(x < cols)
	{
		int current = x - (fWidth/2);
		float pixVal = 0;
		for(int j = 0; j < cols; ++j)
		{
			if(current > -1 && current < cols)
			{
				pixVal += (float)d_s_col[j * cols + current] * d_filter[j];
				
			}
		__syncthreads();
		}
		d_out[x * current] = (unsigned char)pixVal;
	}
}


void your_gauss_blur(uchar4* d_imrgba, uchar4 *d_oimrgba, size_t rows, size_t cols, 
        unsigned char *d_red, unsigned char *d_green, unsigned char *d_blue, 
        unsigned char *d_rblurred, unsigned char *d_gblurred, unsigned char *d_bblurred,
        float *d_filter,  int filterWidth){
 

int BLOCK = 32;
   dim3 blockSize(BLOCK,BLOCK,1);
   dim3 gridSize((cols/BLOCK),(rows/BLOCK),1);
   separateChannels<<<gridSize, blockSize>>>(d_imrgba, d_red, d_green, d_blue, rows, cols); 
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());  
/* gaussianBlur<<<gridSize, blockSize>>>(d_red, d_rblurred, rows, cols, d_filter, filterWidth); 
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());  
   gaussianBlur<<<gridSize, blockSize>>>(d_green, d_gblurred, rows, cols, d_filter, filterWidth);  
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());  
   gaussianBlur<<<gridSize, blockSize>>>(d_blue, d_bblurred, rows, cols, d_filter, filterWidth); 
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());
*/
   gaussian_blur_seperable_row<<<gridSize,blockSize>>>(d_red, d_rblurred, rows, d_filter, filterWidth);
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());
   gaussian_blur_seperable_row<<<gridSize,blockSize>>>(d_blue, d_bblurred, rows, d_filter, filterWidth);
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());
   gaussian_blur_seperable_row<<<gridSize,blockSize>>>(d_green, d_gblurred, rows, d_filter, filterWidth);
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());
   gaussian_blur_seperable_col<<<gridSize,blockSize>>>(d_red, d_rblurred, cols, d_filter, filterWidth);
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());
   gaussian_blur_seperable_col<<<gridSize,blockSize>>>(d_blue, d_bblurred, cols, d_filter, filterWidth);
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());
   gaussian_blur_seperable_col<<<gridSize,blockSize>>>(d_green, d_gblurred, cols, d_filter, filterWidth);
   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());   
   recombineChannels<<<gridSize, blockSize>>>(d_rblurred, d_gblurred, d_bblurred, d_oimrgba, rows, cols); 

   cudaDeviceSynchronize();
   checkCudaErrors(cudaGetLastError());   

}

