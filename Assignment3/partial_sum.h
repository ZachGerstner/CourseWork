#include <cuda_runtime.h>
#include <cuda.h>
#include "utils.h"
#define SIZE 10000
#define BLOCK 1024

void my_prefix_sum(int *in, int *out);
