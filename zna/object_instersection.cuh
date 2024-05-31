#ifndef OBJECT_INTERSECTION_CUH_
#define OBJECT_INTERSECTION_CUH_

#include <cuda_runtime.h>

__global__ void obj_intersect_kernel(int** base, int** C, int base_size, int C_size, int** result);

#endif /* OBJECT_INTERSECTION_CUH_ */
