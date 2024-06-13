#include "object_intersection.cuh"

__global__ void obj_intersect_kernel(int** base, int** C, int base_size, int C_size, int** result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < base_size) {
        for (int j = 0; j < C_size; j++) {
            int* b = base[tid];
            int* c = C[j];
            int* intersect = new int[base[tid][0]];
            int intersect_size = 0;
            for (int i = 0; i < base[tid][0]; i++) {
                for (int k = 0; k < c[0]; k++) {
                    if (b[i] == c[k]) {
                        intersect[intersect_size++] = b[i];
                        break;
                    }
                }
            }
            result[tid * C_size + j] = intersect;
        }
    }
}
