/*
    Created by: Andrew Sexton
          Date: March 21nd, 2022

    CSC258/458 - Parallel & Distributed Systems.
*/
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define min(a, b) a < b ? a : b

/* Use this macro to catch and print out runtime errors from the GPU */
/* Ex. cudaErrChk(cudaMalloc(...)) */
/*     cudaErrChk(cudaDeviceSynchronize()) */
#define cudaErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cout << "GPUAssert: " << cudaGetErrorString(code) << " " << file << " line " << line << std::endl;
        if (abort) { exit(code); }
    }
}

/* Vectorizable version of matrix multiplication for comparison */
void seq_matmul(const float* A, const float* B, float* C, int nsize) {
    float temp;
    for (int i = 0; i < nsize; i++) {
        for (int j = 0; j < nsize; j++) {
            temp = 0.0f;
            for (int k = 0; k < nsize; k++) {
                temp += A[k + (i * nsize)] * B[j + (k * nsize)];
            }
            C[j + (i * nsize)] = temp;
        }
    }
}

/* Simple OMP version of matrix multiplication for comparison */
void omp_matmul(const float* A, const float* B, float* C, int nsize) {
    # pragma omp parallel
    {
        float temp;
        # pragma omp for private(temp)
        for (int i = 0; i < nsize; i++) {
            for (int j = 0; j < nsize; j++) {
                temp = 0.0f;
                for (int k = 0; k < nsize; k++) {
                    temp += A[k + (i * nsize)] * B[j + (k * nsize)];
                }
                C[j + (i * nsize)] = temp;
            }
        }
    }
}

// Function for verifying values between two arrays
// by computing abs(x[i] - Y[i]) < EPSILON
void verify(const float* X, const float* Y, int nsize){
    float EPSILON = 1E-4;
    for(int i = 0; i < nsize; i++) {
        for(int j = 0; j < nsize; j++) {
            int idx = j + (i * nsize);

            if(std::fabs(X[idx] - Y[idx]) > EPSILON) {
                std::cout << std::setprecision(15) << "(" << i << ", " << j << "): " << X[idx] << " != " << Y[idx] << std::endl;
            }
        }
    }
}

// Print a comma-separated 2D array to stdout
void print_array(const float* arr, int nsize) {
    for(int i = 0; i < nsize; i++) {
        for(int j = 0; j < nsize; j++) {
            std::cout << arr[j + (i * nsize)];

            if(j < nsize) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;
    }
}

// GPU Kernel
__global__ void gpu_matmul(float* A, float* B, float* C, int nsize) {
    /* Add your code here */
    int j = threadIdx.x;    // Each thread takes care of 1 cell in 1 row (more than 1 cell if nsize > 1024)
    int i = blockIdx.x;     // Each block takes care of an entire row
    int I = i * nsize;
    float temp;
    for (; j < nsize; j+=1024){
        temp = 0.0f;
        for (int k = 0; k < nsize; k++) {
            temp += A[k + I] * B[j + (k * nsize)];
        }
        C[j + (i * nsize)] = temp;
    }
    /*===================*/
}


int main(int argc, char *argv[]) {
    if(argc < 2) {
        std::cout << "Invalid number of arguments: usage " << argv[0] << " <array size>" << std::endl;
        exit(0);
    }

    // Array size
    int nsize = std::atoi(argv[1]);

    // Timing Stuff
    timespec seq_start, seq_stop;
    timespec omp_start, omp_stop;
    timespec gpu_start, gpu_stop;

    // CPU side arrays
    // Arrays are long 1D, indexing is (i, j) => j + (i * nsize)
    // this gives a single index into the array using two loop variables
    float* A = new float[nsize * nsize]();
    float* B = new float[nsize * nsize]();
    float* C = new float[nsize * nsize]();

    // Fill CPU side arrays
    for(int i = 0; i < nsize; i++) {
        for(int j = 0; j < nsize; j++) {
            int idx = float(j + (i * nsize));
            A[idx] = idx + 1.0f;
            B[idx] = 1.0f / (idx + 1.0f);
        }
    }

    // Stop GPU timer
    clock_gettime(CLOCK_REALTIME, &gpu_start);

    /* Add your code here */

    // Allocate vectors in device memory
    float *d_A, *d_B, *d_C;
    size_t d_size = nsize * nsize * sizeof(float);
    cudaMalloc(&d_A, d_size);
    cudaMalloc(&d_B, d_size);
    cudaMalloc(&d_C, d_size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, A, d_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, d_size, cudaMemcpyHostToDevice);
    
    // Assigning nsize blocks & 1024 threads per block (nsize threads if nsize < 1024)
    // Each block will run on an entire row
    // Each thread will perform the calculations for 1 cell in 1 row
    // In case nsize > 1024, then each thread will work on more than 1 cell in that row
    gpu_matmul<<<nsize, min(1024, nsize)>>>(d_A, d_B, d_C, nsize);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, d_size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    /*===================*/


    // Stop GPU timer
    clock_gettime(CLOCK_REALTIME, &gpu_stop);	
    std::cout << "GPU Time: " << ((gpu_stop.tv_sec - gpu_start.tv_sec) + (gpu_stop.tv_nsec - gpu_start.tv_nsec) / 1E9) << '\n';

    // Compute Vectorized version
    // Modifies C in place.
    clock_gettime(CLOCK_REALTIME, &seq_start);
    seq_matmul(A, B, C, nsize);
    clock_gettime(CLOCK_REALTIME, &seq_stop);
    std::cout << "Seq (vectorized) Time: " << ((seq_stop.tv_sec - seq_start.tv_sec) + (seq_stop.tv_nsec - seq_start.tv_nsec) / 1E9) << '\n';

    // Compute OMP version
    // Modifies C in place.
    clock_gettime(CLOCK_REALTIME, &omp_start);
    omp_matmul(A, B, C, nsize);
    clock_gettime(CLOCK_REALTIME, &omp_stop);
    std::cout << "OMP Time: " << ((omp_stop.tv_sec - omp_start.tv_sec) + (omp_stop.tv_nsec - omp_start.tv_nsec) / 1E9) << '\n';
    
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
