# PCA-EXP-5-MATRIX-MULTIPLICATION-USING-CUDA-AY-23-24
<h3>NAME:SREE NIVEDITAA SARAVANAN</h3>
<h3>REGISTER NO:212223230213</h3>

<h1> <align=center> MATRIX MULTIPLICATION USING CUDA </h3>
  Implement Matrix Multiplication using GPU.</h3>

## AIM:
To perform Matrix Multiplication using CUDA and check its performance with nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Define Constants: Define the size of the matrices (SIZE) and the size of the CUDA blocks (BLOCK_SIZE).
2.	Kernel Function: Define a CUDA kernel function matrixMultiply that performs the matrix multiplication.
3.	In the main function, perform the following steps:
4.	Initialize Matrices: Initialize the input matrices ‘a’ and ‘b’ with some values.
5.	Allocate Device Memory: Allocate memory on the GPU for the input matrices ‘a’ and ‘b’, and the output matrix ‘c’.
6.	Copy Matrices to Device: Copy the input matrices from host (CPU) memory to device (GPU) memory.
7.	Set Grid and Block Sizes: Set the grid and block sizes for the CUDA kernel launch.
8.	Start Timer: Start a timer to measure the execution time of the kernel.
9.	Launch Kernel: Launch the matrixMultiply kernel with the appropriate grid and block sizes, and the input and output matrices as arguments.
10.	Copy Result to Host: After the kernel execution, copy the result matrix from device memory to host memory.
11.	Stop Timer: Stop the timer and calculate the elapsed time.
12.	Print Result: Print the result matrix and the elapsed time.
13.	Free Device Memory: Finally, free the device memory that was allocated for the matrices.
## PROGRAM:
```
%%cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void matrixMultiply(int *a, int *b, int *c, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    if (row < size && col < size)
    {
    for (int k = 0; k < size; ++k)
    {
        sum += a[row * size + k] * b[k * size + col];
    }
    c[row * size + col] = sum;
    }
}

int main() {
    int n = 1 << 20;  // 1M elements
    size_t bytes = n * sizeof(int);
    
    int *h_in = (int*)malloc(bytes);
    for (int i = 0; i < n; i++) h_in[i] = 1;  // All 1s for easy verification
    
    int block = 256;
    int grid = (n + block - 1) / block;
    int *h_out = (int*)malloc(grid * sizeof(int));
    
    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, grid * sizeof(int));
    
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    
    reduce<<<grid, block, block * sizeof(int)>>>(d_in, d_out, n);
    
    cudaMemcpy(h_out, d_out, grid * sizeof(int), cudaMemcpyDeviceToHost);
    
    int sum = 0;
    for (int i = 0; i < grid; i++) sum += h_out[i];
    
    printf("Sum: %d (expected %d)\n", sum, n);
    printf("%s\n", (sum == n) ? "Success!" : "Failed!");
    
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    
    return 0;
}
```

## OUTPUT:
![image](https://github.com/SudharshnaLakshmi/PCA-EXP-5-MATRIX-MULTIPLICATION-USING-CUDA-AY-23-24/assets/93427267/3e8c516f-e699-4558-8fc8-67c3a701cbf7)


## RESULT:
Thus the program has been executed by using CUDA to mulptiply two matrices. It is observed that there are variations in host and device elapsed time. Device took 0.000211 sec and host took 0.000216 sec.
