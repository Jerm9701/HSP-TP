#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * p + col;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            Mout[idx] =  M1[idx] + M2[idx];
        }
    }
}

void MatrixInit(float* M, int n, int p) {
    // Create matrix on CPU with random values in {-1,1}
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            float randomValue = static_cast<float>(rand()) / RAND_MAX;
            M[i * p + j] = randomValue *2 -1;
        }
    }
}

void MatrixPrint(float* M, int n, int p){
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            printf("%.2f\t",M[i * p + j]);
        }
        printf("\n");
    }
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            Mout[i * p + j] =  M1[i*p + j] + M2[i*p + j];
        }
    }
}

int main() {
    const int n = 4; // Number of rows
    const int p = 3; // Number of columns
    const int matrixSize = n * p * sizeof(float);

    // Host matrices
    float *h_M1 = (float*)malloc(matrixSize);
    float *h_M2 = (float*)malloc(matrixSize);
    float *h_Mout = (float*)malloc(matrixSize);

    // Initialize matrices on the host
    srand(time(0));
    MatrixInit(h_M1, n, p);
    MatrixInit(h_M2, n, p);

    printf("Matrix M1:\n");
    MatrixPrint(h_M1, n, p);

    printf("\nMatrix M2:\n");
    MatrixPrint(h_M2, n, p);

    // Device matrices
    float *d_M1, *d_M2, *d_Mout;
    cudaMalloc((void**)&d_M1, matrixSize);
    cudaMalloc((void**)&d_M2, matrixSize);
    cudaMalloc((void**)&d_Mout, matrixSize);

    // Copy matrices from host to device
    cudaMemcpy(d_M1, h_M1, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, h_M2, matrixSize, cudaMemcpyHostToDevice);

    // Define thread and block dimensions
    dim3 blockDim(n*p, n*p);
    dim3 gridDim((p + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    cudaMatrixAdd<<<gridDim, blockDim>>>(d_M1, d_M2, d_Mout, n, p);
    //here gridDim = 1
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(h_Mout, d_Mout, matrixSize, cudaMemcpyDeviceToHost);

    printf("\nMatrix Mout (M1 + M2):\n");
    MatrixPrint(h_Mout, n, p);

    // Free device memory
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    // Free host memory
    free(h_M1);
    free(h_M2);
    free(h_Mout);

    return 0;
}