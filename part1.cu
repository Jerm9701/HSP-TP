#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <cstdlib> 
#include <iostream>

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
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * n + col;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Mout[i * n + j]=0;
            for (int k = 0; k<n;k++){
                Mout[i * n + j] +=  M1[i*n + k]*M2[k*n + j];
            }
            
        }
    }    
}

void initializeMatrix_3D(float* M3, int n, int p, int d) {
    for (int l = 0; l < n; ++l) {
        for (int i = 0; i < p; ++i) {
            for (int k = 0; k<d;k++){
                M3[l*n+i * p + k]=0;
            }
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

void MatrixMult(float *M1, float *M2, float *Mout, int n){
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Mout[i * n + j]=0;
            for (int k = 0; k<n;k++){
                Mout[i * n + j] +=  M1[i*n + k]*M2[k*n + j];
            }
            
        }
    }    
}

int GPU_p1(int n){
    //const int n = 10000; // Number of rows
    //const int p = 3; // Number of columns
    const int matrixSize = n * n * sizeof(float);
    //const int matrixSize = n * p * sizeof(float);

    // Host matrices
    float *M1 = (float*)malloc(matrixSize);
    float *M2 = (float*)malloc(matrixSize);
    float *Mout = (float*)malloc(matrixSize);

    // Initialize matrices on the host
    //MatrixInit(M1, n, p);
    //MatrixInit(M2, n, p);
    MatrixInit(M1, n, n);
    MatrixInit(M2, n, n);

    printf("Matrix M1:\n");
    //MatrixPrint(M1, n, p);
    //MatrixPrint(M1, n, n);
    printf("\nMatrix M2:\n");
    //MatrixPrint(M2, n, p);
    //MatrixPrint(M2, n, n);

    // Device matrices
    float *d_M1, *d_M2, *d_Mout;
    cudaMalloc((void**)&d_M1, matrixSize);
    cudaMalloc((void**)&d_M2, matrixSize);
    cudaMalloc((void**)&d_Mout, matrixSize);

    // Copy matrices from host to device
    cudaMemcpy(d_M1, M1, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, matrixSize, cudaMemcpyHostToDevice);

    // Define thread and block dimensions
    //dim3 blockDim(n*p, n*p);
    dim3 blockDim(n*n, n*n);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    cudaMatrixMult<<<gridDim, blockDim>>>(d_M1, d_M2, d_Mout, n);
    //here gridDim = 1
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(Mout, d_Mout, matrixSize, cudaMemcpyDeviceToHost);

    printf("\nMatrix Mout (M1 + M2):\n");
    //MatrixPrint(Mout, n, p);
    //MatrixPrint(Mout, n, n);

    // Free device memory
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    // Free host memory
    free(M1);
    free(M2);
    free(Mout);

    return 0;    

}

int CPU_p1(int n){
    //const int n = 1000; // Number of rows
    const int p = 3; // Number of columns
    const int matrixSize = n * n * sizeof(float);
    float *M1 = (float*)malloc(matrixSize);
    float *M2 = (float*)malloc(matrixSize);
    float *Mout = (float*)malloc(matrixSize);
    MatrixInit(M1, n, n);
    MatrixInit(M2, n, n);
    MatrixInit(Mout,n,n);

    printf("Matrix M1:\n");
    //MatrixPrint(M1, n, n);

    printf("\nMatrix M2:\n");
    //MatrixPrint(M2, n, n);

    MatrixMult(M1,M2,Mout,n);
    printf("\nMatrix Mout (M1 * M2):\n");
    //MatrixPrint(Mout, n, n);
    return 0;
}

int p2(){
    return 0;
    int rd_sz = 32;
    int mask_depth = 6;
    int C1_data = 28;
    int S1_sz = 14;
    int C1_kernel = 5;
    float *M_data = (float*)malloc(rd_sz*rd_sz*sizeof(float));
    float *C1_mat_data = (float*)malloc(mask_depth*C1_data*C1_data*sizeof(float));
    float *S1_data = (float*)malloc(mask_depth*S1_sz*S1_sz*sizeof(float));
    float *C1_mat_kernel = (float*)malloc(mask_depth*C1_kernel*C1_kernel*sizeof(float));
    MatrixInit(M_data,rd_sz,rd_sz);
    initializeMatrix_3D(C1_mat_data,mask_depth,C1_data,C1_data);
    initializeMatrix_3D(S1_data,mask_depth,S1_sz,S1_sz);
    initializeMatrix_3D(C1_mat_kernel,mask_depth,C1_kernel,C1_kernel);

}

// int main(int argc, char* argv[]) {
//     if (argc < 4) {
//         std::cerr << "n1 = GPU matrix size, n2 = CPU\n";
//         return 1;
//     }
//     int n1 = atoi(argv[1]);
//     time_t begin_GPU = time(NULL);
//     GPU(n1);
//     time_t end_GPU = time(NULL);
//     unsigned long secs_GPU = (unsigned long) difftime( end_GPU, begin_GPU );
//     printf("GPU_time for n=%d =\t %lu sec\n",n1,secs_GPU);
//     time_t begin_CPU = time(NULL);
//     int n2= atoi(argv[2]);
//     CPU(n2);
//     time_t end_CPU = time(NULL);
//     unsigned long secs_CPU = (unsigned long) difftime( end_CPU, begin_CPU );
//     printf("CPU_time for n=%d =\t %lu sec\n",n2,secs_CPU);
// }

int main(){
    p2();
    return 0;
}