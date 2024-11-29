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

int main(){
    int n =5;
    int p = 5;
    float *M1, *M2, *Mout;
    float *d_M1, *d_M2, *d_Mout;
    M1   = (float*)malloc(sizeof(int) * n*p);
    M2   = (float*)malloc(sizeof(int) * n*p);
    Mout   = (float*)malloc(sizeof(int) * n*p);
    MatrixInit(M1,n,p);
    MatrixInit(M2,n,p);
    //MatrixAdd(M1,M2,Mout,n,p);
    //MatrixPrint(Mout,n,p);
    cudaMalloc((void**)&d_M1, sizeof(float)*n*p);
    cudaMalloc((void**)&d_M2, sizeof(float)*n*p);
    cudaMalloc((void**)&d_Mout, sizeof(float)*n*p);

    cudaMemcpy(d_M1,M1,sizeof(float)*n*p,cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2,M2,sizeof(float)*n*p,cudaMemcpyHostToDevice);

    cudaMatrixAdd<<<1,1>>>(d_M1,d_M2,d_Mout,n,p);
    printf("%f",d_Mout[0]);
    cudaMemcpy(M_out,d_Mout,sizeof(float)*n*p,cudaMemcpyDeviceToHost);

    cudaFree(d_M1);
    cudaFree(d_M2);
    MatrixPrint(Mout,n,p);
    cudaFree(d_Mout);

    
    exit(EXIT_SUCCESS);
}