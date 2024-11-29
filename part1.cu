#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            Mout[i * p + j] =  M1[i*p + j] + M2[i*p + j];
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
    M1   = (float*)malloc(sizeof(int) * n*p);
    M2   = (float*)malloc(sizeof(int) * n*p);
    Mout   = (float*)malloc(sizeof(int) * n*p);
    MatrixInit(M1,n,p);
    MatrixInit(M2,n,p);
    MatrixAdd(M1,M2,Mout,n,p);
    MatrixPrint(Mout,n,p);
    exit(EXIT_SUCCESS);
}