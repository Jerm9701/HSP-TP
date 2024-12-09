#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <cstdlib> 
#include <iostream>
#include <math.h>
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
__global__ void Conv2D(float* M_data, float* C1_kernel, float* C1_mat, int img_w, int nb_filter, int filter_sz){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int radius = filter_sz / 2;

	if((i < img_w) && (j < img_w))
	{
		float sum = 0;
		int k;

		int i_top_left = i-radius;
		int j_top_left = j-radius;

		for(k=0;k<filter_sz*filter_sz;k++)
		{
			int _i = i_top_left + (k/filter_sz); 
			int _j = j_top_left + (k%filter_sz); 

			if( (_i >= 0) && (_i < img_w) && (_j >= 0) && (_j < img_w))
			{
				int idx = _i * img_w + _j;
				sum += C1_kernel[k]*M_data[idx];
			}
		}

		C1_mat[i*img_w+j] = sum;
	}
}

 __global__ void Batch_norm(float* C1_mat, float* S1_data, int nb_filter, int C1_sz, int S1_sz){
 	int i = blockDim.x * blockIdx.x + threadIdx.x;
 	int j = blockDim.y * blockIdx.y + threadIdx.y;
 	if((i < S1_sz) && (j < S1_sz))
	{
		float sum = 0;

		if( (i >= 0) && (i < S1_sz) && (j >= 0) && (j < S1_sz))
 			{
				int idx = 2*i * C1_sz + j;
                int idy = (2*i+1) * C1_sz + j;
				sum += (C1_mat[idx]+C1_mat[idx+1]+C1_mat[idy]+C1_mat[idy]+1)/4;
			}
		

		S1_data[i*S1_sz+j] = sum;
	}
}
__global__ void activation_tanh(float *M,float *M_out, int length, int depth){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
 	int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    int index = k*length*length + j*length + i;
    if (i<length && j<length & k<<depth){
        M_out[index] = tanhf(M[index]);
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
void rd_initializeMatrix_3D(float* M3, int n, int p, int d) {
    for (int l = 0; l < n; ++l) {
        for (int i = 0; i < p; ++i) {
            for (int k = 0; k<d;k++){
                float randomValue = static_cast<float>(rand()) / RAND_MAX;
                M3[l*n+i * p + k]=randomValue;
            }
        }
    }
}

void MatrixInit(float* M, int n, int p) {
    // Create matrix on CPU with random values in [-1,1]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            float randomValue = static_cast<float>(rand()) / RAND_MAX;
            //M[i * p + j] = randomValue *2 -1; [-1,1]
            M[i * p + j] = randomValue; //[0,1]
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
    dim3 blockDim(4, 4);
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
    int img_w = 32;
    int nb_filter = 6;
    int C1_sz = 28;
    int S1_sz = 14;
    int C1_kernel = 5;
    float *M_data = (float*)malloc(img_w*img_w*sizeof(float));
    float *C1_mat = (float*)malloc(nb_filter*C1_sz*C1_sz*sizeof(float));
    float *C1_activated = (float*)malloc(nb_filter*C1_sz*sizeof(float));
    float *S1_data = (float*)malloc(nb_filter*S1_sz*S1_sz*sizeof(float));
    float *C1_mat_kernel = (float*)malloc(nb_filter*C1_kernel*C1_kernel*sizeof(float));
    MatrixInit(M_data,img_w,img_w);
    initializeMatrix_3D(C1_mat,nb_filter,C1_sz,C1_sz);
    initializeMatrix_3D(C1_activated,nb_filter,C1_sz,C1_sz);
    initializeMatrix_3D(S1_data,nb_filter,S1_sz,S1_sz);
    rd_initializeMatrix_3D(C1_mat_kernel,nb_filter,C1_kernel,C1_kernel);

    // Device matrices
    float *d_M_data, *d_C1_mat,*d_C1_activated, *d_S1_data, *d_C1_mat_kernel;
    cudaMalloc((void**)&d_M_data, img_w*img_w*sizeof(float));
    cudaMalloc((void**)&d_C1_mat, nb_filter*C1_sz*C1_sz*sizeof(float));
    cudaMalloc((void**)&d_C1_activated, nb_filter*C1_sz*C1_sz*sizeof(float));
    cudaMalloc((void**)&d_S1_data, nb_filter*S1_sz*S1_sz*sizeof(float));
    cudaMalloc((void**)&d_C1_mat_kernel, nb_filter*C1_kernel*C1_kernel*sizeof(float));
    
    cudaMemcpy(d_M_data, M_data, img_w*img_w*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_mat_kernel, C1_mat_kernel, nb_filter*C1_kernel*C1_kernel*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(4,4);
    dim3 gridDim((img_w + blockDim.x - 1) / blockDim.x, (img_w + blockDim.y - 1) / blockDim.y);

    Conv2D<<<gridDim,blockDim>>>(d_M_data,d_C1_mat_kernel,d_C1_mat,img_w,nb_filter,C1_kernel);
    activation_tanh<<<gridDim,blockDim>>>(d_C1_mat,d_C1_activated,C1_sz,nb_filter);
    cudaMemcpy(C1_mat, d_C1_mat, nb_filter*C1_sz*C1_sz*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(C1_activated, d_C1_activated, nb_filter*C1_sz*C1_sz*sizeof(float),cudaMemcpyDeviceToHost);

    //dim3 gridDim((C1_sz + blockDim.x - 1) / blockDim.x, (C1_sz + blockDim.y - 1) / blockDim.y);
    Batch_norm<<<gridDim,blockDim>>>(d_C1_activated,d_S1_data,nb_filter,C1_sz,S1_sz);
    cudaMemcpy(S1_data, d_S1_data,nb_filter*S1_sz*S1_sz*sizeof(float),cudaMemcpyDeviceToHost);

    printf("random init matrix 32x32 -- truncated for display--\n");
    MatrixPrint(M_data,img_w-8,img_w-8);
    printf("convolve matrix 6x5x5\n");
    MatrixPrint(C1_mat_kernel,C1_kernel,C1_kernel);
    printf("test convolved matrix 6x28x28 --truncated for display-- \n");
    for (int i = 0; i < C1_sz-8; ++i) {
        for (int j = 0; j < C1_sz-8; ++j) {
            printf("%.2f\t",C1_mat[2*C1_sz+i * C1_sz + j]);
        }
        printf("\n");
    }
    printf("test tanh activation 6x28x28 --truncated for display-- \n");
    for (int i = 0; i < C1_sz-8; ++i) {
        for (int j = 0; j < C1_sz-8; ++j) {
            printf("%.2f\t",C1_activated[2*C1_sz+i * C1_sz + j]);
        }
        printf("\n");
    }
    printf("test batchnorm matrix 6x14x14 --channel 1--\n");
    MatrixPrint(S1_data,S1_sz,S1_sz);
    // Free device memory
    cudaFree(d_M_data);
    cudaFree(d_C1_mat);
    cudaFree(d_C1_activated);
    cudaFree(d_S1_data);
    cudaFree(d_C1_mat_kernel);
    
    // Free Host memory
    cudaFree(M_data);
    cudaFree(C1_mat);
    cudaFree(C1_activated);
    cudaFree(S1_data);
    cudaFree(C1_mat_kernel);
    return 0;

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