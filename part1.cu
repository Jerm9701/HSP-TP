#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctime>
void MatrixInit(int* M, int n, int p) {
    std::srand(std::time(nullptr)); // Seed the random number generator
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            M[i * p + j] = (std::rand() % 2 == 0) ? -1 : 1; // Randomly assign -1 or 1
        }
    }
}


int main(){
    int n =5;
    int p = 5;
    int *M;
    M   = (int*)malloc(sizeof(int) * n*p);
    MatrixInit(M,n,p);
    exit(EXIT_SUCCESS);
}