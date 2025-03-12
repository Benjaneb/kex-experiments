#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "timer.cuh"

#define SIZE 1000

void gemm(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] + b[k * n + j];
            }
            c[i * n + j] += sum;
        }
    }
}

float *randInitMat(int n) {
    float *m = (float*)malloc(n * n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[i * n + j] = (float)rand() / RAND_MAX;
        }
    }

    return m;
}

int main() {
    printf("Initializing matrices...\n");
    float *a = randInitMat(SIZE);
    float *b = randInitMat(SIZE);
    float *c = randInitMat(SIZE);
    printf("Init done!\n\n");

    clock_t startTime = startTimer();
    gemm(a, b, c, SIZE);
    stopTimer(startTime, "gemm");


    // Free memory
    free(a);
    free(b);
    free(c);

    return 0;
}
