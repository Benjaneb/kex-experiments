#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024         // Matrix dimensions (N x N)
#define TILE_SIZE 16   // Tile size for block-level kernel

// Thread-level GEMM kernel: each thread computes one element
__global__ void matmul_thread(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Block-level GEMM kernel using shared memory
__global__ void matmul_shared(const float *A, const float *B, float *C, int n) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    // Loop over all tiles needed to compute the C element
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile from A
        if (row < n && tile * TILE_SIZE + threadIdx.x < n)
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + tile * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Load tile from B
        if (col < n && tile * TILE_SIZE + threadIdx.y < n)
            tile_B[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * n + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        
        __syncthreads();
    }
    
    if (row < n && col < n)
        C[row * n + col] = sum;
}

// Block-level GEMM using shared memory with coalesced global reads
// and potentially fewer shared memory bank conflicts for B accesses.
__global__ void matmul_coalesced_shared(const float *A, const float *B, float *C, int n) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE]; // Stores B tile transposed

    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // Row in C this thread helps compute
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // Col in C this thread helps compute
    float sum = 0.0f;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile from A into shared memory (Coalesced read)
        int global_row_A = row; // blockIdx.y * TILE_SIZE + ty
        int global_col_A = tile * TILE_SIZE + tx;
        if (global_row_A < n && global_col_A < n)
            tile_A[ty][tx] = A[global_row_A * n + global_col_A];
        else
            tile_A[ty][tx] = 0.0f;

        // Load tile from B into shared memory, transposed (Coalesced read)
        int global_row_B = tile * TILE_SIZE + ty;
        int global_col_B = col; // blockIdx.x * TILE_SIZE + tx
         if (global_row_B < n && global_col_B < n)
            // Store into tile_B[tx][ty] (transposed)
            tile_B[tx][ty] = B[global_row_B * n + global_col_B];
        else
            tile_B[tx][ty] = 0.0f;

        __syncthreads(); // Ensure both tiles are loaded

        // Compute partial product using shared memory tiles
        // Access to tile_B[tx][k] should avoid bank conflicts
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[ty][k] * tile_B[tx][k];
        }

        __syncthreads(); // Ensure computation is done before next tile load
    }

    // Write final result to C in global memory
    if (row < n && col < n)
        C[row * n + col] = sum;
}

// Utility to initialize matrices with random values
void randomInitialize(float *mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    int size = N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // Initialize host matrices
    randomInitialize(h_A, N);
    randomInitialize(h_B, N);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    // --- Choose which kernel to run ---
    // For thread-level GEMM:
    // matmul_thread<<<blocks, threads>>>(d_A, d_B, d_C, N);
    
    // For block-level GEMM with shared memory:
    // matmul_shared<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // For block-level GEMM with shared memory and potentially fewer bank conflicts:
    matmul_coalesced_shared<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Print a sample element to verify results
    printf("Result sample (C[0]): %f\n", h_C[0]);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}