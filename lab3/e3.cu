#include <iostream>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
do {                                                                           \
    const cudaError_t error_code = call;                                       \
    if (error_code != cudaSuccess) {                                           \
        printf("CUDA Error:\n");                                               \
        printf("    File:       %s\n", __FILE__);                              \
        printf("    Line:       %d\n", __LINE__);                              \
        printf("    Error code: %d\n", error_code);                            \
        printf("    Error text: %s\n", cudaGetErrorString(error_code));         \
        exit(1);                                                               \
    }                                                                          \
} while (0)

const int UNROLL_FACTOR = 8;

__global__ void standard_loop(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < UNROLL_FACTOR; ++i) {
            sum += data[idx] * (i + 1.0f);
        }
        data[idx] = sum;
    }
}

__global__ void pragma_unroll_loop(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < UNROLL_FACTOR; ++i) {
            sum += data[idx] * (i + 1.0f);
        }
        data[idx] = sum;
    }
}

__global__ void manual_unroll_loop(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        float val = data[idx];
        sum += val * 1.0f;
        sum += val * 2.0f;
        sum += val * 3.0f;
        sum += val * 4.0f;
        sum += val * 5.0f;
        sum += val * 6.0f;
        sum += val * 7.0f;
        sum += val * 8.0f;
        data[idx] = sum;
    }
}

void time_kernel(void (*kernel)(float*, int), float* d_data, int n, const char* name) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    CHECK(cudaEventRecord(start));
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "Czas wykonania (" << name << "): " << milliseconds << " ms" << std::endl;

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
}

int main() {
    const int N = 1024 * 1024 * 16;
    float* h_data = new float[N];
    for (int i = 0; i < N; ++i) h_data[i] = 1.0f;

    float* d_data;
    CHECK(cudaMalloc(&d_data, N * sizeof(float)));

    CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    time_kernel(standard_loop, d_data, N, "Standardowa petla");

    CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    time_kernel(pragma_unroll_loop, d_data, N, "#pragma unroll");

    CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    time_kernel(manual_unroll_loop, d_data, N, "Reczne odwijanie");

    delete[] h_data;
    CHECK(cudaFree(d_data));

    return 0;
}

