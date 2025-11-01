#include <iostream>
#include <vector>
#include <cmath>

unsigned long ulps_between(float a, float b) {
    a = fabsf(a);
    b = fabsf(b);
    union { float f; int i; } uA, uB;
    uA.f = a; uB.f = b;
    if (uA.i > uB.i) return static_cast<unsigned long>(uA.i - uB.i);
    else return static_cast<unsigned long>(uB.i - uA.i);
}

void rotate_cpu(float2* vectors, int n, float angle_rad) {
    const float s = sinf(angle_rad);
    const float c = cosf(angle_rad);
    for (int i = 0; i < n; ++i) {
        float x_old = vectors[i].x;
        float y_old = vectors[i].y;
        vectors[i].x = x_old * c - y_old * s;
        vectors[i].y = x_old * s + y_old * c;
    }
}

__global__ void rotate_gpu(float2* vectors, int n, float angle_rad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float s = sinf(angle_rad);
        const float c = cosf(angle_rad);
        float x_old = vectors[idx].x;
        float y_old = vectors[idx].y;

        vectors[idx].x = x_old * c - y_old * s;
        vectors[idx].y = x_old * s + y_old * c;
    }
}

int main() {
    const int N = 1024 * 1024;
    const float alpha_deg = 37.7f;
    const float alpha_rad = alpha_deg * (M_PI / 180.0f);

    std::vector<float2> vectors_cpu(N);
    std::vector<float2> vectors_gpu_host(N);

    for (int i = 0; i < N; ++i) {
        vectors_cpu[i] = {static_cast<float>(i), static_cast<float>(N - i)};
        vectors_gpu_host[i] = vectors_cpu[i];
    }

    rotate_cpu(vectors_cpu.data(), N, alpha_rad);

    float2* vectors_gpu_dev;
    cudaMalloc(&vectors_gpu_dev, N * sizeof(float2));
    cudaMemcpy(vectors_gpu_dev, vectors_gpu_host.data(), N * sizeof(float2), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    rotate_gpu<<<blocksPerGrid, threadsPerBlock>>>(vectors_gpu_dev, N, alpha_rad);

    cudaMemcpy(vectors_gpu_host.data(), vectors_gpu_dev, N * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaFree(vectors_gpu_dev);

    unsigned long max_ulps_x = 0;
    unsigned long max_ulps_y = 0;

    for (int i = 0; i < N; ++i) {
        unsigned long ulps_x = ulps_between(vectors_cpu[i].x, vectors_gpu_host[i].x);
        unsigned long ulps_y = ulps_between(vectors_cpu[i].y, vectors_gpu_host[i].y);
        if (ulps_x > max_ulps_x) max_ulps_x = ulps_x;
        if (ulps_y > max_ulps_y) max_ulps_y = ulps_y;
    }

    std::cout << "Obliczenia dla " << N << " wektorow i kata " << alpha_deg << " stopni." << std::endl;
    std::cout << "Maksymalna roznica w ULP dla skladowej X: " << max_ulps_x << std::endl;
    std::cout << "Maksymalna roznica w ULP dla skladowej Y: " << max_ulps_y << std::endl;

    return 0;
}

