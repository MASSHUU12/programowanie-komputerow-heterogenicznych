#include <math.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>
#include <chrono>

#define MAX_THREADS_PER_BLOCK 1024
#define DEFAULT_EPSILON 1e-100

typedef struct
{
    double pi_value;
    double total_time_ms;
    double kernel_time_ms;
} GPUResult;

double get_time_ms()
{
    using namespace std::chrono;
    const auto now = steady_clock::now().time_since_epoch();
    const auto ms = duration_cast<duration<double, std::milli>>(now);
    return ms.count();
}

inline double wallis_term(int n)
{
    double n_sq = 4.0 * n * n;
    return n_sq / (n_sq - 1.0);
}

double wallis_seq(double eps, int *iterations_out)
{
    double product = 1.0;
    double pi_prev = 0.0;
    double pi_curr = 0.0;
    unsigned int n = 1;

    while (1)
    {
        product *= wallis_term(n);
        pi_curr = 2.0 * product;

        if (fabs(pi_curr - pi_prev) < eps)
        {
            break;
        }

        pi_prev = pi_curr;
        n++;
    }

    *iterations_out = n;
    return pi_curr;
}

__global__ void wallis_kernel(int from, int to, double *terms, double *partial_product)
{
    int i = threadIdx.x + from;

    if (i < to)
    {
        double n = (double)(i + 1);
        terms[i] = (4.0 * n * n) / (4.0 * n * n - 1.0);
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        double product = 1.0;
        for (int j = from; j < to; j++)
        {
            product *= terms[j];
        }
        *partial_product = 2.0 * product;
    }
}

GPUResult compute_pi_gpu(int total_terms)
{
    GPUResult result = {0};

    double *host_terms = nullptr;
    double *host_partial_product = nullptr;
    cudaHostAlloc(&host_terms, total_terms * sizeof(double), cudaHostAllocMapped);
    cudaHostAlloc(&host_partial_product, sizeof(double), cudaHostAllocMapped);

    double *device_terms = nullptr;
    double *device_partial_product = nullptr;
    cudaHostGetDevicePointer(&device_terms, host_terms, 0);
    cudaHostGetDevicePointer(&device_partial_product, host_partial_product, 0);

    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);

    const double start_time = get_time_ms();
    double total_product = 1.0;
    double kernel_time_total = 0.0f;

    for (int offset = 0; offset < total_terms; offset += MAX_THREADS_PER_BLOCK)
    {
        int current_chunk = (total_terms - offset > MAX_THREADS_PER_BLOCK)
                                ? MAX_THREADS_PER_BLOCK
                                : (total_terms - offset);

        cudaEventRecord(kernel_start);
        wallis_kernel<<<1, current_chunk>>>(
            offset,
            offset + current_chunk,
            device_terms,
            device_partial_product
        );
        cudaEventRecord(kernel_stop);
        cudaEventSynchronize(kernel_stop);

        float kernel_time = 0.0f;
        cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
        kernel_time_total += kernel_time;

        cudaDeviceSynchronize();
        total_product *= (*host_partial_product) / 2.0;
    }

    result.pi_value = 2.0 * total_product;
    result.total_time_ms = get_time_ms() - start_time;
    result.kernel_time_ms = kernel_time_total;

    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaFreeHost(host_terms);
    cudaFreeHost(host_partial_product);

    return result;
}

inline void print_cpu_results(double pi_value, int num_terms, double time_ms)
{
    std::cout << "[CPU] Sequential result:\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "PI: " << pi_value << "\n";
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "Absolute error = " << fabs(pi_value - M_PI) << "\n";
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "Number of terms (N) = " << num_terms << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU computation time = " << time_ms << " ms\n\n";
    std::cout.unsetf(std::ios_base::floatfield);
}

inline void print_gpu_results(const GPUResult& result, int num_terms)
{
    std::cout << "[GPU] Parallel result (CUDA):\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "PI: " << result.pi_value << "\n";
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "Absolute error = " << fabs(result.pi_value - M_PI) << "\n";
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "Number of terms (N) = " << num_terms << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Total GPU time = " << result.total_time_ms << " ms\n";
    std::cout << "GPU kernel time (sum of kernel durations) = " << result.kernel_time_ms << " ms\n\n";
    std::cout.unsetf(std::ios_base::floatfield);
}

inline void print_comparison(double pi_cpu, double cpu_time, const GPUResult& gpu_result, int num_terms)
{
    const double abs_diff = fabs(pi_cpu - gpu_result.pi_value);
    const double rel_error_gpu = fabs(gpu_result.pi_value - M_PI) / M_PI;

    std::cout << "Comparison:\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Reference PI = " << M_PI << "\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "CPU: PI = " << pi_cpu << " (time = " << std::setprecision(3) << cpu_time << " ms, terms = " << num_terms << ")\n";
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "GPU: PI = " << gpu_result.pi_value
              << " (total time = " << std::setprecision(3) << gpu_result.total_time_ms << " ms, "
              << "kernel = " << std::setprecision(3) << gpu_result.kernel_time_ms << " ms, terms = " << num_terms << ")\n";
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "Absolute difference (CPU - GPU) = " << abs_diff << "\n";
    std::cout << "GPU relative error w.r.t M_PI = " << rel_error_gpu << "\n";
    std::cout.unsetf(std::ios_base::floatfield);
}

int main(void)
{
    std::cout << "Computing PI using the Wallis product (CPU vs CUDA)\n\n";

    int num_terms_cpu = 0;
    const double start_cpu = get_time_ms();
    const double pi_cpu = wallis_seq(DEFAULT_EPSILON, &num_terms_cpu);
    const double cpu_time = get_time_ms() - start_cpu;

    print_cpu_results(pi_cpu, num_terms_cpu, cpu_time);

    GPUResult gpu_result = compute_pi_gpu(num_terms_cpu);
    print_gpu_results(gpu_result, num_terms_cpu);

    print_comparison(pi_cpu, cpu_time, gpu_result, num_terms_cpu);

    return 0;
}
