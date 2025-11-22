#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#include <iomanip>
#include <iostream>
#include <vector>

#define MAX_THREADS_PER_BLOCK 1024

typedef struct {
  double pi_value;
  double total_time_ms;
} GPUResult;

__global__ void wallis_kernel_multistream(int from, int to,
                                          double *partial_product_out) {
  __shared__ double partial_terms[MAX_THREADS_PER_BLOCK];

  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int local_idx = threadIdx.x;

  if (from + global_idx < to) {
    double n = (double)(from + global_idx + 1);
    partial_terms[local_idx] = (4.0 * n * n) / (4.0 * n * n - 1.0);
  } else {
    partial_terms[local_idx] = 1.0;
  }

  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (local_idx < stride) {
      partial_terms[local_idx] *= partial_terms[local_idx + stride];
    }
    __syncthreads();
  }

  if (local_idx == 0) {
    atomicAdd(partial_product_out, partial_terms[0] - 1.0);
  }
}

GPUResult compute_pi_gpu_multistream(int total_terms, int num_streams) {
  GPUResult result = {0};

  double *host_partial_products;
  double *device_partial_products;
  cudaHostAlloc(&host_partial_products, num_streams * sizeof(double),
                cudaHostAllocDefault);
  cudaMalloc(&device_partial_products, num_streams * sizeof(double));

  std::vector<cudaStream_t> streams(num_streams);
  for (int i = 0; i < num_streams; ++i) {
    cudaStreamCreate(&streams[i]);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int terms_per_stream = (total_terms + num_streams - 1) / num_streams;

  cudaMemsetAsync(device_partial_products, 0, num_streams * sizeof(double));
  cudaEventRecord(start);

  for (int i = 0; i < num_streams; ++i) {
    int start_term = i * terms_per_stream;
    int end_term = (i + 1) * terms_per_stream;
    if (end_term > total_terms) {
      end_term = total_terms;
    }

    if (start_term < end_term) {
      int chunk_size = end_term - start_term;
      int num_blocks =
          (chunk_size + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;

      double initial_value = 1.0;
      cudaMemcpyAsync(&device_partial_products[i], &initial_value,
                      sizeof(double), cudaMemcpyHostToDevice, streams[i]);

      wallis_kernel_multistream<<<num_blocks, MAX_THREADS_PER_BLOCK, 0,
                                  streams[i]>>>(start_term, end_term,
                                                &device_partial_products[i]);
    }
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  result.total_time_ms = elapsed_time;

  cudaMemcpy(host_partial_products, device_partial_products,
             num_streams * sizeof(double), cudaMemcpyDeviceToHost);

  double total_product = 1.0;
  for (int i = 0; i < num_streams; ++i) {
    total_product *= host_partial_products[i];
  }

  result.pi_value = 2.0 * total_product;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  for (int i = 0; i < num_streams; ++i) {
    cudaStreamDestroy(streams[i]);
  }
  cudaFreeHost(host_partial_products);
  cudaFree(device_partial_products);

  return result;
}

void print_gpu_results(const GPUResult &result, int num_terms,
                       int num_streams) {
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Streams: " << num_streams << " | Total Time: " << std::setw(8)
            << result.total_time_ms << " ms"
            << " | PI: " << std::fixed << std::setprecision(10)
            << result.pi_value << " | Abs Error: " << std::scientific
            << std::setprecision(4) << fabs(result.pi_value - M_PI)
            << std::endl;
  std::cout.unsetf(std::ios_base::floatfield);
}

int main(void) {
  const int total_terms = 5000000;
  const int MAX_STREAMS_TO_TEST = 8;

  std::cout << "Obliczanie PI za pomocą iloczynu Wallisa na GPU\n";
  std::cout << "Liczba terminów (N) = " << total_terms << "\n\n";
  std::cout << "--- Wyniki w zależności od liczby strumieni ---\n";

  for (int num_streams = 1; num_streams <= MAX_STREAMS_TO_TEST; ++num_streams) {
    GPUResult gpu_result = compute_pi_gpu_multistream(total_terms, num_streams);
    print_gpu_results(gpu_result, total_terms, num_streams);
  }

  std::cout << "\nTest zakończony.\n";

  return 0;
}
