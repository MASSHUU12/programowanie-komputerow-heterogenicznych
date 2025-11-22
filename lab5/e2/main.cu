#include <stdio.h>

#define LOOP_ITERATIONS 2000

__global__ void loop_default(long long *elapsedTime, float *result,
                             float multiplier) {
  float sum = 0.0f;

  clock_t start = clock();

  for (int i = 0; i < LOOP_ITERATIONS; ++i) {
    sum += i * multiplier;
  }

  clock_t stop = clock();

  *elapsedTime = (long long)(stop - start);
  *result = sum;
}

__global__ void loop_unrolled(long long *elapsedTime, float *result,
                              float multiplier) {
  float sum = 0.0f;

  clock_t start = clock();

#pragma unroll
  for (int i = 0; i < LOOP_ITERATIONS; ++i) {
    sum += i * multiplier;
  }

  clock_t stop = clock();

  *elapsedTime = (long long)(stop - start);
  *result = sum;
}

int main() {
  long long *d_elapsedTime;
  cudaMalloc(&d_elapsedTime, sizeof(long long));

  float *d_result;
  cudaMalloc(&d_result, sizeof(float));

  long long h_elapsedTime;

  float multiplier = 2.0f;

  printf("Eksperyment wpływu rozwijania pętli (#pragma unroll)\n");
  printf("Liczba iteracji w pętli: %d\n\n", LOOP_ITERATIONS);

  loop_default<<<1, 1>>>(d_elapsedTime, d_result, multiplier);
  cudaDeviceSynchronize();

  cudaMemcpy(&h_elapsedTime, d_elapsedTime, sizeof(long long),
             cudaMemcpyDeviceToHost);
  printf("Czas wykonania pętli ZWINIĘTEJ: %lld cykli zegara\n", h_elapsedTime);

  loop_unrolled<<<1, 1>>>(d_elapsedTime, d_result, multiplier);
  cudaDeviceSynchronize();

  cudaMemcpy(&h_elapsedTime, d_elapsedTime, sizeof(long long),
             cudaMemcpyDeviceToHost);
  printf("Czas wykonania pętli ROZWINIĘTEJ: %lld cykli zegara\n",
         h_elapsedTime);

  cudaFree(d_elapsedTime);
  cudaFree(d_result);

  return 0;
}
