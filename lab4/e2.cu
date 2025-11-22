#include <stdio.h>

__global__ void MyKernel(float *out, float *in, int size) {
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix < size)
    out[ix] = in[ix] * 2.f;
}

int main(void) {
  const int num_elements_per_stream = 8192;
  const int num_streams = 2;
  const int total_elements = num_elements_per_stream * num_streams;
  const int size_per_stream_bytes = num_elements_per_stream * sizeof(float);
  const int total_size_bytes = total_elements * sizeof(float);

  float *hostPtr;
  cudaMallocHost((void **)&hostPtr, total_size_bytes);

  float *inputDevPtr, *outputDevPtr;
  cudaMalloc(&inputDevPtr, total_size_bytes);
  cudaMalloc(&outputDevPtr, total_size_bytes);

  for (int i = 0; i < total_elements; ++i) {
    hostPtr[i] = (float)i;
  }

  printf("Rozpoczynanie obliczeń z użyciem 2 strumieni...\n");
  cudaStream_t stream[num_streams];
  for (int i = 0; i < num_streams; ++i)
    cudaStreamCreate(&stream[i]);

  cudaEvent_t start_2_streams, stop_2_streams;
  cudaEventCreate(&start_2_streams);
  cudaEventCreate(&stop_2_streams);

  cudaEventRecord(start_2_streams, 0);

  for (int i = 0; i < num_streams; ++i)
    cudaMemcpyAsync(inputDevPtr + i * num_elements_per_stream,
                    hostPtr + i * num_elements_per_stream,
                    size_per_stream_bytes, cudaMemcpyHostToDevice, stream[i]);

  for (int i = 0; i < num_streams; ++i)
    MyKernel<<<100, 512, 0, stream[i]>>>(
        outputDevPtr + i * num_elements_per_stream,
        inputDevPtr + i * num_elements_per_stream, num_elements_per_stream);

  for (int i = 0; i < num_streams; ++i)
    cudaMemcpyAsync(hostPtr + i * num_elements_per_stream,
                    outputDevPtr + i * num_elements_per_stream,
                    size_per_stream_bytes, cudaMemcpyDeviceToHost, stream[i]);

  cudaEventRecord(stop_2_streams, 0);
  cudaEventSynchronize(stop_2_streams);

  float time_2_streams;
  cudaEventElapsedTime(&time_2_streams, start_2_streams, stop_2_streams);

  bool bCorrect = true;
  for (int i = 0; i < total_elements; ++i) {
    float expected_value = (float)i * 2.f;
    if (hostPtr[i] != expected_value) {
      bCorrect = false;
      break;
    }
  }
  if (bCorrect)
    printf("Weryfikacja dla 2 strumieni pomyślna.\n");
  else
    printf("Weryfikacja dla 2 strumieni nie powiodła się.\n");

  printf("\nRozpoczynanie obliczeń z użyciem 1 strumienia...\n");

  for (int i = 0; i < total_elements; ++i) {
    hostPtr[i] = (float)i;
  }

  cudaEvent_t start_1_stream, stop_1_stream;
  cudaEventCreate(&start_1_stream);
  cudaEventCreate(&stop_1_stream);

  cudaEventRecord(start_1_stream, 0);
  cudaMemcpyAsync(inputDevPtr, hostPtr, total_size_bytes,
                  cudaMemcpyHostToDevice, 0);

  MyKernel<<<200, 512, 0, 0>>>(outputDevPtr, inputDevPtr, total_elements);

  cudaMemcpyAsync(hostPtr, outputDevPtr, total_size_bytes,
                  cudaMemcpyDeviceToHost, 0);

  cudaEventRecord(stop_1_stream, 0);
  cudaEventSynchronize(stop_1_stream);

  float time_1_stream;
  cudaEventElapsedTime(&time_1_stream, start_1_stream, stop_1_stream);

  bCorrect = true;
  for (int i = 0; i < total_elements; ++i) {
    float expected_value = (float)i * 2.f;
    if (hostPtr[i] != expected_value) {
      bCorrect = false;
      break;
    }
  }
  if (bCorrect)
    printf("Weryfikacja dla 1 strumienia pomyślna.\n");
  else
    printf("Weryfikacja dla 1 strumienia nie powiodła się.\n");

  printf("\n--- Porównanie czasów ---\n");
  printf("Czas wykonania (2 strumienie): %f ms\n", time_2_streams);
  printf("Czas wykonania (1 strumień):   %f ms\n", time_1_stream);
  printf("---------------------------\n");

  cudaEventDestroy(start_2_streams);
  cudaEventDestroy(stop_2_streams);
  cudaEventDestroy(start_1_stream);
  cudaEventDestroy(stop_1_stream);
  for (int i = 0; i < num_streams; ++i)
    cudaStreamDestroy(stream[i]);

  cudaFree(outputDevPtr);
  cudaFree(inputDevPtr);
  cudaFreeHost(hostPtr);

  return 0;
}
