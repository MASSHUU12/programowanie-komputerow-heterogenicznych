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

  cudaStream_t stream[num_streams];
  for (int i = 0; i < num_streams; ++i)
    cudaStreamCreate(&stream[i]);

  float *hostPtr;
  cudaMallocHost((void **)&hostPtr, total_size_bytes);

  for (int i = 0; i < total_elements; ++i) {
    hostPtr[i] = (float)i;
  }

  float *inputDevPtr, *outputDevPtr;
  cudaMalloc(&inputDevPtr, total_size_bytes);
  cudaMalloc(&outputDevPtr, total_size_bytes);

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

  cudaDeviceSynchronize();

  bool bCorrect = true;
  for (int i = 0; i < total_elements; ++i) {
    float expected_value = (float)i * 2.f;
    if (hostPtr[i] != expected_value) {
      printf("Error at index %d: host value is %f, but expected is %f\n", i,
             hostPtr[i], expected_value);
      bCorrect = false;
      break;
    }
  }

  if (bCorrect) {
    printf("Verification successful! All values are correct.\n");
  } else {
    printf("Verification failed.\n");
  }

  for (int i = 0; i < num_streams; ++i)
    cudaStreamDestroy(stream[i]);

  cudaFree(outputDevPtr);
  cudaFree(inputDevPtr);
  cudaFreeHost(hostPtr);
  return 0;
}
