__global__ void MyKernel(float *out, float *in, int size) {
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix < size)
    out[ix] = in[ix] * 2.f;
}

int main(void) {
  const int size = 8192 * sizeof(float);

  cudaStream_t stream[2];
  for (int i = 0; i < 2; ++i)
    cudaStreamCreate(&stream[i]);

  float *hostPtr;
  cudaMallocHost((void **)&hostPtr, 2 * size);

  float *inputDevPtr, *outputDevPtr;
  cudaMalloc(&inputDevPtr, 2 * size);
  cudaMalloc(&outputDevPtr, 2 * size);

  for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(inputDevPtr + i * size, hostPtr + i * size, size,
                    cudaMemcpyHostToDevice, stream[i]);
  for (int i = 0; i < 2; ++i)
    MyKernel<<<100, 512, 0, stream[i]>>>(outputDevPtr + i * size,
                                         inputDevPtr + i * size, size);
  for (int i = 0; i < 2; ++i)
    cudaMemcpyAsync(hostPtr + i * size, outputDevPtr + i * size, size,
                    cudaMemcpyDeviceToHost, stream[i]);
  cudaDeviceSynchronize();
  for (int i = 0; i < 2; ++i)
    cudaStreamDestroy(stream[i]);

  cudaFree(outputDevPtr);
  cudaFree(inputDevPtr);
  cudaFreeHost(hostPtr);
  return 0;
}
