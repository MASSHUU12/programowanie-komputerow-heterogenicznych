#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define N 10000000

typedef struct WIERZCHOLEK {
  float x, y;
} Wierzcholek;

Wierzcholek Figura[N];
Wierzcholek Figura2[N];

void checkCuda(CUresult result, const char *msg) {
    if (result != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(result, &errStr);
        fprintf(stderr, "Error %s: %s (%d)\n", msg, errStr, result);
        exit(1);
    }
}

int get_max_threads_per_block(CUdevice dev) {
    int max_threads;
    checkCuda(cuDeviceGetAttribute(&max_threads,
                                   CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                   dev),
              "Checking MAX_THREADS_PER_BLOCK");
    return max_threads;
}

int main(void) {
  float alfa = 0.12345;
  int devCount;
  struct timespec start, end;
  double seconds = 0, gpu_copy_to = 0, gpu_copy_from = 0, gpu_wo_copy = 0;

  srand(time(NULL));

  printf("N = %d\n", N);

  for (int i = 0; i < N; ++i) {
    Figura[i].x = (float)(rand() % 100);
    Figura[i].y = (float)(rand() % 100);
    Figura2[i] = Figura[i];
  }

  puts("--- CPU --- ");
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < N; i++) {
    float x = Figura[i].x * cos(alfa) - Figura[i].y * sin(alfa);
    float y = Figura[i].x * sin(alfa) + Figura[i].y * cos(alfa);
    Figura[i].x = x;
    Figura[i].y = y;
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  seconds = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("CPU: %fs\n", seconds);

  CUdevice cuDevice;
  CUcontext cuContext;
  CUmodule cuModule;
  CUfunction cuFunction;
  CUdeviceptr d_W;

  checkCuda(cuInit(0), "cuInit");
  checkCuda(cuDeviceGetCount(&devCount), "cuDeviceGetCount");

  if (devCount == 0) {
      perror("Nie ściemniaj – nie masz CUDY");
      return 1;
  }

  checkCuda(cuDeviceGet(&cuDevice, 0), "cuDeviceGet");
  checkCuda(cuCtxCreate(&cuContext, NULL, 0, cuDevice), "cuCtxCreate");

  int max_threads = get_max_threads_per_block(cuDevice);
  printf("Maksymalna liczba wątków na blok: %d\n", max_threads);

  checkCuda(cuModuleLoad(&cuModule, "main.ptx"), "cuModuleLoad");
  checkCuda(cuModuleGetFunction(&cuFunction, cuModule, "obracanie"), "cuModuleGetFunction");

  puts("--- GPU ---");

  clock_gettime(CLOCK_MONOTONIC, &start);
  checkCuda(cuMemAlloc(&d_W, N * sizeof(Wierzcholek)), "cuMemAlloc");

  checkCuda(cuMemcpyHtoD(d_W, Figura2, N * sizeof(Wierzcholek)), "cuMemcpyHtoD");

  clock_gettime(CLOCK_MONOTONIC, &end);
  gpu_copy_to = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  int watki_na_blok = (max_threads > 1024) ? 1024 : max_threads;
  int bloki_na_siatke = (N + watki_na_blok - 1) / watki_na_blok;

  void *args[] = { &d_W, &alfa };

  clock_gettime(CLOCK_MONOTONIC, &start);
  checkCuda(cuLaunchKernel(cuFunction,
                           bloki_na_siatke, 1, 1,
                           watki_na_blok, 1, 1,
                           0,
                           0,
                           args,
                           0), "cuLaunchKernel");

  checkCuda(cuCtxSynchronize(), "cuCtxSynchronize");

  clock_gettime(CLOCK_MONOTONIC, &end);
  gpu_wo_copy = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  clock_gettime(CLOCK_MONOTONIC, &start);
  checkCuda(cuMemcpyDtoH(Figura2, d_W, N * sizeof(Wierzcholek)), "cuMemcpyDtoH");
  clock_gettime(CLOCK_MONOTONIC, &end);
  gpu_copy_from = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  checkCuda(cuMemFree(d_W), "cuMemFree");
  checkCuda(cuCtxDestroy(cuContext), "cuCtxDestroy");

  printf("GPU w/o copy: %fs\n", gpu_wo_copy);
  printf("GPU copy to device: %fs\n", gpu_copy_to);
  printf("GPU copy to host: %fs\n", gpu_copy_from);
  printf("GPU cost: %f%%\n", (gpu_copy_to + gpu_copy_from) /
                                 (gpu_wo_copy + gpu_copy_to + gpu_copy_from) *
                                 100);

  int diffs = 0;
  float epsilon = 1e-4;

  for (int i = 0; i < N; ++i) {
    if (fabs(Figura[i].x - Figura2[i].x) > epsilon || fabs(Figura[i].y - Figura2[i].y) > epsilon) {
      diffs += 1;
      if (diffs < 5) {
          printf("Diff at %d: CPU(%f, %f) vs GPU(%f, %f)\n", i, Figura[i].x, Figura[i].y, Figura2[i].x, Figura2[i].y);
      }
    }
  }

  printf("\nDiffs: %d\n", diffs);

  return 0;
}
