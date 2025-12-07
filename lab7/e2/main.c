#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N 1000001
#define BLOCK_SIZE 256

void checkCudaError(CUresult res, const char *msg) {
    if (res != CUDA_SUCCESS) {
        const char *errStr;
        cuGetErrorString(res, &errStr);
        fprintf(stderr, "Błąd: %s - %s\n", msg, errStr);
        exit(1);
    }
}

void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)(rand() % 100) / 10.0f;
    }
}

int verify_arrays(float *ref, float *gpu, int n) {
    float epsilon = 1e-4;
    for (int i = 0; i < n; i++) {
        if (fabs(ref[i] - gpu[i]) > epsilon) {
            printf("Błąd weryfikacji na indeksie %d: CPU=%.4f, GPU=%.4f\n", i, ref[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Użycie: %s <nazwa_kernela>\n", argv[0]);
        printf("Dostępne kernele: add_vectors, dot_product, scale_vectors\n");
        return 1;
    }

    char *kernel_name = argv[1];

    checkCudaError(cuInit(0), "cuInit");

    CUdevice device;
    checkCudaError(cuDeviceGet(&device, 0), "cuDeviceGet");

    CUcontext context;
    checkCudaError(cuCtxCreate(&context, NULL, 0, device), "cuCtxCreate");

    CUmodule module;
    checkCudaError(cuModuleLoad(&module, "main.ptx"), "cuModuleLoad");

    CUfunction kernel;
    checkCudaError(cuModuleGetFunction(&kernel, module, kernel_name), "cuModuleGetFunction");

    size_t size = N * sizeof(float);
    float *h_in1 = (float *)malloc(size);
    float *h_in2 = (float *)malloc(size);
    float *h_out = (float *)malloc(size);
    float *h_ref = (float *)malloc(size);

     srand(1234);
    init_vector(h_in1, N);
    init_vector(h_in2, N);

    CUdeviceptr d_in1, d_in2, d_out;
    checkCudaError(cuMemAlloc(&d_in1, size), "cuMemAlloc d_in1");
    checkCudaError(cuMemAlloc(&d_in2, size), "cuMemAlloc d_in2");

    checkCudaError(cuMemcpyHtoD(d_in1, h_in1, size), "cuMemcpyHtoD in1");
    checkCudaError(cuMemcpyHtoD(d_in2, h_in2, size), "cuMemcpyHtoD in2");

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void *args[4];
    int n_val = N;
    float scale_val = 5.0f;

    if (strcmp(kernel_name, "add_vectors") == 0) {
        checkCudaError(cuMemAlloc(&d_out, size), "cuMemAlloc d_out");

        args[0] = &d_in1;
        args[1] = &d_in2;
        args[2] = &d_out;
        args[3] = &n_val;

        printf("Uruchamianie %s...\n", kernel_name);
        checkCudaError(cuLaunchKernel(kernel, blocks, 1, 1, BLOCK_SIZE, 1, 1, 0, 0, args, 0), "cuLaunchKernel");

        checkCudaError(cuMemcpyDtoH(h_out, d_out, size), "cuMemcpyDtoH out");

        printf("Weryfikacja wyników...\n");
        for(int i=0; i<N; i++) h_ref[i] = h_in1[i] + h_in2[i];

        if (verify_arrays(h_ref, h_out, N)) {
            printf("TEST ZALICZONY.\n");
        } else {
            printf("TEST NIEZALICZONY!\n");
        }

        cuMemFree(d_out);
    } else if (strcmp(kernel_name, "dot_product") == 0) {
        checkCudaError(cuMemAlloc(&d_out, sizeof(float)), "cuMemAlloc d_out scalar");
        float zero = 0.0f;
        checkCudaError(cuMemcpyHtoD(d_out, &zero, sizeof(float)), "Zeroing d_out");

        args[0] = &d_in1;
        args[1] = &d_in2;
        args[2] = &d_out;
        args[3] = &n_val;

        printf("Uruchamianie %s...\n", kernel_name);
        checkCudaError(cuLaunchKernel(kernel, blocks, 1, 1, BLOCK_SIZE, 1, 1, 0, 0, args, 0), "cuLaunchKernel");

        float result_gpu;
        checkCudaError(cuMemcpyDtoH(&result_gpu, d_out, sizeof(float)), "cuMemcpyDtoH scalar");

        printf("Obliczanie referencyjne na CPU...\n");
        double result_cpu = 0.0;
        for(int i=0; i<N; i++) {
            result_cpu += (double)h_in1[i] * (double)h_in2[i];
        }

        printf("GPU: %.2f\nCPU: %.2f\n", result_gpu, (float)result_cpu);

        if (fabs(result_cpu - result_gpu) / result_cpu < 0.001) {
             printf("TEST ZALICZONY.\n");
        } else {
             printf("TEST NIEZALICZONY!\n");
        }

        cuMemFree(d_out);
    } else if (strcmp(kernel_name, "scale_vectors") == 0) {
        checkCudaError(cuMemAlloc(&d_out, size), "cuMemAlloc d_out");

        args[0] = &d_in1;
        args[1] = &d_out;
        args[2] = &scale_val;
        args[3] = &n_val;

        printf("Uruchamianie %s ze skala %.2f...\n", kernel_name, scale_val);
        checkCudaError(cuLaunchKernel(kernel, blocks, 1, 1, BLOCK_SIZE, 1, 1, 0, 0, args, 0), "cuLaunchKernel");

        checkCudaError(cuMemcpyDtoH(h_out, d_out, size), "cuMemcpyDtoH");

        printf("Weryfikacja wyników...\n");
        for(int i=0; i<N; i++) h_ref[i] = h_in1[i] * scale_val;

        if (verify_arrays(h_ref, h_out, N)) {
            printf("TEST ZALICZONY.\n");
        } else {
            printf("TEST NIEZALICZONY!\n");
        }

        cuMemFree(d_out);
    } else {
        printf("Nieobsługiwany typ argumentów dla kernela: %s\n", kernel_name);
    }

    cuMemFree(d_in1);
    cuMemFree(d_in2);
    cuCtxDestroy(context);
    free(h_in1);
    free(h_in2);
    free(h_out);
    free(h_ref);

    return 0;
}
