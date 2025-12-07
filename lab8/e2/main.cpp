#include <cuda.h>
#include <iostream>
#include <vector>
#include <stdio.h>

#define CHECK_CUDA(call) \
    do { \
        CUresult res = call; \
        if (res != CUDA_SUCCESS) { \
            const char* errStr; \
            cuGetErrorString(res, &errStr); \
            std::cerr << "CUDA Error: " << errStr << " at line " << __LINE__ << std::endl; \
            return 1; \
        } \
    } while(0)

int main() {
    CHECK_CUDA(cuInit(0));

    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));

    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, NULL, 0, device));

    CUmodule module;
    CHECK_CUDA(cuModuleLoad(&module, "polynomial.ptx"));

    CUfunction kernel;
    CHECK_CUDA(cuModuleGetFunction(&kernel, module, "polynomial_kernel"));

    int n = 3;
    float h_x = 2.0f;
    std::vector<float> h_a = {5.0f, 1.0f, 2.0f, 3.0f};

    CUdeviceptr d_x, d_a;
    CHECK_CUDA(cuMemAlloc(&d_x, sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_a, (n + 1) * sizeof(float)));

    CHECK_CUDA(cuMemcpyHtoD(d_x, &h_x, sizeof(float)));
    CHECK_CUDA(cuMemcpyHtoD(d_a, h_a.data(), (n + 1) * sizeof(float)));

    void* args[] = { &d_x, &d_a, &n };

    CHECK_CUDA(cuLaunchKernel(kernel,
                              1, 1, 1,
                              1, 1, 1,
                              0,
                              0,
                              args,
                              0
    ));

    CHECK_CUDA(cuCtxSynchronize());

    float result;
    CHECK_CUDA(cuMemcpyDtoH(&result, d_x, sizeof(float)));

    std::cout << "Dla x = " << h_x << ", wynik wielomianu to: " << result << std::endl;
    std::cout << "Oczekiwano: 39" << std::endl;

    cuMemFree(d_x);
    cuMemFree(d_a);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}
