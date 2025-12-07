extern "C" {

__global__ void add_vectors(float *in1, float *in2, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in1[idx] + in2[idx];
    }
}

__global__ void dot_product(float *in1, float *in2, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;
    if (idx < N) {
        temp = in1[idx] * in2[idx];
    }

    if (idx < N) {
        atomicAdd(out, temp);
    }
}

__global__ void scale_vectors(float *in, float *out, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx] * scale;
    }
}

}
