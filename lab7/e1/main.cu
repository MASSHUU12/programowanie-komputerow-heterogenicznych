#define N 10000000

typedef struct WIERZCHOLEK {
  float x, y;
} Wierzcholek;

extern "C" __global__ void obracanie(WIERZCHOLEK *W, float alfa) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N) {
    float x = W[i].x * cos(alfa) - W[i].y * sin(alfa);
    float y = W[i].x * sin(alfa) + W[i].y * cos(alfa);
    W[i].x = x;
    W[i].y = y;
  }
}
