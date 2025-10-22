#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <stdlib.h>

//#define N	50000000
#define N 10000000

typedef struct WIERZCHOLEK {
	float x,y;
} Wierzcholek;

Wierzcholek Figura[N];
Wierzcholek Figura2[N];

__global__ void obracanie(WIERZCHOLEK *W, float alfa) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < N) {
		float x = W[i].x * cos(alfa) - W[i].y * sin(alfa);
		float y = W[i].x * sin(alfa) + W[i].y * cos(alfa);
		W[i].x = x;
		W[i].y = y;
	}
}

int main(void) {
	float 		alfa = 0.12345;
	int		ile_cudow;
	WIERZCHOLEK    *d_W;
	struct timespec start, end;
	double seconds = 0, gpu_copy_to = 0, gpu_copy_from = 0, gpu_wo_copy = 0;

	srand(time(NULL));

	printf("N = %d\n", N);

	for (int i = 0; i < N; ++i) {
		Figura[i].x = (float) (rand() % 100);
		Figura[i].y = (float) (rand() % 100);
	}

	puts("--- CPU --- ");
	clock_gettime(CLOCK_MONOTONIC, &start);
        for(int i = 0; i < N; i++) {
                float x = Figura[i].x * cos(alfa) - Figura[i].y * sin(alfa);
                float y = Figura[i].x * sin(alfa) + Figura[i].y * cos(alfa);
                Figura[i].x = x;
                Figura[i].y = y;
        }
	clock_gettime(CLOCK_MONOTONIC, &end);
	seconds = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
	printf("CPU: %fs\n", seconds);

	cudaGetDeviceCount(&ile_cudow);
	if(ile_cudow == 0) {
		perror("Nie ściemniaj – nie masz CUDY");
		return 1;
	}

	puts("--- GPU ---");
	clock_gettime(CLOCK_MONOTONIC, &start);
	cudaMalloc(&d_W, sizeof(Figura2));
	cudaMemcpy(d_W, Figura2, sizeof(Figura2), cudaMemcpyHostToDevice);
	int watki_na_blok = 1024;
  	int bloki_na_siatke = (N + watki_na_blok - 1) / watki_na_blok;
	clock_gettime(CLOCK_MONOTONIC, &end);
	gpu_copy_to = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

	clock_gettime(CLOCK_MONOTONIC, &start);
  	obracanie<<<bloki_na_siatke, watki_na_blok>>>(d_W, alfa);
	clock_gettime(CLOCK_MONOTONIC, &end);
	gpu_wo_copy = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  	cudaError_t err = cudaGetLastError();
  	assert(err == cudaSuccess);
  	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &start);
  	cudaMemcpy(Figura2, d_W, sizeof(Figura2), cudaMemcpyDeviceToHost);
	clock_gettime(CLOCK_MONOTONIC, &end);
	gpu_copy_from = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
	cudaFree(d_W);

	printf("GPU w/o copy: %fs\n", gpu_wo_copy);
	printf("GPU copy to device: %fs\n", gpu_copy_to);
	printf("GPU copy to host: %fs\n", gpu_copy_from);
	printf("GPU cost: %f%%\n", (gpu_copy_to + gpu_copy_from)/(gpu_wo_copy + gpu_copy_to + gpu_copy_from) * 100);

	// Verification
	int diffs = 0;
	for (int i = 0; i < N; ++i) {
		if (Figura[i].x != Figura2[i].x || Figura[i].y != Figura2[i].y) {
			//printf("\nDiff: i = %d", i);
			//printf("\tFigura: x = %f, y = %f", Figura[i].x, Figura[i].y);
			//printf("\tFigura2: x = %f, y = %f", Figura2[i].x, Figura2[i].y);
			diffs += 1;
		}
	}

	printf("\nDiffs: %d\n", diffs);

	return 0;
}
