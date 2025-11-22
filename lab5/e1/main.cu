#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

typedef unsigned char uchar;

uchar *load_jpeg_as_rgb(const char *filename, int *width, int *height) {
  int channels;
  uchar *data = stbi_load(filename, width, height, &channels, 3);
  if (!data) {
    fprintf(stderr, "Błąd wczytywania obrazu: %s\n", stbi_failure_reason());
    return NULL;
  }
  return data;
}

__global__ void histogram_kernel(const uchar *pixels, int width, int height,
                                 unsigned int *histogram) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pixels = width * height;

  if (idx < total_pixels) {
    uchar r = pixels[idx * 3 + 0];
    uchar g = pixels[idx * 3 + 1];
    uchar b = pixels[idx * 3 + 2];

    atomicAdd(&histogram[r], 1);
    atomicAdd(&histogram[256 + g], 1);
    atomicAdd(&histogram[512 + b], 1);
  }
}

void print_histogram(const unsigned int *histogram, int total_pixels) {
  printf("Procentowy rozkład nasycenia składowych:\n");
  const char *channels[] = {"Czerwony (R)", "Zielony (G)", "Niebieski (B)"};

  for (int c = 0; c < 3; ++c) {
    printf("\n--- Kanał: %s ---\n", channels[c]);
    for (int i = 0; i < 256; ++i) {
      float percentage = (float)histogram[c * 256 + i] / total_pixels * 100.0f;
      if (percentage > 0.0f) {
        printf("  Poziom %3d: %6.3f%%\n", i, percentage);
      }
    }
  }
}

void save_histogram_to_csv(const char *output_filename,
                           const unsigned int *histogram,
                           long long total_pixels) {
  FILE *fp = fopen(output_filename, "w");
  if (!fp) {
    fprintf(stderr, "Błąd: Nie można otworzyć pliku do zapisu: %s\n",
            output_filename);
    return;
  }

  fprintf(fp, "LVL;R;G;B\n");

  for (int i = 0; i < 256; ++i) {
    float r_perc = (float)histogram[i] / total_pixels * 100.0f;
    float g_perc = (float)histogram[256 + i] / total_pixels * 100.0f;
    float b_perc = (float)histogram[512 + i] / total_pixels * 100.0f;

    fprintf(fp, "%d;%f;%f;%f\n", i, r_perc, g_perc, b_perc);
  }

  fclose(fp);
  printf("\nWyniki histogramu zostały zapisane do pliku: %s\n",
         output_filename);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    fprintf(stderr, "Sposób użycia: %s <nazwa_pliku.jpg>\n", argv[0]);
    return 1;
  }
  const char *filename = argv[1];
  const char *output_csv_filename = "histogram_output.csv";

  int width, height;
  uchar *h_pixels = load_jpeg_as_rgb(filename, &width, &height);
  if (!h_pixels) {
    return 1;
  }
  printf("Obraz wczytany: %s, wymiary: %d x %d\n", filename, width, height);

  long long total_pixels = (long long)width * height;
  size_t image_bytes = total_pixels * 3 * sizeof(uchar);
  size_t histogram_bytes = 3 * 256 * sizeof(unsigned int);

  uchar *d_pixels = NULL;
  unsigned int *d_histogram = NULL;

  cudaError_t err;
  err = cudaMalloc((void **)&d_pixels, image_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "Błąd alokacji pamięci dla obrazu na GPU: %s\n",
            cudaGetErrorString(err));
    return 1;
  }

  err = cudaMalloc((void **)&d_histogram, histogram_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "Błąd alokacji pamięci dla histogramu na GPU: %s\n",
            cudaGetErrorString(err));
    return 1;
  }

  err = cudaMemcpy(d_pixels, h_pixels, image_bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Błąd kopiowania obrazu na GPU: %s\n",
            cudaGetErrorString(err));
    return 1;
  }

  err = cudaMemset(d_histogram, 0, histogram_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "Błąd inicjalizacji histogramu na GPU: %s\n",
            cudaGetErrorString(err));
    return 1;
  }

  int threads_per_block = 256;
  int blocks_per_grid =
      (total_pixels + threads_per_block - 1) / threads_per_block;

  printf("Uruchamianie kernela CUDA z %d blokami i %d wątkami na blok...\n",
         blocks_per_grid, threads_per_block);
  histogram_kernel<<<blocks_per_grid, threads_per_block>>>(d_pixels, width,
                                                           height, d_histogram);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Błąd kernela: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaDeviceSynchronize();

  unsigned int *h_histogram = (unsigned int *)malloc(histogram_bytes);
  err = cudaMemcpy(h_histogram, d_histogram, histogram_bytes,
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Błąd kopiowania histogramu z GPU: %s\n",
            cudaGetErrorString(err));
    return 1;
  }
  printf("Obliczenia na GPU zakończone.\n");

  // print_histogram(h_histogram, total_pixels);
  save_histogram_to_csv(output_csv_filename, h_histogram, total_pixels);

  stbi_image_free(h_pixels);
  free(h_histogram);
  cudaFree(d_pixels);
  cudaFree(d_histogram);

  return 0;
}
