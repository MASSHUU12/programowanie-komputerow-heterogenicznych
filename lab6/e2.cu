#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cuda_runtime.h>

#define gpuErrchk(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

__host__ __device__ void sort9(unsigned char *arr) {
  for (int i = 1; i < 9; ++i) {
    unsigned char key = arr[i];
    int j = i - 1;
    while (j >= 0 && arr[j] > key) {
      arr[j + 1] = arr[j];
      j = j - 1;
    }
    arr[j + 1] = key;
  }
}

void medianFilterCPU(const unsigned char *input, unsigned char *output,
                     int width, int height, int channels) {
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      for (int c = 0; c < channels; ++c) {
        unsigned char neighborhood[9];
        int k = 0;
        for (int j = -1; j <= 1; ++j) {
          for (int i = -1; i <= 1; ++i) {
            neighborhood[k++] =
                input[((y + j) * width + (x + i)) * channels + c];
          }
        }
        sort9(neighborhood);
        output[(y * width + x) * channels + c] = neighborhood[4];
      }
    }
  }
}

__global__ void medianFilterKernel(const unsigned char *input,
                                   unsigned char *output, int width, int height,
                                   int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
    for (int c = 0; c < channels; ++c) {
      unsigned char neighborhood[9];
      int k = 0;
      for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
          neighborhood[k++] = input[((y + j) * width + (x + i)) * channels + c];
        }
      }
      sort9(neighborhood);
      output[(y * width + x) * channels + c] =
          neighborhood[4]; // Mediana to 5. element (indeks 4)
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Uzycie: " << argv[0] << " <sciezka_do_obrazu.jpg>"
              << std::endl;
    return 1;
  }
  const char *inputFilename = argv[1];
  const char *outputFilenameCPU = "output_cpu.png";
  const char *outputFilenameGPU = "output_gpu.png";

  int width, height, channels;
  unsigned char *img = stbi_load(inputFilename, &width, &height, &channels, 0);
  if (img == nullptr) {
    std::cerr << "Blad: Nie mozna wczytac obrazu '" << inputFilename << "'."
              << std::endl;
    return 1;
  }
  if (channels < 3) {
    std::cerr << "Blad: Obraz musi byc kolorowy (co najmniej 3 kanaly)."
              << std::endl;
    stbi_image_free(img);
    return 1;
  }
  std::cout << "Wczytano obraz: " << width << "x" << height
            << ", kanaly: " << channels << std::endl;
  size_t img_size = width * height * channels * sizeof(unsigned char);

  unsigned char *out_cpu = new unsigned char[img_size];
  unsigned char *out_gpu = new unsigned char[img_size];
  std::memcpy(out_cpu, img, img_size);
  std::memcpy(out_gpu, img, img_size);

  std::cout << "\n--- Uruchamiam przetwarzanie CPU... ---" << std::endl;
  auto start_cpu = std::chrono::high_resolution_clock::now();
  medianFilterCPU(img, out_cpu, width, height, channels);
  auto stop_cpu = std::chrono::high_resolution_clock::now();
  auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(
      stop_cpu - start_cpu);
  std::cout << "Czas CPU: " << duration_cpu.count() << " ms" << std::endl;

  stbi_write_png(outputFilenameCPU, width, height, channels, out_cpu,
                 width * channels);
  std::cout << "Zapisano wynik CPU do: " << outputFilenameCPU << std::endl;

  std::cout << "\n--- Uruchamiam przetwarzanie GPU... ---" << std::endl;
  unsigned char *d_input, *d_output;

  gpuErrchk(cudaMalloc(&d_input, img_size));
  gpuErrchk(cudaMalloc(&d_output, img_size));

  gpuErrchk(cudaMemcpy(d_input, img, img_size, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_output, out_gpu, img_size, cudaMemcpyHostToDevice));

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);
  std::cout << "Konfiguracja CUDA: Rozmiar bloku (" << blockSize.x << ","
            << blockSize.y << "), Rozmiar siatki (" << gridSize.x << ","
            << gridSize.y << ")" << std::endl;

  cudaEvent_t start_gpu, stop_gpu;
  gpuErrchk(cudaEventCreate(&start_gpu));
  gpuErrchk(cudaEventCreate(&stop_gpu));

  gpuErrchk(cudaEventRecord(start_gpu));
  medianFilterKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height,
                                              channels);
  gpuErrchk(cudaEventRecord(stop_gpu));

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  float milliseconds_gpu = 0;
  gpuErrchk(cudaEventElapsedTime(&milliseconds_gpu, start_gpu, stop_gpu));
  std::cout << "Czas GPU (kernel): " << milliseconds_gpu << " ms" << std::endl;

  gpuErrchk(cudaMemcpy(out_gpu, d_output, img_size, cudaMemcpyDeviceToHost));

  stbi_write_png(outputFilenameGPU, width, height, channels, out_gpu,
                 width * channels);
  std::cout << "Zapisano wynik GPU do: " << outputFilenameGPU << std::endl;

  gpuErrchk(cudaFree(d_input));
  gpuErrchk(cudaFree(d_output));
  gpuErrchk(cudaEventDestroy(start_gpu));
  gpuErrchk(cudaEventDestroy(stop_gpu));

  std::cout << "\n--- Weryfikacja wynikow... ---" << std::endl;
  bool identical = true;
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      for (int c = 0; c < channels; ++c) {
        size_t index = (y * width + x) * channels + c;
        if (out_cpu[index] != out_gpu[index]) {
          std::cerr << "Rozbieznosc na pikselu (" << x << ", " << y
                    << "), kanal " << c << std::endl;
          identical = false;
          goto verification_end;
        }
      }
    }
  }

verification_end:
  if (identical) {
    std::cout << "Sukces: Obrazy wyjsciowe z CPU i GPU sa identyczne."
              << std::endl;
  } else {
    std::cout << "Blad: Obrazy wyjsciowe sa rozne." << std::endl;
  }

  stbi_image_free(img);
  delete[] out_cpu;
  delete[] out_gpu;

  return 0;
}
