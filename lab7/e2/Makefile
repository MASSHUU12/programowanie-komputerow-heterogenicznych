CC = gcc
NVCC = nvcc
CFLAGS = -O3
LDFLAGS = -lcuda

CUDA_PATH ?= /usr/local/cuda
INCLUDES = -I$(CUDA_PATH)/include -I/usr/include

TARGET = main
PTX = main.ptx

all: $(TARGET) $(PTX)

$(PTX): main.cu
	$(NVCC) -ptx main.cu -o main.ptx

$(TARGET): main.c
	$(CC) $(CFLAGS) $(INCLUDES) main.c -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET) $(PTX)

run_add: $(TARGET) $(PTX)
	./$(TARGET) add_vectors

run_dot: $(TARGET) $(PTX)
	./$(TARGET) dot_product

run_scale: $(TARGET) $(PTX)
	./$(TARGET) scale_vectors
