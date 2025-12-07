nvcc -ptx main.cu -o main.ptx
nvcc main.c -o main -lcuda
./main
