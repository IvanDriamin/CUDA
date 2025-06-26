#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define ASCENDING 1
#define DESCENDING 0

__global__ void bitonicSortStep(int* dev_values, int j, int k, int size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    unsigned int ixj = i ^ j;

    if (ixj > i && ixj < size) {
        int dir = ((i & k) == 0) ? ASCENDING : DESCENDING;

        if ((dir == ASCENDING && dev_values[i] > dev_values[ixj]) ||
            (dir == DESCENDING && dev_values[i] < dev_values[ixj])) {
            int temp = dev_values[i];
            dev_values[i] = dev_values[ixj];
            dev_values[ixj] = temp;
        }
    }
}

int isPowerOfTwo(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    if (!isPowerOfTwo(size)) {
        fprintf(stderr, "Error: Array size must be a power of two.\n");
        return 1;
    }

    // Выделение и инициализация
    int* h_data = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        h_data[i] = rand() % 1000000;
    }

    int* d_data;
    cudaMalloc(&d_data, size * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    clock_t start = clock();

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortStep << <blocks, threadsPerBlock >> > (d_data, j, k, size);
            cudaDeviceSynchronize();
        }
    }

    clock_t end = clock();

    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Проверка результата
    bool sorted = true;
    for (int i = 1; i < size; i++) {
        if (h_data[i - 1] > h_data[i]) {
            sorted = false;
            break;
        }
    }

    // Запись времени
    FILE* f = fopen("cuda_bitonic_sort_time.txt", "a");
    if (f != NULL) {
        if (sorted != true)
            fprintf(f, "Sort error");
        fprintf(f, "%lf\n", ((double)(end - start)) / CLOCKS_PER_SEC);
        fclose(f);
    }

    free(h_data);
    return 0;
}
