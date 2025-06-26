#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Ядро для суммирования массива (1 блок, 256 потоков)
__global__ void cuda_array_sum(long long* result, int* array, int size) {
    __shared__ int shared_sum[256]; // Shared memory для промежуточных сумм

    int tid = threadIdx.x;
    int index = tid;
    int stride = blockDim.x;

    // Каждый поток суммирует свои элементы
    long long local_sum = 0;
    for (int i = index; i < size; i += stride) {
        local_sum += array[i];
    }
    shared_sum[tid] = local_sum;
    __syncthreads();

    // Редукция в shared memory (суммируем результаты потоков)
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    // Первый поток записывает результат
    if (tid == 0) {
        *result = shared_sum[0];
    }
}

int main(int argc, char* argv[]) {
    int array_size = 0;
    int* array = NULL;
    int* cuda_array = NULL;
    long long sum = 0;
    long long* cuda_sum = NULL;

    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    array_size = atoi(argv[1]);
    if (array_size <= 10000) {
        fprintf(stderr, "Error: Array size must be greater than 10000\n");
        return 1;
    }

    array = (int*)malloc(array_size * sizeof(int));
    if (array == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return 1;
    }

    srand(time(NULL));
    for (int i = 0; i < array_size; i++) {
        array[i] = rand() % 1000;
    }

    clock_t start_time = clock();

    // Выделение памяти GPU
    cudaMalloc((void**)&cuda_array, sizeof(int) * array_size);
    cudaMalloc((void**)&cuda_sum, sizeof(long long));

    // Инициализация суммы на устройстве
    long long zero = 0;
    cudaMemcpy(cuda_sum, &zero, sizeof(long long), cudaMemcpyHostToDevice);

    // Копирование данных в GPU
    cudaMemcpy(cuda_array, array, sizeof(int) * array_size, cudaMemcpyHostToDevice);

    // Выполнение вычислений
    cuda_array_sum << <1, 256 >> > (cuda_sum, cuda_array, array_size);

    // Возвращение результата
    cudaMemcpy(&sum, cuda_sum, sizeof(long long), cudaMemcpyDeviceToHost);

    // Очистка памяти GPU
    cudaFree(cuda_array);
    cudaFree(cuda_sum);

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    free(array);

    FILE* f = fopen("cuda_sum_time.txt", "a");
    if (f == NULL) {
        fprintf(stderr, "Error: Cannot open output file\n");
        return 1;
    }
    fprintf(f, "%lf\n", elapsed_time);
    fclose(f);

    return 0;
}
