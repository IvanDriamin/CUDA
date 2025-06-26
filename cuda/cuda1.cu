#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void addArraysKernel ( double *arr1, double *arr2, double *result, int size ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = arr1[i] + arr2[i];
    }
}

__global__ void subtractArraysKernel ( double *arr1, double *arr2, double *result, int size ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = arr1[i] - arr2[i];
    }
}

__global__ void multiplyArraysKernel ( double *arr1, double *arr2, double *result, int size ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        result[i] = arr1[i] * arr2[i];
    }
}

__global__ void divideArraysKernel ( double *arr1, double *arr2, double *result, int size ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (arr2[i] == 0.0) {
            result[i] = 0.0;
        } else {
            result[i] = arr1[i] / arr2[i];
        }
    }
}

int main() {
    int array_size = 0;
    double *arr1 = NULL;
    double *arr2 = NULL;
    double *result_add = NULL;
    double *result_sub = NULL;
    double *result_mul = NULL;
    double *result_div = NULL;
    FILE *fp = NULL;

    double *d_arr1 = NULL;
    double *d_arr2 = NULL;
    double *d_result_add = NULL;
    double *d_result_sub = NULL;
    double *d_result_mul = NULL;
    double *d_result_div = NULL;

    fp = fopen("array_size.txt", "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open file array_size.txt\n");
        return 1;
    }

    if (fscanf(fp, "%d", &array_size) != 1) {
        fprintf(stderr, "Error: Could not read array size from file\n");
        fclose(fp);
        return 1;
    }

    fclose(fp);

    if (array_size <= 10) {
        fprintf(stderr, "Error: Array size must be greater than 100000\n");
        return 1;
    }

    arr1 = (double *)malloc(array_size * sizeof(double));
    arr2 = (double *)malloc(array_size * sizeof(double));
    result_add = (double *)malloc(array_size * sizeof(double));
    result_sub = (double *)malloc(array_size * sizeof(double));
    result_mul = (double *)malloc(array_size * sizeof(double));
    result_div = (double *)malloc(array_size * sizeof(double));

    if (arr1 == NULL || arr2 == NULL || result_add == NULL || result_sub == NULL || result_mul == NULL || result_div == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        if (arr1) free(arr1);
        if (arr2) free(arr2);
        if (result_add) free(result_add);
        if (result_sub) free(result_sub);
        if (result_mul) free(result_mul);
        if (result_div) free(result_div);
        return 1;
    }

    srand(time(NULL));
    for (int i = 0; i < array_size; i++) {
        arr1[i] = (double)rand() / RAND_MAX * 10.0;
        arr2[i] = (double)rand() / RAND_MAX * 10.0;
    }

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void**)&d_arr1, array_size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return 1;
    }
    cudaStatus = cudaMalloc((void**)&d_arr2, array_size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
         return 1;
    }
    cudaStatus = cudaMalloc((void**)&d_result_add, array_size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return 1;
    }
    cudaStatus = cudaMalloc((void**)&d_result_sub, array_size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return 1;
    }
    cudaStatus = cudaMalloc((void**)&d_result_mul, array_size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return 1;
    }
    cudaStatus = cudaMalloc((void**)&d_result_div, array_size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return 1;
    }

    cudaStatus = cudaMemcpy(d_arr1, arr1, array_size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return 1;
    }
    cudaStatus = cudaMemcpy(d_arr2, arr2, array_size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return 1;
    }

    int blockSize = 256;
    int numBlocks = (array_size + blockSize - 1) / blockSize;

    clock_t start_time, end_time;
    double elapsed_time;

    start_time = clock();

    addArraysKernel<<<numBlocks, blockSize>>>(d_arr1, d_arr2, d_result_add, array_size);
    subtractArraysKernel<<<numBlocks, blockSize>>>(d_arr1, d_arr2, d_result_sub, array_size);
    multiplyArraysKernel<<<numBlocks, blockSize>>>(d_arr1, d_arr2, d_result_mul, array_size);
    divideArraysKernel<<<numBlocks, blockSize>>>(d_arr1, d_arr2, d_result_div, array_size);

    cudaDeviceSynchronize();

    end_time = clock();
    elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("All time: %.6f seconds\n", elapsed_time);

    FILE* f = fopen("script3.txt", "a");
    if (f == NULL) {
        fprintf(stderr, "Error: Cannot open output file\n");
        return 1;
    }
    fprintf(f, "%lf\n", elapsed_time);
    fflush(f);
    fclose(f);

    cudaStatus = cudaMemcpy(result_add, d_result_add, array_size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return 1;
    }
    cudaStatus = cudaMemcpy(result_sub, d_result_sub, array_size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return 1;
    }
    cudaStatus = cudaMemcpy(result_mul, d_result_mul, array_size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return 1;
    }
    cudaStatus = cudaMemcpy(result_div, d_result_div, array_size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return 1;
    }

    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_result_add);
    cudaFree(d_result_sub);
    cudaFree(d_result_mul);
    cudaFree(d_result_div);

    free(arr1);
    free(arr2);
    free(result_add);
    free(result_sub);
    free(result_mul);
    free(result_div);

    return 0;
}
