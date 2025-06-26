#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void addMatricesKernel(double *matrix1, double *matrix2, double *result, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        result[row * cols + col] = matrix1[row * cols + col] + matrix2[row * cols + col];
    }
}

__global__ void subtractMatricesKernel(double *matrix1, double *matrix2, double *result, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        result[row * cols + col] = matrix1[row * cols + col] - matrix2[row * cols + col];
    }
}

__global__ void multiplyMatricesKernel(double *matrix1, double *matrix2, double *result, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        result[row * cols + col] = matrix1[row * cols + col] * matrix2[row * cols + col];
    }
}

__global__ void divideMatricesKernel(double *matrix1, double *matrix2, double *result, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        if (matrix2[row * cols + col] == 0.0) {
            result[row * cols + col] = 0.0;
        } else {
            result[row * cols + col] = matrix1[row * cols + col] / matrix2[row * cols + col];
        }
    }
}

double *convertMatrixTo1D(double **matrix, int rows, int cols) {
    double *oneDMatrix = (double *)malloc(rows * cols * sizeof(double));
    if (oneDMatrix == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for 1D matrix\n");
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            oneDMatrix[i * cols + j] = matrix[i][j];
        }
    }

    return oneDMatrix;
}

int main() {
    int rows = 0, cols = 0;
    double **matrix1 = NULL, **matrix2 = NULL, **result_add = NULL, **result_sub = NULL, **result_mul = NULL, **result_div = NULL;
    FILE *fp = NULL;
    char filename[] = "array_size.txt";

    double *h_matrix1 = NULL, *h_matrix2 = NULL, *h_result_add = NULL, *h_result_sub = NULL, *h_result_mul = NULL, *h_result_div = NULL;
    double *d_matrix1 = NULL, *d_matrix2 = NULL, *d_result_add = NULL, *d_result_sub = NULL, *d_result_mul = NULL, *d_result_div = NULL;

    fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return 1;
    }
    if (fscanf(fp, "%d %d", &rows, &cols) != 2) {
        fprintf(stderr, "Error: Could not read rows and columns from %s\n", filename);
        fclose(fp);
        return 1;
    }
    fclose(fp);

    if (rows <= 0 || cols <= 0) {
        fprintf(stderr, "Error: Matrix dimensions must be positive\n");
        return 1;
    }

    matrix1 = allocateMatrix(rows, cols);
    matrix2 = allocateMatrix(rows, cols);
    result_add = allocateMatrix(rows, cols);
    result_sub = allocateMatrix(rows, cols);
    result_mul = allocateMatrix(rows, cols);
    result_div = allocateMatrix(rows, cols);

    if (matrix1 == NULL || matrix2 == NULL || result_add == NULL || result_sub == NULL || result_mul == NULL || result_div == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        if (matrix1) freeMatrix(matrix1, rows);
        if (matrix2) freeMatrix(matrix2, rows);
        if (result_add) freeMatrix(result_add, rows);
        if (result_sub) freeMatrix(result_sub, rows);
        if (result_mul) freeMatrix(result_mul, rows);
        if (result_div) freeMatrix(result_div, rows);
        return 1;
    }

    srand(time(NULL));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix1[i][j] = (double)rand() / RAND_MAX * 10.0;
            matrix2[i][j] = (double)rand() / RAND_MAX * 10.0;
        }
    }

    h_matrix1 = convertMatrixTo1D(matrix1, rows, cols);
    h_matrix2 = convertMatrixTo1D(matrix2, rows, cols);
    h_result_add = convertMatrixTo1D(result_add, rows, cols);
    h_result_sub = convertMatrixTo1D(result_sub, rows, cols);
    h_result_mul = convertMatrixTo1D(result_mul, rows, cols);
    h_result_div = convertMatrixTo1D(result_div, rows, cols);

    if (!h_matrix1 || !h_matrix2 || !h_result_add || !h_result_sub || !h_result_mul || !h_result_div) {
        fprintf(stderr, "Error: Conversion to 1D array failed\n");
        return 1;
    }

    size_t matrixSize = rows * cols * sizeof(double);
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&d_matrix1, matrixSize);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!\n"); return 1;}
    cudaStatus = cudaMalloc((void**)&d_matrix2, matrixSize);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!\n"); return 1;}
    cudaStatus = cudaMalloc((void**)&d_result_add, matrixSize);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!\n"); return 1;}
    cudaStatus = cudaMalloc((void**)&d_result_sub, matrixSize);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!\n"); return 1;}
    cudaStatus = cudaMalloc((void**)&d_result_mul, matrixSize);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!\n"); return 1;}
    cudaStatus = cudaMalloc((void**)&d_result_div, matrixSize);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!\n"); return 1;}

    cudaStatus = cudaMemcpy(d_matrix1, h_matrix1, matrixSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!\n"); return 1;}
    cudaStatus = cudaMemcpy(d_matrix2, h_matrix2, matrixSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!\n"); return 1;}

    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    clock_t start_time, end_time;
    double elapsed_time;

    start_time = clock();

    addMatricesKernel<<<gridDim, blockDim>>>(d_matrix1, d_matrix2, d_result_add, rows, cols);
    subtractMatricesKernel<<<gridDim, blockDim>>>(d_matrix1, d_matrix2, d_result_sub, rows, cols);
    multiplyMatricesKernel<<<gridDim, blockDim>>>(d_matrix1, d_matrix2, d_result_mul, rows, cols);
    divideMatricesKernel<<<gridDim, blockDim>>>(d_matrix1, d_matrix2, d_result_div, rows, cols);

    cudaDeviceSynchronize();

    end_time = clock();
    elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("All time: %.6f seconds\n", elapsed_time);

    FILE* f = fopen("script4.txt", "a");
    if (f == NULL) {
        fprintf(stderr, "Error: Cannot open output file\n");
        return 1;
    }
    fprintf(f, "%lf\n", elapsed_time);
    fflush(f);
    fclose(f);

    cudaStatus = cudaMemcpy(h_result_add, d_result_add, matrixSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!\n"); return 1;}
    cudaStatus = cudaMemcpy(h_result_sub, d_result_sub, matrixSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!\n"); return 1;}
    cudaStatus = cudaMemcpy(h_result_mul, d_result_mul, matrixSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!\n"); return 1;}
    cudaStatus = cudaMemcpy(h_result_div, d_result_div, matrixSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!\n"); return 1;}

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result_add[i][j] = h_result_add[i * cols + j];
            result_sub[i][j] = h_result_sub[i * cols + j];
            result_mul[i][j] = h_result_mul[i * cols + j];
            result_div[i][j] = h_result_div[i * cols + j];
        }
    }


    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_result_add);
    cudaFree(d_result_sub);
    cudaFree(d_result_mul);
    cudaFree(d_result_div);

    free(h_matrix1);
    free(h_matrix2);
    free(h_result_add);
    free(h_result_sub);
    free(h_result_mul);

    freeMatrix(matrix1, rows);
    freeMatrix(matrix2, rows);
    freeMatrix(result_add, rows);
    freeMatrix(result_sub, rows);
    freeMatrix(result_mul, rows);
    freeMatrix(result_div, rows);

    return 0;
}

void addMatrices(double **matrix1, double **matrix2, double **result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}

void subtractMatrices(double **matrix1, double **matrix2, double **result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }
}

void multiplyMatrices(double **matrix1, double **matrix2, double **result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = matrix1[i][j] * matrix2[i][j];
        }
    }
}

void divideMatrices(double **matrix1, double **matrix2, double **result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (matrix2[i][j] == 0.0) {
                result[i][j] = 0.0;
                fprintf(stderr, "Warning: Division by zero at [%d][%d]\n", i, j);
            } else {
                result[i][j] = matrix1[i][j] / matrix2[i][j];
            }
        }
    }
}

double **allocateMatrix(int rows, int cols) {
    double **matrix = (double **)malloc(rows * sizeof(double *));
    if (matrix == NULL) {
        fprintf(stderr, "Error: Could not allocate memory for rows\n");
        return NULL;
    }
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double *)malloc(cols * sizeof(double));
        if (matrix[i] == NULL) {
            fprintf(stderr, "Error: Could not allocate memory for columns in row %d\n", i);
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

void freeMatrix(double **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}
