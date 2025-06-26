#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void addMatrices(double **matrix1, double **matrix2, double **result, int rows, int cols);
void subtractMatrices(double **matrix1, double **matrix2, double **result, int rows, int cols);
void multiplyMatrices(double **matrix1, double **matrix2, double **result, int rows, int cols);
void divideMatrices(double **matrix1, double **matrix2, double **result, int rows, int cols);
double **allocateMatrix(int rows, int cols);
void freeMatrix(double **matrix, int rows);

int main()
{
    int rows = 0, cols = 0;
    double **matrix1 = NULL, **matrix2 = NULL, **result_add = NULL, **result_sub = NULL, **result_mul = NULL, **result_div = NULL;
    FILE *fp = NULL;
    char filename[] = "array_size.txt";

    fp = fopen(filename, "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return 1;
    }
    if (fscanf(fp, "%d %d", &rows, &cols) != 2)
    {
        fprintf(stderr, "Error: Could not read rows and columns from %s\n", filename);
        fclose(fp);
        return 1;
    }
    fclose(fp);

    if (rows <= 0 || cols <= 0)
    {
        fprintf(stderr, "Error: Matrix dimensions must be positive\n");
        return 1;
    }

    // 2. Allocate memory for matrices
    matrix1 = allocateMatrix(rows, cols);
    matrix2 = allocateMatrix(rows, cols);
    result_add = allocateMatrix(rows, cols);
    result_sub = allocateMatrix(rows, cols);
    result_mul = allocateMatrix(rows, cols);
    result_div = allocateMatrix(rows, cols);

    if (matrix1 == NULL || matrix2 == NULL || result_add == NULL || result_sub == NULL || result_mul == NULL || result_div == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed\n");
        if (matrix1)
            freeMatrix(matrix1, rows);
        if (matrix2)
            freeMatrix(matrix2, rows);
        if (result_add)
            freeMatrix(result_add, rows);
        if (result_sub)
            freeMatrix(result_sub, rows);
        if (result_mul)
            freeMatrix(result_mul, rows);
        if (result_div)
            freeMatrix(result_div, rows);
        return 1;
    }

    srand(time(NULL));
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix1[i][j] = (double)rand() / RAND_MAX * 10.0;
            matrix2[i][j] = (double)rand() / RAND_MAX * 10.0;
        }
    }

    clock_t start_time, end_time;
    double elapsed_time;

    start_time = clock();
    addMatrices(matrix1, matrix2, result_add, rows, cols);
    // end_time = clock();
    // elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    // printf("Time for addition: %.6f seconds\n", elapsed_time);

    // start_time = clock();
    subtractMatrices(matrix1, matrix2, result_sub, rows, cols);
    // end_time = clock();
    // elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    // printf("Time for subtraction: %.6f seconds\n", elapsed_time);

    // start_time = clock();
    multiplyMatrices(matrix1, matrix2, result_mul, rows, cols);
    // end_time = clock();
    // elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    // printf("Time for multiplication: %.6f seconds\n", elapsed_time);

    // start_time = clock();
    divideMatrices(matrix1, matrix2, result_div, rows, cols);
    end_time = clock();
    elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    // printf("Time for division: %.6f seconds\n", elapsed_time);
    printf("All time: %.6f seconds\n", elapsed_time);

    freeMatrix(matrix1, rows);
    freeMatrix(matrix2, rows);
    freeMatrix(result_add, rows);
    freeMatrix(result_sub, rows);
    freeMatrix(result_mul, rows);
    freeMatrix(result_div, rows);

    return 0;
}

void addMatrices(double **matrix1, double **matrix2, double **result, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}

void subtractMatrices(double **matrix1, double **matrix2, double **result, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }
}

void multiplyMatrices(double **matrix1, double **matrix2, double **result, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[i][j] = matrix1[i][j] * matrix2[i][j];
        }
    }
}

void divideMatrices(double **matrix1, double **matrix2, double **result, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (matrix2[i][j] == 0.0)
            {
                result[i][j] = 0.0;
                fprintf(stderr, "Warning: Division by zero at [%d][%d]\n", i, j);
            }
            else
            {
                result[i][j] = matrix1[i][j] / matrix2[i][j];
            }
        }
    }
}

double **allocateMatrix(int rows, int cols)
{
    double **matrix = (double **)malloc(rows * sizeof(double *));
    if (matrix == NULL)
    {
        fprintf(stderr, "Error: Could not allocate memory for rows\n");
        return NULL;
    }
    for (int i = 0; i < rows; i++)
    {
        matrix[i] = (double *)malloc(cols * sizeof(double));
        if (matrix[i] == NULL)
        {
            fprintf(stderr, "Error: Could not allocate memory for columns in row %d\n", i);
            for (int j = 0; j < i; j++)
            {
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

void freeMatrix(double **matrix, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}