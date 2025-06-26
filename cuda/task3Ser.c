#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void addArrays(double *arr1, double *arr2, double *result, int size);
void subtractArrays(double *arr1, double *arr2, double *result, int size);
void multiplyArrays(double *arr1, double *arr2, double *result, int size);
void divideArrays(double *arr1, double *arr2, double *result, int size);

int main()
{
    int array_size = 0;
    double *arr1 = NULL;
    double *arr2 = NULL;
    double *result_add = NULL;
    double *result_sub = NULL;
    double *result_mul = NULL;
    double *result_div = NULL;
    FILE *fp = NULL;

    fp = fopen("array_size.txt", "r");
    if (fp == NULL)
    {
        fprintf(stderr, "Error: Could not open file array_size.txt\n");
        return 1;
    }

    if (fscanf(fp, "%d", &array_size) != 1)
    {
        fprintf(stderr, "Error: Could not read array size from file\n");
        fclose(fp);
        return 1;
    }

    fclose(fp);

    if (array_size <= 100000)
    {
        fprintf(stderr, "Error: Array size must be greater than 100000\n");
        return 1;
    }

    arr1 = (double *)malloc(array_size * sizeof(double));
    arr2 = (double *)malloc(array_size * sizeof(double));
    result_add = (double *)malloc(array_size * sizeof(double));
    result_sub = (double *)malloc(array_size * sizeof(double));
    result_mul = (double *)malloc(array_size * sizeof(double));
    result_div = (double *)malloc(array_size * sizeof(double));

    if (arr1 == NULL || arr2 == NULL || result_add == NULL || result_sub == NULL || result_mul == NULL || result_div == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed\n");
        if (arr1)
            free(arr1);
        if (arr2)
            free(arr2);
        if (result_add)
            free(result_add);
        if (result_sub)
            free(result_sub);
        if (result_mul)
            free(result_mul);
        if (result_div)
            free(result_div);

        return 1;
    }

    srand(time(NULL));
    for (int i = 0; i < array_size; i++)
    {
        arr1[i] = (double)rand() / RAND_MAX * 10.0;
        arr2[i] = (double)rand() / RAND_MAX * 10.0;
    }

    clock_t start_time, end_time;
    double elapsed_time;

    start_time = clock();
    addArrays(arr1, arr2, result_add, array_size);
    // end_time = clock();
    // elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    // printf("Time for addition: %.6f seconds\n", elapsed_time);

    // start_time = clock();
    subtractArrays(arr1, arr2, result_sub, array_size);
    // end_time = clock();
    // elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    // printf("Time for subtraction: %.6f seconds\n", elapsed_time);

    // start_time = clock();
    multiplyArrays(arr1, arr2, result_mul, array_size);
    // end_time = clock();
    // elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    // printf("Time for multiplication: %.6f seconds\n", elapsed_time);

    // start_time = clock();
    divideArrays(arr1, arr2, result_div, array_size);
    end_time = clock();
    elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    // printf("Time for division: %.6f seconds\n", elapsed_time);
    printf("All time: %.6f seconds\n", elapsed_time);

    free(arr1);
    free(arr2);
    free(result_add);
    free(result_sub);
    free(result_mul);
    free(result_div);

    return 0;
}

void addArrays(double *arr1, double *arr2, double *result, int size)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = arr1[i] + arr2[i];
    }
}

void subtractArrays(double *arr1, double *arr2, double *result, int size)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = arr1[i] - arr2[i];
    }
}

void multiplyArrays(double *arr1, double *arr2, double *result, int size)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = arr1[i] * arr2[i];
    }
}

void divideArrays(double *arr1, double *arr2, double *result, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (arr2[i] == 0.0)
        {
            result[i] = 0.0; // Handle division by zero
            fprintf(stderr, "Warning: Division by zero at index %d\n", i);
        }
        else
        {
            result[i] = arr1[i] / arr2[i];
        }
    }
}