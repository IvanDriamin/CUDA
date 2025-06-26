#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[])
{
    int array_size = 0;
    int* array = NULL;

    if (argc != 2) {
        printf("Usage: %s <array_size\n", argv[0]);
        return 1;
    }

    array_size = atoi(argv[1]);
    if (array_size <= 10000)
    {
        fprintf(stderr, "Error: Array size must be greater than 10000\n");
        return 1;
    }


    array = (int*)malloc(array_size * sizeof(int));
    if (array == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return 1;
    }

    srand(time(NULL));
    for (int i = 0; i < array_size; i++)
    {
        array[i] = rand() % 1000;
    }

    clock_t start_time = clock();

    long long sum = 0;
    for (int i = 0; i < array_size; i++)
    {
        sum += array[i];
    }

    clock_t end_time = clock();

    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    free(array);

    FILE* f = fopen("ser_sum_time.txt", "a");
    if (f == NULL) {
        fprintf(stderr, "Error: Cannot open output file\n");
        return 1;
    }
    fprintf(f, "%lf\n", elapsed_time);
    fflush(f);
    fclose(f);

    return 0;
}
