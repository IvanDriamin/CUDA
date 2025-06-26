#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void merge(int arr[], int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* L = (int*)malloc(n1 * sizeof(int));
    int* R = (int*)malloc(n2 * sizeof(int));


    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    i = 0; 
    j = 0; 
    k = left; 

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

int main(int argc, char* argv[]) {
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

    int* arr = (int*)malloc(array_size * sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < array_size; i++) {
        arr[i] = rand() % 1000; // „исла от 0 до 999
    }

    clock_t start = clock();
    mergeSort(arr, 0, array_size - 1);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    free(array);

    FILE* f = fopen("ser_merge_sort_time.txt", "a");
    if (f == NULL) {
        fprintf(stderr, "Error: Cannot open output file\n");
        return 1;
    }
    fprintf(f, "%lf\n", time_taken);
    fflush(f);
    fclose(f);

    return 0;
}