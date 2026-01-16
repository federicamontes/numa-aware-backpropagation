#include "utils.h"
#include <stdlib.h>

int seed; // Actual definition

double randn() {
    int a = 1103515245;
    int m = 2147483647;
    int c = 12345;
    seed = (a * seed + c) % m;
    return (double)seed / (double)m;
}

void shuffle(int* arr, size_t n) {
    if (n > 1) {
        for (size_t i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = arr[j];
            arr[j] = arr[i];
            arr[i] = t;
        }
    }
}