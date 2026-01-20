#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <sched.h>

extern int seed; // Declare as extern so it can be accessed elsewhere

double randn(void);
void shuffle(int* arr, size_t n);

#endif