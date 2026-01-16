#ifndef READ_DATA_H
#define READ_DATA_H


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define N_SAMPLES 60000
#define N_DIMS 784
#define N_CLASSES 10
#define N_TEST_SAMPLES 10000

// Only declarations (prototypes) here
void read_csv_file(double** X, double* y_temp, double** y, char* type);
void scale_data(double** X, char* type);
void normalize_data(double** X_train, double** X_test);

#endif
