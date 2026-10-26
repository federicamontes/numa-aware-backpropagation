#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include "neural_network.h"

#define DEBUG if(0)

int main(int argc, char *argv[]) {

    // Deterministic random seed
    seed = 12345;
    srand(12345);

    int n_layers = 4;
    int n_neurons_per_layer[] = {784, 64, 32, 10};

    double learning_rate = 1e-4;
    double init_lr = 1e-4;
    char* activation_fun = "relu";
    char* loss = "ce";
    char* opt = "adam";
    int num_samples_to_train = 10000;
    int batch_size = 1;
    int epochs = 5;
    int n_processes = 1;
    // todo: cambia criterio di arresto, wct invece di epoche

    if (argc > 1) learning_rate = atof(argv[1]);
    if (argc > 2) init_lr = atof(argv[2]);
    if (argc > 3) activation_fun = argv[3];
    if (argc > 4) loss = argv[4];
    if (argc > 5) opt = argv[5];
    if (argc > 6) num_samples_to_train = atoi(argv[6]);
    if (argc > 7) batch_size = atoi(argv[7]);
    if (argc > 8) epochs = atoi(argv[8]);
    if (argc > 9) n_processes = atoi(argv[9]);

    char batch_str[16], workers_str[16];
    snprintf(batch_str, sizeof(batch_str), "%d", batch_size);
    snprintf(workers_str, sizeof(workers_str), "%d", n_processes);
    setenv("NUMA_BATCH_SIZE", batch_str, 1);
    setenv("NUMA_WORKERS", workers_str, 1);


    struct NeuralNet* nn = newNet(n_layers, n_neurons_per_layer);
    init_nn(nn);

    double** X_train = malloc(N_SAMPLES * sizeof(double*));
    double** y_train = malloc(N_SAMPLES * sizeof(double*));
    double* y_train_temp = malloc(N_SAMPLES * sizeof(double));
    for (int i = 0; i < N_SAMPLES; i++) {
        X_train[i] = malloc(N_DIMS * sizeof(double));
        y_train[i] = malloc(N_CLASSES * sizeof(double));
    }

    double** X_test = malloc(N_TEST_SAMPLES * sizeof(double*));
    double** y_test = malloc(N_TEST_SAMPLES * sizeof(double*));
    double* y_test_temp = malloc(N_TEST_SAMPLES * sizeof(double));
    for (int i = 0; i < N_TEST_SAMPLES; i++) {
        X_test[i] = malloc(N_DIMS * sizeof(double));
        y_test[i] = malloc(N_CLASSES * sizeof(double));
    }

    printf("Loading dataset and normalizing...\n");
    read_csv_file(X_train, y_train_temp, y_train, "train");
    scale_data(X_train, "train");
    read_csv_file(X_test, y_test_temp, y_test, "test");
    scale_data(X_test, "test");
    normalize_data(X_train, X_test);

    FILE* file = fopen("metrics_training.txt", "w");
    if (file) {
        fprintf(file, "epoch,train_loss,train_acc,test_loss,test_acc\n");
    }

    struct NeuralNet* base_nn = nn;

    for (int itr = 0; itr < epochs; itr++) {
        double* train_m = model_train(base_nn, X_train, y_train, y_train_temp, 
                                      activation_fun, loss, opt, learning_rate, 
                                      num_samples_to_train, itr + 1);

        double* test_m = model_test(base_nn, X_test, y_test, y_test_temp, activation_fun, loss);

        if (file) {
            fprintf(file, "%d,%lf,%lf,%lf,%lf\n", itr + 1, train_m[0], train_m[1], test_m[0], test_m[1]);
            fflush(file);
        }

        printf("Epoch: %02d | Train Loss: %.4f | Acc: %.4f | Test Loss: %.4f | Acc: %.4f\n", 
                itr + 1, train_m[0], train_m[1], test_m[0], test_m[1]);

        learning_rate = init_lr * exp(-0.1 * (itr + 1));
    }

    if (file) fclose(file);

    free_NN(nn);

    for (int i = 0; i < N_SAMPLES; i++) { 
        free(X_train[i]); 
        free(y_train[i]); 
    }
    free(X_train); 
    free(y_train); 
    free(y_train_temp);
    
    for (int i = 0; i < N_TEST_SAMPLES; i++) { 
        free(X_test[i]); 
        free(y_test[i]); 
    }
    free(X_test); 
    free(y_test); 
    free(y_test_temp);

    return 0;
}