#ifdef NUMA_API_ENABLED
#include <stdio.h>
#include <stdlib.h>
#include <numa.h>
#include <unistd.h>
#include "neural_network.h"

extern struct NeuralNet* __real_newNet(int n_layers, int n_neurons_per_layer[]);
extern void __real_free_NN(struct NeuralNet* nn);
extern void __real_init_nn(struct NeuralNet* nn);
extern double* __real_model_train(struct NeuralNet* nn, double** X, double** y, double* y_t, 
                                 char* act, char* loss, char* opt, double lr, int samples, int ep);
extern double* __real_model_test(struct NeuralNet* nn, double** X, double** y, 
                                double* y_t, char* act, char* loss);
extern void __real_forward_propagation(struct NeuralNet* nn, char* activation_fun, char* loss);

// Intercept model allocation -> steer to 2MB aligned NUMA shared allocation
struct NeuralNet* __wrap_newNet(int n_layers, int n_neurons_per_layer[]) {
    printf("[NUMA-WRAP] Redirecting newNet -> setup_numa_model\n");
    return setup_numa_model(n_layers, n_neurons_per_layer);
}

// Initialization is handled per-slab in setup_numa_model
void __wrap_init_nn(struct NeuralNet* nn) {
    (void)nn;
    printf("[NUMA-WRAP] init_nn bypassed (handled in setup_numa_model)\n");
}

void __wrap_free_NN(struct NeuralNet* nn) {
    (void)nn;
    printf("[NUMA-WRAP] Redirecting free_NN -> unmap_nn\n");
    unmap_nn();
}

// Intercept sequential training -> launch parallel NUMA training
double* __wrap_model_train(struct NeuralNet* nn, double** X, double** y, double* y_t, 
                           char* act, char* loss, char* opt, double lr, int samples, int ep) {
    int batch_size = 1;
    int nproc = 1;

    char* env_batch = getenv("BATCH_SIZE");
    if (env_batch != NULL) {
        int b = atoi(env_batch);
        if (b > 0) batch_size = b;
    }

    char* env_workers = getenv("WORKERS");
    if (env_workers != NULL) {
        int w = atoi(env_workers);
        if (w > 0) nproc = w;
    }

    printf("[NUMA-WRAP] Intercepted model_train -> Launching parallel_training "
           "(Workers: %d, Batch Size: %d, Samples: %d, Epoch: %d)\n", 
           nproc, batch_size, samples, ep);

    return parallel_training(nn, X, y, y_t, act, loss, opt, lr, samples, ep, batch_size, nproc);
}

// Direct forward propagation wrapper
void __wrap_forward_propagation(struct NeuralNet* nn, char* activation_fun, char* loss) {
    __real_forward_propagation(nn, activation_fun, loss);
}

/**
 * Intercepts model_test:
 * Allocates temporary process-private evaluation buffers for the Master process
 * and evaluates directly against the canonical Slab 0 weights.
 */
double* __wrap_model_test(struct NeuralNet* nn, double** X, double** y, 
                          double* y_t, char* act, char* loss) {
    init_process_working_buffers(nn);
    double* results = __real_model_test(nn, X, y, y_t, act, loss);
    free_process_working_buffers(nn);
    return results;
}

#endif