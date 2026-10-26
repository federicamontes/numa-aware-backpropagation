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

extern double* parallel_training_sync_lock(struct NeuralNet* base_nn, double** X_train, double** y_train, 
                                           double* y_train_temp, char* activation_fun, char* loss, 
                                           char* opt, double learning_rate, int total_samples, 
                                           int itr, int batch_size, int n_workers);

struct NeuralNet* __wrap_newNet(int n_layers, int n_neurons_per_layer[]) {
    printf("[NUMA-WRAP] Redirecting newNet -> setup_numa_model\n");
    int batch_size = 1;
    int nproc = 1;

    char* env_b = getenv("NUMA_BATCH_SIZE");
    if (!env_b) env_b = getenv("BATCH_SIZE");
    if (env_b) batch_size = atoi(env_b);

    char* env_w = getenv("NUMA_WORKERS");
    if (!env_w) env_w = getenv("WORKERS");
    if (env_w) nproc = atoi(env_w);

    size_t raw_slab_size = calculate_shared_slab_size(n_layers, n_neurons_per_layer);
    size_t aligned_slab_size = ALIGN_PAGE(raw_slab_size);

    printf("==================================================\n");
    printf("        NUMA-Aware Engine Initialization          \n");
    printf("==================================================\n");
    printf("Architecture:          [%d] -> [%d] -> [%d] -> [%d]\n", 
           n_neurons_per_layer[0], n_neurons_per_layer[1], 
           n_neurons_per_layer[2], n_neurons_per_layer[3]);
    printf("NUMA Sockets / Slabs:  %d\n", num_numa_nodes);
    printf("Parallel Workers:      %d\n", nproc);
    printf("Mini-Batch Size:       %d\n", batch_size);
    printf("Raw Shared Footprint:  %.3f MB\n", (double)raw_slab_size / (1024.0 * 1024.0));
    printf("Aligned 2MB Slab:      %.3f MB\n", (double)aligned_slab_size / (1024.0 * 1024.0));
    printf("Total Mmap Allocation: %.3f MB\n", (double)(aligned_slab_size * num_numa_nodes) / (1024.0 * 1024.0));
    printf("==================================================\n\n");

    return setup_numa_model(n_layers, n_neurons_per_layer);
}

void __wrap_init_nn(struct NeuralNet* nn) {
    (void)nn;
    printf("[NUMA-WRAP] init_nn bypassed (handled in setup_numa_model)\n");
}

void __wrap_free_NN(struct NeuralNet* nn) {
    (void)nn;
    printf("[NUMA-WRAP] Redirecting free_NN -> unmap_nn\n");
    unmap_nn();
}

double* __wrap_model_train(struct NeuralNet* nn, double** X, double** y, double* y_t, 
                           char* act, char* loss, char* opt, double lr, int samples, int ep) {
    int batch_size = 1;
    int nproc = 1;

    // Checks both NUMA_BATCH_SIZE / NUMA_WORKERS and BATCH_SIZE / WORKERS
    char* env_batch = getenv("NUMA_BATCH_SIZE");
    if (!env_batch) env_batch = getenv("BATCH_SIZE");
    if (env_batch != NULL) {
        int b = atoi(env_batch);
        if (b > 0) batch_size = b;
    }

    char* env_workers = getenv("NUMA_WORKERS");
    if (!env_workers) env_workers = getenv("WORKERS");
    if (env_workers != NULL) {
        int w = atoi(env_workers);
        if (w > 0) nproc = w;
    }

    printf("[NUMA-WRAP] Intercepted model_train -> Launching parallel_training "
           "(Workers: %d, Batch Size: %d, Samples: %d, Epoch: %d)\n", 
           nproc, batch_size, samples, ep);
    #ifdef SYNC_LOCK
    // Synchronous Mini-Batch Training (Lock-Based Reduction)
    printf("[NUMA-WRAP] Mode: SYNCHRONOUS LOCK-BASED REDUCTION\n");
    return parallel_training_sync_lock(nn, X, y, y_t, act, loss, opt, lr, samples, ep, batch_size, nproc);
#else
    // Asynchronous / Atomic Training
    printf("[NUMA-WRAP] Mode: ASYNC/ATOMIC REDUCTION\n");
    return parallel_training(nn, X, y, y_t, act, loss, opt, lr, samples, ep, batch_size, nproc);
#endif
}

void __wrap_forward_propagation(struct NeuralNet* nn, char* activation_fun, char* loss) {
    __real_forward_propagation(nn, activation_fun, loss);
}

double* __wrap_model_test(struct NeuralNet* nn, double** X, double** y, 
                          double* y_t, char* act, char* loss) {
    init_process_working_buffers(nn);
    double* results = __real_model_test(nn, X, y, y_t, act, loss);
    free_process_working_buffers(nn);
    return results;
}

#endif