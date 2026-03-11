#ifdef NUMA_API_ENABLED
#include <stdio.h>

#include <numa.h>
#include "neural_network.h"

// Dichiarazione delle funzioni reali (servono per il fallback)
extern struct NeuralNet* __real_newNet(int n_layers, int n_neurons_per_layer[]);
extern void __real_free_NN(NeuralNet* nn);
extern void __real_init_nn(struct NeuralNet* nn);
extern double* __real_model_train(struct NeuralNet* nn, double** X, double** y, double* y_t, 
                                      char* act, char* loss, char* opt, double lr, int samples, int ep);


// Wrapper for allocation
struct NeuralNet* __wrap_newNet(int n_layers, int n_neurons_per_layer[]) {
    printf("[NUMA-AUTO] Intercepted newNet -> Redirecting to setup_numa_model\n");
    return setup_numa_model(n_layers, n_neurons_per_layer);
}

// Wrapper for neural network initialization: do nothing, init already done in setup_numa_model
void __wrap_init_nn(struct NeuralNet* nn) {
    printf("[NUMA-AUTO] Intercepted init_nn -> Bypassing (already handled)\n");
    // (Vuoto)
}

void __wrap_free_NN(struct NeuralNet* nn) {
    printf("[NUMA-AUTO] Intercepted free\n");
    return unmap_nn(nn);
}

// Wrapper for training: parallel version
double* __wrap_model_train(struct NeuralNet* nn, double** X, double** y, double* y_t, 
                           char* act, char* loss, char* opt, double lr, int samples, int ep) {
    
    // TODO change these parameters in some way
    int batch = 256; 
    int nproc = 0; // 0 = usa tutti i core disponibili

    printf("[NUMA-AUTO] Intercepted model_train -> Executing parallel_training\n");
    return parallel_training(nn, X, y, y_t, act, loss, opt, lr, samples, ep, batch, nproc);
}
#endif