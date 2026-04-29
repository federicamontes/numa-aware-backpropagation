#ifdef NUMA_API_ENABLED
#include <stdio.h>

#include <numa.h>
#include "neural_network.h"
#include <unistd.h> // Per getpid()

// Dichiarazione delle funzioni reali (servono per il fallback)
extern struct NeuralNet* __real_newNet(int n_layers, int n_neurons_per_layer[]);
extern void __real_free_NN(NeuralNet* nn);
extern void __real_init_nn(struct NeuralNet* nn);
extern double* __real_model_train(struct NeuralNet* nn, double** X, double** y, double* y_t, 
                                      char* act, char* loss, char* opt, double lr, int samples, int ep);

// Dichiarazione della funzione reale che usa la struct "Standard"
extern double* __real_model_test(struct NeuralNet* nn, double** X, double** y, 
                                 double* y_t, char* act, char* loss);

extern void __real_forward_propagation(struct NeuralNet* nn, char* activation_fun, char* loss);

extern unsigned long base_address;

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
    return unmap_nn();
}

// Wrapper for training: parallel version
double* __wrap_model_train(struct NeuralNet* nn, double** X, double** y, double* y_t, 
                           char* act, char* loss, char* opt, double lr, int samples, int ep) {
    
    // TODO change these parameters in some way
    int batch = 256; 
    int nproc = 1; // 0 = usa tutti i core disponibili

    printf("[NUMA-AUTO] Intercepted model_train -> Executing parallel_training\n");
    return parallel_training(nn, X, y, y_t, act, loss, opt, lr, samples, ep, batch, nproc);
}


// Wrapper che intercetta la chiamata dal main
double* __wrap_model_test(struct NeuralNet* nn_numa, double** X, double** y, 
                                 double* y_t, char* act, char* loss) {
    
    printf("[NUMA-AUTO] Intercepted model_test -> Mapping NUMA struct to Standard layout\n");

    // CREIAMO UNA STRUCT "FINTA" (Layout Standard) sullo stack
    // Nota: Questa struct deve corrispondere alla versione "else" dell'header
    struct NeuralNet standard_nn;
    
    // Copiamo i campi comuni manualmente. 
    // Qui 'standard_nn' userà l'offset della versione non-NUMA.
    standard_nn.magic_test_value = nn_numa->magic_test_value;
    standard_nn.n_layers = nn_numa->n_layers;
    standard_nn.n_neurons_per_layer = nn_numa->n_neurons_per_layer;
    standard_nn.w = nn_numa->w;
    standard_nn.b = nn_numa->b;
    standard_nn.in = nn_numa->in;
    standard_nn.out = nn_numa->out;   // <--- Qui risolviamo il problema dell'offset!
    standard_nn.targets = nn_numa->targets;
    

    // Chiamiamo la funzione reale passandole l'indirizzo della struct standard
    return __real_model_test(&standard_nn, X, y, y_t, act, loss);
}

// Wrapper per forward_propagation
void __wrap_forward_propagation(struct NeuralNet* nn_or_ws, char* activation_fun, char* loss) {

    // In questo contesto, nn_or_ws è SEMPRE un NNWorkerWorkspace*
    NNWorkerWorkspace* ws = (NNWorkerWorkspace*)nn_or_ws;

    // Creiamo l'adapter locale per parlare con la funzione sequenziale
    struct NeuralNet adapter_nn;
    
    // Mappiamo i dati: Pesi globali + Buffer privati del worker
    adapter_nn.n_layers = ws->shared_nn->n_layers;
    adapter_nn.n_neurons_per_layer = ws->shared_nn->n_neurons_per_layer;
    adapter_nn.w = ws->shared_nn->w;
    adapter_nn.b = ws->shared_nn->b;
    
    adapter_nn.in = ws->in;
    adapter_nn.out = ws->out;
    adapter_nn.targets = ws->targets;

    // Eseguiamo la funzione reale
    __real_forward_propagation(&adapter_nn, activation_fun, loss);
}



#endif