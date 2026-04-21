#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stddef.h>

// Definizioni che usi nel main
#define N_SAMPLES 60000
#define N_TEST_SAMPLES 10000
#define N_DIMS 784
#define N_CLASSES 10

#ifdef NUMA_API_ENABLED

// --- Memory & Alignment Settings ---
#define PAGE_SIZE 4096
// alignment for huge pages 2MB
#define PAGE_ALIGNMENT 2097152

#define PDE_ALIGN_SIZE (2 * 1024 * 1024)      // 2MB Alignment for PDE Switching

#define ALIGN_SIZE(s) (((size_t)(s) + 7) & ~(size_t)7)
#define ALIGN_BLOCK(ptr) (char*)(((uintptr_t)(ptr) + 7) & ~7)
#define ALIGN_PAGE(sz) ({            \
    size_t __s = (sz);               \
    (__s + PAGE_ALIGNMENT - 1) & ~(PAGE_ALIGNMENT - 1); \
})


/**
 * Neural Network structure 
 *  */
typedef struct NeuralNet{
    double magic_test_value; // DEBUG sanity check flag for checking pte switcher LKM

    int n_layers; // total number of layers
    int* n_neurons_per_layer; // number of neurons in each layer
    
    /* weights and bias */
    double*** w; // weights matrices between layer and layer+1, 3D: [layer][prev_neuron][curr_neuron]
    double** b; // bias vectors between layer and layer+1, 2D: [layer][curr_neuron]
    
    double** in; // pre-activation values z, z = W·x + b, 2D: [layer][neuron]
    double** out; // post-activation values a, a = activation_func(z), 2D: [layer][neuron]

    double* targets; // Target output values for the current sample (one-hot encoded vector)

} NeuralNet;


typedef struct NNWorkerWorkspace {
    /* Reference to global network to access weights and bias */
    NeuralNet* shared_nn;

    /* temporary gradients for accumulation in parallel backpropagation */
    double*** dw;
    double** db;

    /* optimzer variables (Adam/Momentum) */
    double*** momentum_w; // first order momentum for weights (mobile avg for derivatives)
    double*** momentum2_w; // second order momentum for the weights (mobile avg for quadratic derivatives)
    double** momentum_b; // first order momentum for the bias
    double** momentum2_b; // second order momentum for the bias
    
    /* backpropagation parameters */
    double** delta; // errors computed in backprop for each neuron in each layer, 2D: [layer][neuron]
    double** in; // pre-activation values z, z = W·x + b, 2D: [layer][neuron]
    double** out; // post-activation values a, a = activation_func(z), 2D: [layer][neuron]

    double* targets; // Target output values for the current sample (one-hot encoded vector)

    /* Memory Management */
    void *private_zone_start; 
    size_t private_zone_size;

    int rank;
    int local_numa_node;
} NNWorkerWorkspace;

#else 

/**
 * Neural Network structure 
 *  */
typedef struct NeuralNet{
    double magic_test_value; // DEBUG sanity check flag for checking pte switcher LKM

    int n_layers; // total number of layers
    int* n_neurons_per_layer; // number of neurons in each layer
    
    /* weights and bias */
    double*** w; // weights matrices between layer and layer+1, 3D: [layer][prev_neuron][curr_neuron]
    double** b; // bias vectors between layer and layer+1, 2D: [layer][curr_neuron]


    
    /* backpropagation parameters */
    double** in; // pre-activation values z, z = W·x + b, 2D: [layer][neuron]
    double** out; // post-activation values a, a = activation_func(z), 2D: [layer][neuron]

    double* targets; // Target output values for the current sample (one-hot encoded vector)

    /* temporary gradients for accumulation in parallel backpropagation */
    double*** dw;
    double** db;

    /* optimzer variables (Adam/Momentum) */
    double*** momentum_w; // first order momentum for weights (mobile avg for derivatives)
    double*** momentum2_w; // second order momentum for the weights (mobile avg for quadratic derivatives)
    double** momentum_b; // first order momentum for the bias
    double** momentum2_b; // second order momentum for the bias
    
    /* backpropagation parameter*/
    double** delta; // errors computed in backprop for each neuron in each layer, 2D: [layer][neuron]

   
} NeuralNet;

#endif

#ifdef NUMA_API_ENABLED
extern int num_numa_nodes;
#endif
extern const double beta_1;
extern const double beta_2;
extern const double epsilon;

// Prototipi delle funzioni che il main e i wrapper chiamano
struct NeuralNet* newNet(int n_layers, int n_neurons_per_layer[]);
void init_nn(struct NeuralNet* nn);
void free_NN(struct NeuralNet* nn);
double* model_train(struct NeuralNet* nn, double** X, double** y, double* y_t, char* act, char* loss, char* opt, double lr, int samples, int ep);
double* model_test(struct NeuralNet* nn, double** X, double** y, double* y_t, char* act, char* loss);
void read_csv_file(double** X, double* y_temp, double** y, char* type);
void scale_data(double** X, char* type);
void normalize_data(double** X_train, double** X_test);
void shuffle(int *array, size_t n);
double randn(void);

#ifdef NUMA_API_ENABLED
size_t sum_all_mmap_allocations(int n_layers, const int n_neurons_per_layer[]);
size_t calculate_shared_model_size(int n_layers, const int n_neurons_per_layer[]);
size_t calculate_private_workspace_size(int n_layers,  const int n_neurons_per_layer[]);
void unmap_nn();
void forward_propagation(struct NNWorkerWorkspace* ws, char* activation_fun, char* loss);
struct NeuralNet* setup_numa_model(int n_layers, int n_neurons_per_layer[]);
double* parallel_training(struct NeuralNet* nn, double** X, double** y, double* y_t, char* act, char* loss, char* opt, double lr, int samples, int ep, int batch, int nproc);
#endif

#endif