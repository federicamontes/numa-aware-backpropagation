#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define N_SAMPLES 60000
#define N_TEST_SAMPLES 10000
#define N_DIMS 784
#define N_CLASSES 10

#define PAGE_SIZE 4096
#define PAGE_ALIGNMENT 2097152                 // 2MB HugeTLB Alignment
#define PDE_ALIGN_SIZE (2 * 1024 * 1024)       // 2MB PMD Alignment

#define CACHELINE_SIZE 64                      // L1 Cache Line size
#define PREFETCH_ALIGN_SIZE 128                // L2 hardware prefetcher alignment

#define ALIGN_SIZE(s) (((size_t)(s) + 7) & ~(size_t)7)
#define ALIGN_BLOCK(ptr) (char*)(((uintptr_t)(ptr) + 7) & ~(uintptr_t)7)

#define ALIGN_CACHELINE(sz) (((size_t)(sz) + (CACHELINE_SIZE - 1)) & ~(size_t)(CACHELINE_SIZE - 1))
#define ALIGN_CACHELINE_PTR(ptr) (char*)(((uintptr_t)(ptr) + (CACHELINE_SIZE - 1)) & ~(uintptr_t)(CACHELINE_SIZE - 1))

#define ALIGN_PAGE(sz) ({                      \
    size_t __s = (sz);                         \
    (__s + PAGE_ALIGNMENT - 1) & ~(PAGE_ALIGNMENT - 1); \
})

typedef struct NeuralNet {
    double magic_test_value;           // Verification canary for kernel PDE switcher LKM
    int n_layers;                      // Total number of layers
    int* n_neurons_per_layer;          // Neurons per layer

    /* --- Shared NUMA Parameters & Optimizer States (0x1000000000) --- */
    double*** w;                       // Weights: [layer][prev_neuron][curr_neuron]
    double** b;                        // Biases:  [layer][curr_neuron]
    double*** momentum_w;              // First-order momentum for weights
    double*** momentum2_w;             // Second-order momentum for weights
    double** momentum_b;               // First-order momentum for biases
    double** momentum2_b;              // Second-order momentum for biases

    /* --- Process-Private Working Buffers (First-Touch Local RAM) --- */
    double** in;                       // Pre-activations:  [layer][neuron]
    double** out;                      // Post-activations: [layer][neuron]
    double* targets;                   // Target vector
    double** delta;                    // Error signals:    [layer][neuron]

    /* --- Process-Private Gradient Accumulators --- */
    double*** dw;                      // Weight gradients: [layer][prev_neuron][curr_neuron]
    double** db;                       // Bias gradients:   [layer][curr_neuron]
} NeuralNet;

extern int seed;

#ifdef NUMA_API_ENABLED
extern int num_numa_nodes;
extern size_t global_aligned_slab_size;
extern size_t global_mmap_total_size;
extern void* global_mmap_start;
extern unsigned long base_address;
#endif

extern const double beta;
extern const double beta_1;
extern const double beta_2;
extern const double epsilon;

struct NeuralNet* newNet(int n_layers, int n_neurons_per_layer[]);
void init_nn(struct NeuralNet* nn);
void free_NN(struct NeuralNet* nn);
double* model_train(struct NeuralNet* nn, double** X, double** y, double* y_t, 
                    char* act, char* loss, char* opt, double lr, int samples, int ep);
double* model_test(struct NeuralNet* nn, double** X, double** y, 
                   double* y_t, char* act, char* loss);
void forward_propagation(struct NeuralNet* nn, char* activation_fun, char* loss);
void back_propagation(struct NeuralNet* nn, char* activation_fun, double learning_rate, 
                      char* loss, char* opt, int itr);
void shuffle(int *array, size_t n);
double randn(void);
void read_csv_file(double** X, double* y_temp, double** y, char* type);
void scale_data(double** X, char* type);
void normalize_data(double** X_train, double** X_test);

#ifdef NUMA_API_ENABLED
void detect_numa_topology(void);
size_t calculate_shared_slab_size(int n_layers, const int n_neurons_per_layer[]);
struct NeuralNet* setup_numa_model(int n_layers, int n_neurons_per_layer[]);
int activate_numa_view(int numa_node_id, void* base_ptr, size_t base_nn_size);
void init_process_working_buffers(struct NeuralNet* nn);
void free_process_working_buffers(struct NeuralNet* nn);
void unmap_nn(void);
void pin_process_to_node(int rank, int target_node);
void back_propagation_accumulate(struct NeuralNet* nn, char* activation_fun, char* loss);
void update_weights_batch(struct NeuralNet* nn, double lr, char* opt, int itr, int batch_size);

double* new_model_train(struct NeuralNet* nn, double** X_train, double** y_train, 
                        double* y_train_temp, char* activation_fun, char* loss, 
                        char* opt, double learning_rate, int num_samples_to_train, 
                        int itr, int batch_size);

double* parallel_training(struct NeuralNet* base_nn, double** X_train, double** y_train, 
                         double* y_train_temp, char* activation_fun, char* loss, 
                         char* opt, double learning_rate, int total_samples, 
                         int itr, int batch_size, int n_workers);
#endif

#endif // NEURAL_NETWORK_H