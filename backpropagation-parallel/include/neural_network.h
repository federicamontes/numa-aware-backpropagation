#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/mman.h>
#include <numa.h>
#include <numaif.h>

// --- Optimizer Constants ---
extern const double epsilon;
extern const double beta;
extern const double beta_1;
extern const double beta_2;



// --- Global State ---
extern int num_numa_nodes;


/**
 * Neural Network structure 
 *  */
typedef struct NeuralNet{
    int n_layers; // total number of layers
    int* n_neurons_per_layer; // number of neurons in each layer
    
    /* weights and bias */
    double*** w; // weights between each neurons in each pair of layers, 3D: [layer][prev_neuron][curr_neuron]
    double** b; // bias weights from bias unit to neurons in the next layer, 2D: [layer][curr_neuron]
    
    /* optimzer variables (Adam/Momentum) */
    double*** momentum_w; // first order moment for the weights (mobile avg for derivatives)
    double*** momentum2_w; // second order moment for the weights (mobile avg for quadratic derivatives)
    double** momentum_b; // first order moment for the bias
    double** momentum2_b; // second order moment for the bias
    
    /* backpropagation parameters */
    double** delta; // errors computed for each neuron in each layer, 2D: [layer][neuron]
    double** in; // input to the activation function in each layer, 2D: [layer][neuron]
    double** out; // output of the activation function in each layer, 2D: [layer][neuron]

    double* targets; // Target values for the current sample (one-hot encoded vector)

    /* Memory Management */
    void* initial_mmap_addr; //start address of mmap allocation
    size_t total_mmap_size; // total mapped region

    int numa_node_id; // ID of the numa node
} NeuralNet;

/**
 * Worker Context: One per process
 * This tracks the private workspace for a specific process
 */
typedef struct {
    int process_id;
    int start_sample;
    int end_sample;
    int chunk; // number of samples to train

    // Memory info
    char* my_zone_start; // private zone start      
    NeuralNet* shared_nn; // neural network base address

    /* PRIVATE Workspace: Each process needs its own buffers 
       to compute forward/backward passes for its specific batch */
    double** private_in;    // Private activation inputs
    double** private_out;   // Private activation outputs
    double** private_delta; // Private local errors
    double* private_targets; // Private target buffer

    /* GRADIENT STORAGE: 
       Each process computes its own gradients before 
       contributing to the shared weights update */
    double*** local_grad_w;
    double** local_grad_b;

} NNWorker;


// --- Core Math Prototypes ---

/**
 * Performs a forward pass through the network.
 * Computes 'in' (weighted sums) and 'out' (activations) for all layers.
 */
void forward_propagation(NeuralNet* nn, char* activation_fun, char* loss);

/**
 * Calculates the total loss for the current sample.
 * Supports "mse" (Mean Squared Error) and "ce" (Cross-Entropy).
 */
double calc_loss(NeuralNet* nn, char* loss);

/**
 * Performs backpropagation to compute gradients and update weights/biases.
 * Supports optimizers: "sgd", "momentum", "rmsprop", and "adam".
 */
void back_propagation(NeuralNet* nn, char* activation_fun, double learning_rate, char* loss, char* opt, int itr);

/**
 * Trains the model for one epoch.
 * Orchestrates shuffling, forward pass, backpropagation, and weight updates.
 * @return Pointer to static array [average_loss, accuracy].
 */
double* model_train(NeuralNet* nn, double** X_train, double** y_train, double* y_train_temp, 
                    char* activation_fun, char* loss, char* opt, double learning_rate,
                    int num_samples_to_train, int itr);

/**
 * Evaluates the model on a test dataset.
 * Performs forward passes only to calculate loss and accuracy.
 * @return Pointer to static array [average_loss, accuracy].
 */
double* model_test(NeuralNet* nn, double** X_test, double** y_test, double* y_test_temp, 
                   char* activation_fun, char* loss);

#endif // NEURAL_NETWORK_H