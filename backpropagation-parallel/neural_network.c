#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <numaif.h>
#include <sys/mman.h>

#include "read_data.h"
#include "activation.h"


// --- Memory & Alignment Settings ---
//this allows to allocate memory in contiguous zones of 2-level page table
//if memory is not enough, reserve another 512*512 zone and keep track of it
//to allow switching between k-size 2-level page table entries
//best: k = 1
#define MAX_ALLOC_LIMIT (512 * 512)
#define PAGE_SIZE 4096
// native alignement for pointers
#define PTR_ALIGNMENT 8
// alignement for huge pages 2MB
#define PAGE_ALIGNMENT 2097152

#define PDE_ALIGN_SIZE (2 * 1024 * 1024)      // 2MB Alignment for PDE Switching
#define WORKER_SLOT_SIZE (1024 * 1024)         // 0x100000 (1MB) per process
#define ALIGN_BLOCK(ptr) (char*)(((uintptr_t)(ptr) + 7) & ~7)
#define ALIGN_PAGE(sz) ({            \
    size_t __s = (sz);               \
    (__s + PAGE_ALIGNMENT - 1) & ~(PAGE_ALIGNMENT - 1); \
})


// Define global base address
unsigned long base_address = 0x1000000000; // Aligned to 2MB
int num_numa_nodes = 2;

// Initialize constants used in optimizers
const double epsilon = 1e-8;
const double beta = 0.9;
const double beta_1 = 0.9;
const double beta_2 = 0.999;


// Pseudo-random number generator 
int seed;
double randn(){
  int a = 1103515245;
  int m = 2147483647;
  int c = 12345;
  seed = (a * seed + c) % m;
  double x = (double)seed/(double)m;
  return x;
}


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
    void* initial_mmap_addr; //start address of mmap allocation
    size_t total_mmap_size; // total mapped region
    void *private_zone_start; // start of private per-process memory area

    int numa_node_id; // ID of the numa node
} NeuralNet;



/**
 * It computes the amount of memory required for one instance of Neural Network struct
 * with all its field. It sums up the sizes of the metadata, the pointers, 
 * and the raw double data based on the network architecture (n_layers and neurons per layer)
 * @param n_layers: Number of layers (L).
 * @param n_neurons_per_layer: Array specifying neurons per layer.
 * @return Total required size in bytes.
 * */
size_t sum_all_mmap_allocations(int n_layers, const int n_neurons_per_layer[]) {
    size_t total_required_size = 0;
    
    // Space for the struct itself
    total_required_size += sizeof(NeuralNet);
    
    // architecture metadata: the number of layers 
    total_required_size += (size_t)n_layers * sizeof(int);
    
    // Top-Level Pointers (Weight and Bias Matrices)
    // Weights: nn->w, nn->momentum_w, nn->momentum2_w: 3 pointer arrays
    total_required_size += (size_t)(n_layers - 1) * 3 * sizeof(double***); 
    
    // Bias: nn->b, nn->momentum_b, nn->momentum2_b: 3 pointer arrays
    total_required_size += (size_t)(n_layers - 1) * 3 * sizeof(double**);  

    // Loop over Weight Layers (Intermediate Pointers and Data)
    for (int i = 0; i < n_layers - 1; i++) {
        int n_in  = n_neurons_per_layer[i] + 1;    // Current layer (plus bias)
        int n_out = n_neurons_per_layer[i+1] + 1;  // Next layer (plus bias)

        // Intermediate Pointers (w[i], m_w[i], m2_w[i])
        // 3 arrays of pointers to rows
        total_required_size += (size_t)n_in * 3 * sizeof(double*);

        // Bias Data (b[i], m_b[i], m2_b[i])
        // 3 arrays of doubles
        total_required_size += (size_t)n_in * 3 * sizeof(double);
        
        // Weight Data (w[i][j], m_w[i][j], m2_w[i][j])
        // For each input neuron, we have 3 weight vectors of size n_out
        size_t weight_row_data_size = (size_t)n_out * 3 * sizeof(double);
        total_required_size += (size_t)n_in * weight_row_data_size;
    }
    
    // Top-Level Pointers (Backpropagation & Activations)
    
    // nn->delta, nn->in, nn->out: 3 pointer arrays
    total_required_size += (size_t)n_layers * 3 * sizeof(double*);

    
    // Backprop and Activation Data (Loop over all L layers)
    
    for (int i = 0; i < n_layers; i++) {
        int n = n_neurons_per_layer[i] + 1; 

        // delta[i], in[i], out[i]: 3 data arrays
        total_required_size += (size_t)n * 3 * sizeof(double);
    }

    // Target Vector
    // Used to store the desired one-hot labels for the current sample
    total_required_size += (size_t)(n_neurons_per_layer[n_layers-1] + 1) * sizeof(double);
    

    return total_required_size;
}



void* mmap_alloc(size_t size) {
    
    size_t aligned_size = (size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    
    void* addr = base_address;

    void* ret = mmap(addr, aligned_size,
                     PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
                     -1, 0);

    if (ret == MAP_FAILED) {
        perror("mmap");
        fprintf(stderr, "Failed to map at fixed address %p for size %zu (aligned to %zu)\n", addr, size, aligned_size);
        exit(1);
    }

    base_address = (void*)((char*)base_address + aligned_size);

    return ret;
}


/**
 * Function to create a neural network and allocate all its memory 
 * in a single contiguous block, aligned to 512 pages (2MB Huge Pages)
 * @param n_layers: int, number of layers
 * @param n_neurons_per_layer: int[], array of neurons per layer
 * @return struct NeuralNet*, pointer to the neural network
 */
NeuralNet* newNetSingleAlloc(int n_layers, int n_neurons_per_layer[], size_t aligned_size) {

    size_t total_numa_map_size = (size_t)num_numa_nodes * aligned_size; //total size considering numa nodes
    printf("[newNetSingleAlloc] size %.3f MB\n", (double)total_numa_map_size / (1024 * 1024));

    void* mmap_block = mmap_alloc(total_numa_map_size); //wraps mmap

    for (int numa_node = 0; numa_node < num_numa_nodes; numa_node++) {

        char * net_addr = (char *)mmap_block + (numa_node * PDE_ALIGN_SIZE); //each copy of the NN starts at fixed 2MB offsets
        
        //TODO mbind

        for(size_t i=0; i < PDE_ALIGN_SIZE; i += 4096)
            net_addr[i] = 0;

        struct NeuralNet* nn = (struct NeuralNet*)net_addr;
        nn->magic_test_value = (numa_node == 0) ? 123.456 : 999.888; //magic number for debugging pte switcher LKM

        char * current_ptr = net_addr; 
        // metadata and topology
        nn->n_layers = n_layers;
        nn->total_mmap_size = PDE_ALIGN_SIZE;
        nn->initial_mmap_addr = net_addr;
        
        current_ptr += sizeof(struct NeuralNet);
        current_ptr = ALIGN_BLOCK(current_ptr);

        nn->n_neurons_per_layer = (int*)current_ptr;
        for (int i = 0; i < n_layers; i++) 
            nn->n_neurons_per_layer[i] = n_neurons_per_layer[i];
        current_ptr += (n_layers * sizeof(int));
        current_ptr = ALIGN_BLOCK(current_ptr);

        // top level pointers
        size_t lp_sz = (n_layers - 1) * sizeof(double**); //weights
        size_t bp_sz = (n_layers - 1) * sizeof(double*); //bias
        size_t back_sz = n_layers * sizeof(double*); //activation

        nn->w = (double***)current_ptr;
        current_ptr += lp_sz;
        nn->momentum_w = (double***)current_ptr;
        current_ptr += lp_sz;
        nn->momentum2_w = (double***)current_ptr;
        current_ptr += lp_sz;
        nn->b = (double**)current_ptr;
        current_ptr += bp_sz;
        nn->momentum_b = (double**)current_ptr;
        current_ptr += bp_sz;
        nn->momentum2_b = (double**)current_ptr;
        current_ptr += bp_sz;
        nn->delta = (double**)current_ptr;
        current_ptr += back_sz;
        nn->in = (double**)current_ptr;
        current_ptr += back_sz;
        nn->out = (double**)current_ptr;
        current_ptr += back_sz;
        current_ptr = ALIGN_BLOCK(current_ptr);

        // intermediate (rows) pointers
        for (int i = 0; i < n_layers - 1; i++) {
            size_t rows = (size_t)(nn->n_neurons_per_layer[i] + 1); // + bias
            nn->w[i] = (double**)current_ptr;
            current_ptr += rows * sizeof(double*);
            nn->momentum_w[i] = (double**)current_ptr;
            current_ptr += rows * sizeof(double*);
            nn->momentum2_w[i] = (double**)current_ptr;
            current_ptr += rows * sizeof(double*);
        }
        current_ptr = ALIGN_BLOCK(current_ptr);

        // real data (Weights/Biases)
        for (int i = 0; i < n_layers - 1; i++) {
            int n_in = nn->n_neurons_per_layer[i] + 1;
            int n_out = nn->n_neurons_per_layer[i+1] + 1;
            size_t b_sz = (size_t)n_in * sizeof(double);
            size_t w_row_sz = (size_t)n_out * sizeof(double);

            // bias
            nn->b[i] = (double*)current_ptr;
            current_ptr += b_sz;
            nn->momentum_b[i] = (double*)current_ptr;
            current_ptr += b_sz;
            nn->momentum2_b[i] = (double*)current_ptr;
            current_ptr += b_sz;
            current_ptr = ALIGN_BLOCK(current_ptr);

            // weights 
            for (int j = 0; j < n_in; j++) {
                nn->w[i][j] = (double*)current_ptr;
                current_ptr += w_row_sz;
                nn->momentum_w[i][j] = (double*)current_ptr;
                current_ptr += w_row_sz;
                nn->momentum2_w[i][j] = (double*)current_ptr;
                current_ptr += w_row_sz;
            }
            current_ptr = ALIGN_BLOCK(current_ptr);
        }

        // private zone for backpropagation done by different processes
        // force to start on a new 4KB page to allow non-overlapping
        current_ptr = (char*)(((uintptr_t)current_ptr + 4095) & ~4095);
        // Store the start of private zone for easier mmap access
        nn->private_zone_start = (void*)current_ptr;

        // 5. Backprop Data (Scratchpad)
        for (int i = 0; i < n_layers; i++) {
            size_t sz = (size_t)(nn->n_neurons_per_layer[i] + 1) * sizeof(double);
            nn->delta[i] = (double*)current_ptr;
            current_ptr += sz;
            nn->in[i] = (double*)current_ptr;
            current_ptr += sz;
            nn->out[i] = (double*)current_ptr;
            current_ptr += sz;
            current_ptr = (char*)(((uintptr_t)current_ptr + 7) & ~7); 
        }

        // target vector
        nn->targets = (double*)current_ptr;
        current_ptr += (size_t)(nn->n_neurons_per_layer[n_layers-1] + 1) * sizeof(double);

        // sanity check
        if ((size_t)(current_ptr - net_addr) > PDE_ALIGN_SIZE) {
            fprintf(stderr, "FATAL: Node %d overflow\n", numa_node);
            exit(EXIT_FAILURE);
        }
    }
    return (struct NeuralNet*)mmap_block;
}




/** Function to free the dynamically allocated memory
 * @param nn: struct NeuralNet*, the neural network to free
 * 
 * */
void free_NN(NeuralNet* nn) {
    if (nn && nn->initial_mmap_addr) {
        // Since we mapped a huge chunk for all NUMA nodes, 
        // a real free_NN would likely need to munmap the whole shared region.
        // For now, we munmap the portion associated with this instance.
        munmap(nn->initial_mmap_addr, nn->total_mmap_size);
    }
}


/** Initialize the neural network
 * @param nn: struct NeuralNet* the neural network
 * */
void init_nn(struct NeuralNet* nn){

    // for each layer with out weights
    for(int k=0;k<nn->n_layers-1;k++){
        // for each neuron in layer k
        for(int i=1;i<nn->n_neurons_per_layer[k]+1;i++){
            nn->b[k][i] = 0.0;
            nn->momentum_b[k][i] = 0.0;
            nn->momentum2_b[k][i] = 0.0;
            //for each neuron on layer k+1
            for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){
                nn->w[k][i][j] = randn();
                nn->momentum_w[k][i][j] = 0.0;
                nn->momentum2_w[k][i][j] = 0.0;
            }
        }
    }
}


// Function to shuffle elements of an array
void shuffle(int* arr, size_t n){
    if(n > 1){
        for(size_t i=0;i<n-1;i++){
        size_t j = i+rand()/(RAND_MAX/(n-i)+1);
          int t = arr[j];
          arr[j] = arr[i];
          arr[i] = t;
        }
    }
}


/** Function for forward propagation step
 * @param nn: struct NeuralNet *, ptr to the neural network
 * @param activation_fun: char *, activation function
 * @param loss: char *, loss function
 * */
void forward_propagation(struct NeuralNet* nn, char* activation_fun, char* loss){

    // cleanup of input for each layer
    for(int i=0;i<nn->n_layers;i++){
        for(int j=0;j<nn->n_neurons_per_layer[i]+1;j++){
            nn->in[i][j] = 0.0;
        }
    }

    //for each layer
    for(int k=1;k<nn->n_layers;k++){

        /* Compute the weighted sum */
        // add bias
        for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
            nn->in[k][j] += 1.0 * nn->b[k-1][j];
        }

        // add weighed sum of outputs of prev layer
        for(int i=1;i<nn->n_neurons_per_layer[k-1]+1;i++){
            for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
                nn->in[k][j] += nn->out[k-1][i] * nn->w[k-1][i][j];
            }
        }

        /* Apply non-linear activation function to the weighted sums */

        //if last layer apply one of the loss functions
        if(k == nn->n_layers-1){
            if(strcmp(loss, "mse") == 0){ // mean square error
                for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
                    nn->out[k][j] = sigmoid(nn->in[k][j]); // apply sigmoid on in to obtain out
                }
            }
            else if(strcmp(loss, "ce") == 0){ // if cross-entropy
                double max_input_to_softmax = (double)INT_MIN; //use softmax

                for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){ // find max input
                    if(fabs(nn->in[k][j]) > max_input_to_softmax){
                        max_input_to_softmax = fabs(nn->in[k][j]);
                    }
                }
                double deno = 0.0;
                for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){ //compute denominator
                    nn->in[k][j] /= max_input_to_softmax;
                    deno += exp(nn->in[k][j]);
                }
                for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){ //apply softmax
                    nn->out[k][j] = (double)exp(nn->in[k][j])/(double)deno;
    
                }
            }
        } else{ // if hidden layer
            //for each neuron per layer apply activation function specified among sigmoid, tanh or relu
            for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
                if(strcmp(activation_fun, "sigmoid") == 0){
                    nn->out[k][j] = sigmoid(nn->in[k][j]);
                }
                else if(strcmp(activation_fun, "tanh") == 0){
                    nn->out[k][j] = tanh(nn->in[k][j]);
                }
                else if(strcmp(activation_fun, "relu") == 0){
                    nn->out[k][j] = relu(nn->in[k][j]);
                }
                else{
                    nn->out[k][j] = sigmoid(nn->in[k][j]);
                }
            }
        }
    }
}


// Function to calculate loss
double calc_loss(struct NeuralNet* nn, char* loss){
    double loss_val = 0.0;
    int last_layer = nn->n_layers-1;

    for(int i=1;i<nn->n_neurons_per_layer[last_layer]+1;i++){
        if(strcmp(loss, "mse") == 0){
            loss_val += (0.5)*(nn->out[last_layer][i] - nn->targets[i]) * (nn->out[last_layer][i] - nn->targets[i]);
        }
        else if(strcmp(loss, "ce") == 0){
            loss_val -= nn->targets[i]*(log(nn->out[last_layer][i]));
        }
	}
    return loss_val;
}


/** Function for back propagation step
 * @param nn: struct NeuralNet*, ptr to the neural network
 * @param activation_fun: char *, activation function
 * @param learning_rate: double, the learning rate
 * @param loss: char *, the loss function
 * @param opt: char *, optimizer used
 * @param itr: int, current iteration
 * 
 * */
void back_propagation(struct NeuralNet* nn, char* activation_fun, double learning_rate, char* loss, char* opt, int itr){

    int last_layer = nn->n_layers-1;

    /* Calculate the error in the output layer */
    for(int i=1;i<nn->n_neurons_per_layer[last_layer]+1;i++){
        if(strcmp(loss, "mse") == 0){
            double grad = sigmoid_d(nn->out[last_layer][i]);
            nn->delta[last_layer][i] = grad*(nn->out[last_layer][i] - nn->targets[i]);
        }
        else if(strcmp(loss, "ce") == 0){
            nn->delta[last_layer][i] = nn->out[last_layer][i] - nn->targets[i];
        }
    }

    /* Backpropagate the error from the last layer to the first layer */
    for(int k=nn->n_layers-2;k>0;k--){
        
        for(int i=1;i<nn->n_neurons_per_layer[k]+1;i++){

            // weighted sum of deltas of next layer
            double sum = 0.0;
            for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){
                sum += nn->b[k][j] * nn->delta[k+1][j];
            }
            for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){
                sum += nn->w[k][i][j] * nn->delta[k+1][j];
            }

            // compute gradient of activation function
            double grad;
            if(strcmp(activation_fun, "sigmoid") == 0){
                grad = sigmoid_d(nn->out[k][i]);
            }
            else if(strcmp(activation_fun, "tanh") == 0){
                grad = tanh_d(nn->out[k][i]);
            }
            else if(strcmp(activation_fun, "relu") == 0){
                grad = relu_d(nn->out[k][i]);
            }
            else{
                grad = sigmoid_d(nn->out[k][i]);
            }
            // delta of hidden layer
            nn->delta[k][i] = grad * sum;
        }
    }

    /* Update the weights according to the given optimization technique */
    //for each layer to update
    for(int k=0;k<nn->n_layers-1;k++){

        for(int i=1;i<nn->n_neurons_per_layer[k]+1;i++){
            for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){

                // dw is partial gradient of weight w
                double dw = nn->delta[k+1][j] * nn->out[k][i];

                if(strcmp(opt, "sgd") == 0){ //stochastic gradient descent
                    nn->w[k][i][j] -= learning_rate * dw;
                }
                else if(strcmp(opt, "momentum") == 0){
                    nn->momentum_w[k][i][j] = beta * nn->momentum_w[k][i][j] + (1.0-beta) * dw * learning_rate;
                    nn->w[k][i][j] -= nn->momentum_w[k][i][j];
                }
                else if(strcmp(opt, "rmsprop") == 0){
                    nn->momentum_w[k][i][j] = beta * nn->momentum_w[k][i][j] + (1.0-beta) * dw * dw;
                    nn->w[k][i][j] -= (learning_rate * dw)/(sqrt(nn->momentum_w[k][i][j]) + 1e-6);
                }
                else if(strcmp(opt, "adam") == 0){
                    nn->momentum_w[k][i][j] = beta_1 * nn->momentum_w[k][i][j] + (1.0-beta_1) * dw;
                    nn->momentum2_w[k][i][j] = beta_2 * nn->momentum2_w[k][i][j] + (1.0-beta_2) * dw * dw;
                    double m_cap = (double)nn->momentum_w[k][i][j]/(double)(1.0 - pow(beta_1, itr));
                    double v_cap = (double)nn->momentum2_w[k][i][j]/(double)(1.0 - pow(beta_2, itr));
                    nn->w[k][i][j] -= (learning_rate * m_cap)/(sqrt(v_cap) + epsilon);
                }
            }
        }

        /* Update the bias weights */
        for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){

            // db is partial gradient for bias b
            double db = nn->delta[k+1][j] * 1.0;

            if(strcmp(opt, "sgd") == 0){
                nn->b[k][j] -= learning_rate * db;
            }
            else if(strcmp(opt, "momentum") == 0){
                nn->momentum_b[k][j] = beta * nn->momentum_b[k][j] + (1.0-beta) * db * learning_rate;
                nn->b[k][j] -= nn->momentum_b[k][j];
            }
            else if(strcmp(opt, "rmsprop") == 0){
                nn->momentum_b[k][j] = beta * nn->momentum_b[k][j] + (1.0-beta) * db * db;
                nn->b[k][j] -= (learning_rate * db)/(sqrt(nn->momentum_b[k][j]) + 1e-6);
            }
            else if(strcmp(opt, "adam") == 0){
                nn->momentum_b[k][j] = beta_1 * nn->momentum_b[k][j] + (1.0-beta_1) * db;
                nn->momentum2_b[k][j] = beta_2 * nn->momentum2_b[k][j] + (1.0-beta_2) * db * db;
                double m_cap = (double)nn->momentum_b[k][j]/(double)(1.0 - pow(beta_1, itr));
                double v_cap = (double)nn->momentum2_b[k][j]/(double)(1.0 - pow(beta_2, itr));
                nn->b[k][j] -= (learning_rate * m_cap)/(sqrt(v_cap) + epsilon);
            }
        }
    }
}


/** Function to train the model for 1 epoch
 * @param nn: struct NeuralNet *, ptr to the neural network
 * @param X_train: double **, input training set
 * @param y_rain: double **, output labels one-hot encoding
 * @param y_train_temp: double *, output labels used for computing accuracy
 * @param activation_fun: char *, activation function
 * @param loss: char *, loss function
 * @param opt: char *, optimizer
 * @param learning_rate: double, learning rate
 * @param num_samples_to_train: int, num samples to train per epoch
 * @param itr: int, current iteration, used for bias correction
 * 
 * @return double *: metrics found
 * */
double* model_train(struct NeuralNet* nn, double** X_train, double** y_train, double* y_train_temp, 
                    char* activation_fun, char* loss, char* opt, double learning_rate,
                    int num_samples_to_train, int itr){

    // Create an array for generating random permutation of training sample indices
    int arr[N_SAMPLES]; //all samples
    for(int i=0;i<N_SAMPLES;i++){
        arr[i] = i;
    }
    shuffle(arr, N_SAMPLES); //shuffle array

    int shuffler[num_samples_to_train]; //only the samples used in this epoch
    for(int i=0;i<num_samples_to_train;i++){
        shuffler[i] = arr[i];
    }

    // Start training the model for 1 epoch and simultaneously calculate the training error and accuracy
    int correct = 0;
    double loss_val = 0.0;

    for(int i=0;i<num_samples_to_train;i++){ //for each sample

        //shuffle(shuffler, num_samples_to_train); //it doesn't seem to have any impact since arr[i] has already been shuffled before

        int idx = -1; //predicted class init
        double max_val = (double)INT_MIN;

        // arr[i] is a random index
        for(int j=1;j<nn->n_neurons_per_layer[0]+1;j++){
            // out[0] is the input layer, load the training set
            nn->out[0][j] = X_train[arr[i]][j-1];
        }
        for(int j=1;j<nn->n_neurons_per_layer[nn->n_layers-1]+1;j++){
            // targets is the desired output, load the labels
            nn->targets[j] = y_train[arr[i]][j-1];
        }

        forward_propagation(nn, activation_fun, loss);
        back_propagation(nn, activation_fun, learning_rate, loss, opt, itr);
        loss_val += calc_loss(nn, loss); // compute and accumulate loss
            
        // compute max and class predicted
        for(int j=1;j<nn->n_neurons_per_layer[nn->n_layers-1]+1;j++){
            if(nn->out[nn->n_layers-1][j] > max_val){
                max_val =nn->out[nn->n_layers-1][j];
                idx = j-1;
            }
        }
        if(idx == (int)y_train_temp[arr[i]]){
            correct++;
        }
    }

    //avg loss
    loss_val /=(double)num_samples_to_train;
    double accuracy = (double)correct/(double)num_samples_to_train;
    static double metrics[2];
    metrics[0] = loss_val;
    metrics[1] = accuracy;
    return metrics;
}


// Function to test the model
double* model_test(struct NeuralNet* nn, double** X_test, double** y_test, double* y_test_temp, char* activation_fun, char* loss){
    int correct = 0;
    double loss_val = 0.0;
    for(int i=0;i<N_TEST_SAMPLES;i++){
        int idx = -1;
        double max_val = (double)INT_MIN;
        for(int j=1;j<nn->n_neurons_per_layer[0]+1;j++){
            nn->out[0][j] = X_test[i][j-1];
        }
        for(int j=1;j<nn->n_neurons_per_layer[nn->n_layers-1]+1;j++){
            nn->targets[j] = y_test[i][j-1];
        }
        forward_propagation(nn, activation_fun, loss);
        loss_val += calc_loss(nn, loss);
            
        for(int j=1;j<nn->n_neurons_per_layer[nn->n_layers-1]+1;j++){
            if(nn->out[nn->n_layers-1][j] > max_val){
                max_val =nn->out[nn->n_layers-1][j];
                idx = j-1;
            }
        }
        if(idx == (int)y_test_temp[i]){
            correct++;
        }
    }
    loss_val /= (double)N_TEST_SAMPLES;
    double accuracy = (double)correct/(double)N_TEST_SAMPLES;
    static double metrics[2];
    metrics[0] = loss_val;
    metrics[1] = accuracy;
    return metrics;
}

int main(int argc, char *argv[]) {

    srand(time(NULL));

    // neural network architecture definition
    int n_layers = 4;
    int n_neurons_per_layer[] = {784, 64, 32, 10};
    int n_processes = 1;

    // computation of sizes for allocation
    size_t raw_size = sum_all_mmap_allocations(n_layers, n_neurons_per_layer);
    size_t aligned_size = ALIGN_PAGE(raw_size);

    printf("--- Network Configuration ---\n");
    printf("Architecture: [%d] -> [%d] -> [%d] -> [%d]\n", 
           n_neurons_per_layer[0], n_neurons_per_layer[1], 
           n_neurons_per_layer[2], n_neurons_per_layer[3]);
    printf("Raw Data Size: %.3f MB\n", (double)raw_size / (1024 * 1024));
    printf("Aligned Mmap Size (incl. Huge Pages): %.3f MB\n\n", (double)aligned_size / (1024 * 1024));

    // 4. Initialize Network with Single Allocation (Hogwild/NUMA style)
    printf("Initializing NUMA-aware shared memory allocation...\n");
    NeuralNet** nn_array = newNetSingleAlloc(n_layers, n_neurons_per_layer, aligned_size);
    printf("[DEBUG] nn_array addr = %p\n", (void*)nn_array);
    for (int i = 0; i < num_numa_nodes; i++) {
        printf("[DEBUG] nn_array[%d] = %p\n", i, (void*)nn_array+i*PDE_ALIGN_SIZE);
    }
    
    for (int i = 0; i < num_numa_nodes; i++) {
        NeuralNet *nn = (NeuralNet*)((char*)nn_array + i * PDE_ALIGN_SIZE);
        if (nn != NULL)  {
            nn->numa_node_id = i;
            nn->initial_mmap_addr = nn_array;
            nn->total_mmap_size = aligned_size;

            init_nn(nn);

            printf("Node %d: Base %p | NN %p | Weights[0][0] %p\n",
                   nn->numa_node_id,
                   nn->initial_mmap_addr,   // base pointer
                   (void*)nn,
                   (void*)nn->w[0][0]);
        } else {
            fprintf(stderr, "Critical Error: Allocation failed on NUMA node %d\n", i);
            return EXIT_FAILURE;
        }
    }

    // 5. Setup Hyper-parameters
    double learning_rate = 1e-4;
    double init_lr = 1e-4;
    char* activation_fun = "relu";
    char* loss = "ce";
    char* opt = "adam";
    int num_samples_to_train = 10000;
    int epochs = 5;

    // Override via CLI if provided
    if (argc > 1) learning_rate = atof(argv[1]);
    if (argc > 2) init_lr = atof(argv[2]);
    if (argc > 3) activation_fun = argv[3];
    if (argc > 4) loss = argv[4];
    if (argc > 5) opt = argv[5];
    if (argc > 6) num_samples_to_train = atoi(argv[6]);
    if (argc > 7) epochs = atoi(argv[7]);

    // 6. Data Loading and Pre-processing
    // Training Data
    double** X_train = malloc(N_SAMPLES * sizeof(double*));
    double** y_train = malloc(N_SAMPLES * sizeof(double*));
    double* y_train_temp = malloc(N_SAMPLES * sizeof(double));
    for(int i=0; i<N_SAMPLES; i++) {
        X_train[i] = malloc(N_DIMS * sizeof(double));
        y_train[i] = malloc(N_CLASSES * sizeof(double));
    }

    // Test Data
    double** X_test = malloc(N_TEST_SAMPLES * sizeof(double*));
    double** y_test = malloc(N_TEST_SAMPLES * sizeof(double*));
    double* y_test_temp = malloc(N_TEST_SAMPLES * sizeof(double));
    for(int i=0; i<N_TEST_SAMPLES; i++) {
        X_test[i] = malloc(N_DIMS * sizeof(double));
        y_test[i] = malloc(N_CLASSES * sizeof(double));
    }

    printf("Loading dataset and normalizing...\n");
    read_csv_file(X_train, y_train_temp, y_train, "train");
    scale_data(X_train, "train");
    read_csv_file(X_test, y_test_temp, y_test, "test");
    scale_data(X_test, "test");
    normalize_data(X_train, X_test);

    // 7. Training Loop
    FILE* file = fopen("metrics_64_32.txt", "w");
    fprintf(file, "train_loss,train_acc,test_loss,test_acc\n");
    
    struct NeuralNet* base_nn = nn_array;

    for(int itr = 0; itr < epochs; itr++) {
        double* train_m = model_train(base_nn, X_train, y_train, y_train_temp, activation_fun, loss, opt, learning_rate, num_samples_to_train, itr+1);
        double* test_m = model_test(base_nn, X_test, y_test, y_test_temp, activation_fun, loss);

        fprintf(file, "%lf,%lf,%lf,%lf\n", train_m[0], train_m[1], test_m[0], test_m[1]);

        printf("Epoch: %02d | Train Loss: %.4f | Acc: %.4f | Test Loss: %.4f | Acc: %.4f\n", 
                itr+1, train_m[0], train_m[1], test_m[0], test_m[1]);

        // Learning Rate Decay
        learning_rate = init_lr * exp(-0.1 * (itr+1));
    }

    // 8. Cleanup
    fclose(file);
    for (int i = 0; i < num_numa_nodes; i++) {
        NeuralNet* nn = (NeuralNet*)((char*)nn_array + i * PDE_ALIGN_SIZE);
        free_NN(nn);  // must handle sub-blocks, not nn_array[i]
    }

    // Free dataset
    for(int i=0; i<N_SAMPLES; i++) { 
        free(X_train[i]); 
        free(y_train[i]); 
    }
    free(X_train); 
    free(y_train); 
    free(y_train_temp);
    
    for(int i=0; i<N_TEST_SAMPLES; i++) { 
        free(X_test[i]); 
        free(y_test[i]); 
    }
    
    free(X_test); 
    free(y_test); 
    free(y_test_temp);

    return 0;
}