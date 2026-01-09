#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <string.h>
#include <sys/mman.h>

#include <numa.h>
#include <numaif.h>

#include "read_data.h"
#include "activation.h"


// Initialize constants used in optimizers
const double epsilon = 1e-8;
const double beta = 0.9;
const double beta_1 = 0.9;
const double beta_2 = 0.999;

//this allows to allocate memory in contiguous zones of 2-level page table
//if memory is not enough, reserve another 512*512 zone and keep track of it
//to allow switching between k-size 2-level page table entries
//best: k = 1
#define MAX_ALLOC_LIMIT 512*512

int num_numa_nodes = 1; //this is 1 for now

#define PAGE_SIZE 4096
// Allineamento nativo per i tipi di dati (double e puntatori su 64-bit)
#define PTR_ALIGNMENT 8

// Allineamento richiesto per le Huge Pages / 512 Pagine (2MB = 2097152 Byte)
#define PAGE_ALIGNMENT 2097152

/**
 * Funzione di utilità per allineare la dimensione al multiplo più vicino di 8 byte
 * Questo è usato per garantire che i campi all'interno del blocco contiguo siano allineati correttamente.
 * @param current_size: dimensione corrente.
 * @return Dimensione allineata a 8 byte.
 */
size_t align_block(size_t current_size) {
    return (current_size + PTR_ALIGNMENT - 1) & ~(PTR_ALIGNMENT - 1);
}

/**
 * Funzione di utilità per allineare la dimensione finale al multiplo più vicino di PAGE_ALIGNMENT (2MB).
 * Questo è usato per garantire che la dimensione totale per mmap_alloc rispetti l'allineamento.
 * @param current_size: dimensione corrente.
 * @return Dimensione allineata a 2MB.
 */
size_t align_page(size_t current_size) {
    return (current_size + PAGE_ALIGNMENT - 1) & ~(PAGE_ALIGNMENT - 1);
}
unsigned long base_address = 0x1000000000; // aligned to 2MB


size_t calculate_total_nn_size_for_single_mmap(int n_layers, const int n_neurons_per_layer[]);


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


// Neural Network struct definition

struct NeuralNet{
    int n_layers; // Stores the number of layers
    int* n_neurons_per_layer; // Stores the number of neurons in each layer
    
    /* weights and bias */
    double*** w; // Stores the weights between each neurons in each pair of layers, 3D: [layer][prev_neuron][curr_neuron]
    double** b; // Stores the bias weights from bias unit to neurons in the next layer, 2D: [layer][curr_neuron]
    
    /* optimzer variables */
    double*** momentum_w; // Stores the first order moment for the weights (mobile avg for derivatives)
    double*** momentum2_w; // Stores the second order moment for the weights (mobile avg for quadratic derivatives)
    double** momentum_b; // Stores the first order moment for the bias
    double** momentum2_b; // Stores the second order moment for the weights
    
    /* backpropagation parameters */
    double** delta; // Stores the errors computed for each neuron in each layer, 2D: [layer][neuron]
    double** in; // Stores the input to the activation function in each layer, 2D: [layer][neuron]
    double** out; // Stores the output of the activation function in each layer, 2D: [layer][neuron]

    double* targets; // Stores the actual output for a given sample. It is a one-hot vector

    void* initial_mmap_addr; //start address of mmap allocation
    size_t total_mmap_size; // total mapped region

    int numa_node_id;
};


void bind_memory_to_numa_node(void* addr, size_t size, int node) {
    
    unsigned long nodemask = (1UL << node);
    printf("numa node %d mask %lu\n", node, nodemask);
    if (mbind(addr, size, MPOL_BIND, &nodemask, 
              sizeof(unsigned long) * 8, MPOL_MF_STRICT) < 0) 
    {
        perror("mbind failed");
        // Handle error: possibly fall back to first-touch policy
    }
    
    printf("INFO: Memory range %p-%p bound to NUMA Node %d.\n", addr, (char*)addr + size, node);
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
struct NeuralNet* newNetSingleAlloc(int n_layers, int n_neurons_per_layer[]) {
    
    // Compute the total required size, aligned to the 2MB page boundary.
    // This size includes all data, pointers, and necessary padding for native alignment.
    size_t total_size = calculate_total_nn_size_for_single_mmap(n_layers, n_neurons_per_layer);


    // nn_array_size = num_numa_nodes * size of NeuralNet *
    size_t nn_array_size = (size_t)num_numa_nodes * sizeof(struct NeuralNet *);
    size_t nn_array_size_aligned = (nn_array_size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);

    // total_numa_map_size = total_size * num_numa_nodes + nn_array_size
    size_t total_numa_map_size = (total_size * (size_t)num_numa_nodes) + nn_array_size_aligned;


    // Map the entire memory block in a single call
    void* mmap_block = mmap_alloc(total_numa_map_size);
    
    // TODO NeuralNet **nn_array = (struct NeuralNet **)mmap_block
    struct NeuralNet **nn_array = (struct NeuralNet **) mmap_block;
    // char * current_start = (char *) mmap_block + nn_array_size
    char * current_start = (char *)mmap_block + nn_array_size_aligned;
    
    // Initialize the current pointer to traverse the allocated block.
    //char* current_ptr = (char*)mmap_block;


    for (int numa_node = 0; numa_node < num_numa_nodes; numa_node++) {
        /** for each numa node do
         *          void *net_addr = current_start + curr_numa_node * total_size) //address relativo alla singola copia per nodo numa
         *          mbind to numa node mbind(net_addr, total_size, numa_node,....)
         *          char *current_ptr = (char *) net_addr //current starts from this network
         *          *current_ptr = 'x'
         *          struct NeuralNet *nn = current_ptr
         *          nn_array[numa_node] = nn
         *          current_ptr += sizeof(struct NeuralNet);
         *          current_ptr = (char*)align_native((size_t)current_ptr);
         *          
         *          do the same as below
         * 
         * */

        void *net_addr = (void*)(current_start + numa_node*total_size);
        size_t aligned_net_size = (total_size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
        
        bind_memory_to_numa_node(net_addr, aligned_net_size, numa_node); //this should not be mbind! use it for a baseline only
        // this should be replaced with LKM numa support

        char * current_ptr = (char *) net_addr;

        // Materialization/First touch: Write a dummy value to trigger page allocation 
        *current_ptr = 'x'; 
        
        
        // The 'nn' pointer points to the very start of the memory block.
        struct NeuralNet* nn = (struct NeuralNet*)current_ptr;
        nn_array[numa_node] = nn;

        current_ptr += sizeof(struct NeuralNet);
        // Align the pointer for the next component - 8 byte
        current_ptr = (char*)align_block((size_t)current_ptr); 

        nn->n_layers = n_layers;
        nn->total_mmap_size = total_size;
        nn->initial_mmap_addr = net_addr;
        

        // Integer array storing neuron counts per layer
        nn->n_neurons_per_layer = (int*)current_ptr;
        for (int i = 0; i < n_layers; i++) {
            nn->n_neurons_per_layer[i] = n_neurons_per_layer[i];
        }
        current_ptr += (size_t)n_layers * sizeof(int);
        current_ptr = (char*)align_block((size_t)current_ptr);

        
        // Allocate space for (n_layers - 1) arrays of double*** for W and its derivatives 1 and 2 degree
        size_t top_ptr_size = (size_t)(n_layers - 1) * sizeof(double**); 

        nn->w           = (double***)current_ptr; 
        current_ptr += top_ptr_size;
        nn->momentum_w  = (double***)current_ptr; 
        current_ptr += top_ptr_size;
        nn->momentum2_w = (double***)current_ptr; 
        current_ptr += top_ptr_size;

        // Allocate space for (n_layers - 1) arrays of double** for B and its derivatives 1 and 2 degree
        size_t top_b_ptr_size = (size_t)(n_layers - 1) * sizeof(double*); 

        nn->b           = (double**)current_ptr; 
        current_ptr += top_b_ptr_size;
        nn->momentum_b  = (double**)current_ptr; 
        current_ptr += top_b_ptr_size;
        nn->momentum2_b = (double**)current_ptr; 
        current_ptr += top_b_ptr_size;
        
        current_ptr = (char*)align_block((size_t)current_ptr); //this keeps track of pointers

        
        // Pointer to track where the real data section starts (from w[0])
        char* data_start_ptr = current_ptr; 
        

        for (int i = 0; i < n_layers - 1; i++) {
            int n_in = nn->n_neurons_per_layer[i] + 1; //#rows in w[i]
            
            // INTERMEDIATE W POINTERS (w[i], m_w[i], m2_w[i])
            size_t mid_ptr_size = (size_t)n_in * sizeof(double*); //size of w[i]
            data_start_ptr = (char*)align_block((size_t)data_start_ptr + mid_ptr_size * 3); // consider w, momentum_w and momentum2_w
        }
        
        // data_ptr starts where all intermediate pointers end (start of real data section)
        char* data_ptr = data_start_ptr; //this keeps track of data inside current pointer

        // for each layer assign w, b and their momentums pointers to data_ptr
        for (int i = 0; i < n_layers - 1; i++) {
            int n_in  = nn->n_neurons_per_layer[i] + 1;    
            int n_out = nn->n_neurons_per_layer[i+1] + 1;

            // W ptr assignment
            size_t mid_ptr_size = (size_t)n_in * sizeof(double*);
            // Assign the pointer arrays using current_ptr
            nn->w[i]           = (double**)current_ptr; 
            current_ptr += mid_ptr_size;
            nn->momentum_w[i]  = (double**)current_ptr; 
            current_ptr += mid_ptr_size;
            nn->momentum2_w[i] = (double**)current_ptr; 
            current_ptr += mid_ptr_size;
            
            // B data assignment
            size_t bias_data_size = (size_t)n_in * sizeof(double);
            nn->b[i]           = (double*)data_ptr; 
            data_ptr += bias_data_size;
            nn->momentum_b[i]  = (double*)data_ptr; 
            data_ptr += bias_data_size;
            nn->momentum2_b[i] = (double*)data_ptr; 
            data_ptr += bias_data_size;
            data_ptr = (char*)align_block((size_t)data_ptr); // Align data block
            
            // W data assignment
            size_t weight_data_size = (size_t)n_out * sizeof(double);
            for (int j = 0; j < n_in; j++) {
                nn->w[i][j]           = (double*)data_ptr; 
                data_ptr += weight_data_size;
                nn->momentum_w[i][j]  = (double*)data_ptr; 
                data_ptr += weight_data_size;
                nn->momentum2_w[i][j] = (double*)data_ptr; 
                data_ptr += weight_data_size;
                data_ptr = (char*)align_block((size_t)data_ptr); // Align row data block
            }
        }
        
        // Update current_ptr after data assignment
        current_ptr = data_ptr; 


        // Allocate space for backpropagation parameters pointers
        size_t top_back_ptr_size = (size_t)n_layers * sizeof(double*);

        nn->delta = (double**)current_ptr; 
        current_ptr += top_back_ptr_size;
        nn->in    = (double**)current_ptr; 
        current_ptr += top_back_ptr_size;
        nn->out   = (double**)current_ptr; 
        current_ptr += top_back_ptr_size;
        
        current_ptr = (char*)align_block((size_t)current_ptr);

        
        for (int i = 0; i < n_layers; i++) {
            int n = nn->n_neurons_per_layer[i] + 1; 
            size_t data_size = (size_t)n * sizeof(double);

            // Assign data pointers and advance current_ptr
            nn->delta[i] = (double*)current_ptr; 
            current_ptr += data_size;
            nn->in[i]    = (double*)current_ptr; 
            current_ptr += data_size;
            nn->out[i]   = (double*)current_ptr; 
            current_ptr += data_size;
            current_ptr = (char*)align_block((size_t)current_ptr);
        }

        
        // Allocate space for the final target vector
        size_t targets_size = (size_t)(nn->n_neurons_per_layer[n_layers-1] + 1) * sizeof(double);
        nn->targets = (double*)current_ptr; current_ptr += targets_size;
        current_ptr = (char*)align_block((size_t)current_ptr);

        size_t actual_used_size = (size_t)(current_ptr - (char*)net_addr);

        printf("DEBUG: Calculated total size (2MB aligned): %zu bytes\n", total_size);
        printf("DEBUG: Actual size used (relative to net_addr): %zu bytes\n", actual_used_size);
        
        // The check must be against total_size, which is the allocated space for this single network
        if (actual_used_size > total_size) {
            fprintf(stderr, "FATAL ERROR: Network %d Assignment exceeded its allocated size (Used: %zu, Max: %zu)!\n", 
                    numa_node, actual_used_size, total_size);
            exit(EXIT_FAILURE);
        }
    }

    return nn_array;
}



/** Function to free the dynamically allocated memory
 * @param nn: struct NeuralNet*, the neural network to free
 * 
 * */
void free_NN(struct NeuralNet* nn){

    
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

    fprintf(stderr, "\n[DEBUG TRAIN] Starting model_train (NN Addr: %p, Samples: %d). n_layers: %d.\n", 
            (void*)nn, num_samples_to_train, nn->n_layers);
   
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
    fprintf(stderr, "[DEBUG TRAIN] Epoch finished. Final Loss: %f, Accuracy: %f.\n", loss_val, accuracy);
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


/**
 * Calcola la somma esatta di tutte le dimensioni passate alle chiamate mmap_alloc 
 * nella funzione newNet originale.
 *
 * @param n_layers: Numero di strati (L).
 * @param n_neurons_per_layer: Array che specifica i neuroni per strato.
 * @return La somma totale delle dimensioni richieste in byte.
 */
size_t sum_all_mmap_allocations(int n_layers, const int n_neurons_per_layer[]) {
    size_t total_required_size = 0;
    
    // -----------------------------------------------------------
    // 1. Struct NeuralNet
    // mmap_alloc(sizeof(struct NeuralNet))
    total_required_size += sizeof(struct NeuralNet);
    
    // -----------------------------------------------------------
    // 2. n_neurons_per_layer array
    // mmap_alloc(n_layers * sizeof(int))
    total_required_size += (size_t)n_layers * sizeof(int);
    
    // -----------------------------------------------------------
    // 3. Puntatori di Livello Superiore (Matrici di peso/bias)
    
    // nn->w, nn->momentum_w, nn->momentum2_w: 3 allocazioni
    total_required_size += (size_t)(n_layers - 1) * 3 * sizeof(double***); 
    
    // nn->b, nn->momentum_b, nn->momentum2_b: 3 allocazioni
    total_required_size += (size_t)(n_layers - 1) * 3 * sizeof(double**);  

    
    // -----------------------------------------------------------
    // 4. Loop sui livelli di peso (Puntatori Intermedi e Dati)
    
    for (int i = 0; i < n_layers - 1; i++) {
        int n_in  = n_neurons_per_layer[i] + 1;    
        int n_out = n_neurons_per_layer[i+1] + 1;  

        // 4.1. Puntatori Intermedi per W (w[i], m_w[i], m2_w[i]): 3 allocazioni
        // 3 * mmap_alloc(n_in * sizeof(double*))
        total_required_size += (size_t)n_in * 3 * sizeof(double*);

        // 4.2. Dati Bias (b[i], m_b[i], m2_b[i]): 3 allocazioni
        // 3 * mmap_alloc(n_in * sizeof(double))
        total_required_size += (size_t)n_in * 3 * sizeof(double);
        
        // 4.3. Dati Pesi (w[i][j], m_w[i][j], m2_w[i][j]): 3 allocazioni per riga, ripetute n_in volte
        // Loop interno: for (int j = 0; j < n_in; j++)
        // n_in * (3 * mmap_alloc(n_out * sizeof(double)))
        size_t weight_row_data_size = (size_t)n_out * 3 * sizeof(double);
        total_required_size += (size_t)n_in * weight_row_data_size;
    }
    
    // -----------------------------------------------------------
    // 5. Puntatori di Livello Superiore (Backprop)
    
    // nn->delta, nn->in, nn->out: 3 allocazioni
    // La dimensione è (n_layers * sizeof(double*)), non sizeof(double**), 
    // ma la differenza è solo il nome del tipo (puntatore a puntatore). Assumiamo sizeof(double*) == sizeof(double**)
    total_required_size += (size_t)n_layers * 3 * sizeof(double*);

    
    // -----------------------------------------------------------
    // 6. Dati Backprop (Loop su tutti gli L strati)
    
    for (int i = 0; i < n_layers; i++) {
        int n = n_neurons_per_layer[i] + 1; 

        // delta[i], in[i], out[i]: 3 allocazioni
        // 3 * mmap_alloc(n * sizeof(double))
        total_required_size += (size_t)n * 3 * sizeof(double);
    }

    // -----------------------------------------------------------
    // 7. Vettore Target
    // mmap_alloc((n_neurons_per_layer[n_layers-1] + 1) * sizeof(double))
    total_required_size += (size_t)(n_neurons_per_layer[n_layers-1] + 1) * sizeof(double);
    
    return total_required_size;
}



/**
 * Calculates the exact total size in bytes required for a single, contiguous allocation
 * that holds all pointers and data for one network instance. The result is aligned to 2MB.
 * @param n_layers: Number of layers in the network.
 * @param n_neurons_per_layer: Array defining the neuron count for each layer.
 * @return The total size required for mmap_alloc, aligned to the page size.
 */
size_t calculate_total_nn_size_for_single_mmap(int n_layers, const int n_neurons_per_layer[]) {
    size_t total_size = 0;
    
    // space for the NeuralNet struct itself
    total_size = align_block(total_size + sizeof(struct NeuralNet));
    
    // space for n_neurons_per_layer 
    total_size = align_block(total_size + (size_t)n_layers * sizeof(int));
    
    // space for w, b, momentum, momentum2
    // Three pointer arrays (w, m_w, m2_w) of double***
    size_t top_ptr_w_size = (size_t)(n_layers - 1) * 3 * sizeof(double**); 
    // Three pointer arrays (b, m_b, m2_b) of double**
    size_t top_ptr_b_size = (size_t)(n_layers - 1) * 3 * sizeof(double*); 
    total_size = align_block(total_size + top_ptr_w_size + top_ptr_b_size);

    
    // space for Intermediate Pointers and Real Data (Loop over weight layers)
    for (int i = 0; i < n_layers - 1; i++) {
        int n_in  = n_neurons_per_layer[i] + 1;    
        int n_out = n_neurons_per_layer[i+1] + 1;  

        // intermediate Pointers (w[i], m_w[i], m2_w[i])
        // This calculates the space for the array of double* pointers for this layer
        total_size = align_block(total_size + (size_t)n_in * 3 * sizeof(double*));

        // bias Data (b[i], m_b[i], m2_b[i])
        // Calculates the space for the actual double values for biases
        total_size = align_block(total_size + (size_t)n_in * 3 * sizeof(double));
        
        // weight Data (w[i][j], m_w[i][j], m2_w[i][j])
        // Calculates the space for all the weight rows (n_in rows)
        size_t weight_row_data_size = (size_t)n_out * 3 * sizeof(double);
        total_size = align_block(total_size + (size_t)n_in * weight_row_data_size);
    }
    
    // space Backprop: delta, in, out)
    // Three pointer arrays (delta, in, out) of double*
    size_t top_ptr_back_size = (size_t)n_layers * 3 * sizeof(double*);      
    total_size = align_block(total_size + top_ptr_back_size);
    
    // space for Backprop Data (delta[i], in[i], out[i])
    for (int i = 0; i < n_layers; i++) {
        int n = n_neurons_per_layer[i] + 1; 
        // Calculates the space for the actual double values for delta, in, and out
        total_size = align_block(total_size + (size_t)n * 3 * sizeof(double));
    }

    // space for the Target Vector
    size_t targets_size = (size_t)(n_neurons_per_layer[n_layers-1] + 1) * sizeof(double);
    total_size = align_block(total_size + targets_size);
    
    // final Alignment to 2MB (PAGE_ALIGNMENT)
    return align_page(total_size);
}


int main(int argc, char *argv[]){

    // Used for setting a random seed
    srand(time(NULL));
    int seed = rand();


    if (base_address % 0x200000 == 0)
        printf("base address aligned to 2MB\n");



    // Initialize neural network architecture parameters
    int n_layers = 4;
    int n_neurons_per_layer[] = {784, 64, 32, 10};


    size_t total_size = sum_all_mmap_allocations(n_layers, n_neurons_per_layer);

    printf("--- Compute network size ---\n");
    printf("Network: [Input: %d] -> [Hidden: %d] -> [Output: %d]\n", 
           n_neurons_per_layer[0], n_neurons_per_layer[1], n_neurons_per_layer[2]);
    printf("\n");
    printf("Sum total size (Byte): %zu\n", total_size);
    printf("Sum total size (MB): %.3f\n", (double)total_size / (1024 * 1024));
    printf("\n");
    printf("NOTE: This value is the sum of all mmalloc requests and DOES NOT include\n");
    printf("internal padding nor alignment to 2MB pages.\n");


    printf("Init neural network with a single allocation\n");

    struct NeuralNet** nn_array = newNetSingleAlloc(n_layers, n_neurons_per_layer);

    printf("\n--- Recap NUMA allocation ---\n");
    for (int i = 0; i < num_numa_nodes; i++) {
        struct NeuralNet* nn = nn_array[i];
        if (nn!=NULL) {
            printf("Node %d: Base Addr: %p, Total Size: %zu bytes, NN Addr: %p, Weights[0][0]: %p\n",
                   nn->numa_node_id, nn->initial_mmap_addr, nn->total_mmap_size, nn, nn->w[0][0]);
            
            // Initialize each network instance
            init_nn(nn); 
        } else {
            fprintf(stderr, "ERROR: Network instance %d failed to allocate.\n", i);
        }
    }
    printf("\nMulti-NUMA allocation and initialization completed successfully.\n");


    struct NeuralNet* base_nn = nn_array[0]; 
    
    // Create and initialize the neural network
    //struct NeuralNet* nn = newNet(n_layers, n_neurons_per_layer);
    //init_nn(nn);

    // Initialize the learning rate, optimizer, loss, and other hyper-parameters
    double learning_rate = 1e-4;
    double init_lr = 1e-4;
    char* activation_fun = "relu";
    char* loss = "ce";
    char* opt = "adam";
    int num_samples_to_train = 10000;
    int epochs = 5;

    

    if (argc > 1) learning_rate = atof(argv[1]);
    if (argc > 2) init_lr = atof(argv[2]);
    if (argc > 3) activation_fun = argv[3];
    if (argc > 4) loss = argv[4];
    if (argc > 5) opt = argv[5];
    if (argc > 6) num_samples_to_train = atoi(argv[6]);
    if (argc > 7) epochs = atoi(argv[7]);

    printf("Using parameters:\n");
    printf("  learning_rate = %f\n", learning_rate);
    printf("  init_lr = %f\n", init_lr);
    printf("  activation_fun = %s\n", activation_fun);
    printf("  loss = %s\n", loss);
    printf("  opt = %s\n", opt);
    printf("  num_samples_to_train = %d\n", num_samples_to_train);
    printf("  epochs = %d\n", epochs);



    // Fetch the training and test data and pre-process them
    double** X_train = malloc(N_SAMPLES*sizeof(double*));
    for(int i=0;i<N_SAMPLES;i++){
        X_train[i] = malloc(N_DIMS*sizeof(double));
    }
    double** y_train = malloc(N_SAMPLES * sizeof(double*));
    for(int i=0;i<N_SAMPLES;i++){
        y_train[i] = malloc(N_CLASSES * sizeof(double));
    }
    double* y_train_temp = malloc(N_SAMPLES*sizeof(double));
    read_csv_file(X_train, y_train_temp, y_train, "train");
    scale_data(X_train, "train");

    double** X_test = malloc(N_TEST_SAMPLES*sizeof(double*));
    for(int i=0;i<N_TEST_SAMPLES;i++){
        X_test[i] = malloc(N_DIMS*sizeof(double));
    }
    double** y_test = malloc(N_TEST_SAMPLES * sizeof(double*));
    for(int i=0;i<N_TEST_SAMPLES;i++){
        y_test[i] = malloc(N_CLASSES * sizeof(double));
    }
    double* y_test_temp = malloc(N_TEST_SAMPLES*sizeof(double));
    read_csv_file(X_test, y_test_temp, y_test, "test");
    scale_data(X_test, "test");
    normalize_data(X_train, X_test);

    // Initialize file to store metrics info for each epoch
    FILE* file = fopen("metrics_64_32.txt", "w");
    fprintf(file, "train_loss,train_acc,test_loss,test_acc\n");
    
    // Train the model for given number of epoch and test it after every epoch
    for(int itr=0;itr<epochs;itr++){
        double* train_metrics = model_train(base_nn, X_train, y_train, y_train_temp, activation_fun, loss, opt, learning_rate, num_samples_to_train, itr+1);
        double train_loss = train_metrics[0];
        double train_acc = train_metrics[1];
        double* test_metrics = model_test(base_nn, X_test, y_test, y_test_temp, activation_fun, loss);
        double test_loss = test_metrics[0];
        double test_acc = test_metrics[1];

        fprintf(file, "%lf,", train_loss);
        fprintf(file, "%lf,", train_acc);
        fprintf(file, "%lf,", test_loss);
        fprintf(file, "%lf\n", test_acc);

        printf("Epoch: %d -> ", itr+1);
        printf("Train loss: %lf, ", train_loss);
        printf("Train Accuracy: %lf, ", train_acc);
        printf("Test loss: %lf, ", test_loss);
        printf("Test Accuracy: %lf\n", test_acc);

        learning_rate = init_lr * exp(-0.1 * (itr+1));

    }

    // Close the file
    fclose(file);

    // Free the dynamically allocated memory
    //free_NN(nn);

    // Free the individual network resources
    for (int i = 0; i < num_numa_nodes; i++) {
        if (nn_array[i]) {
            free_NN(nn_array[i]); // Should call munmap/free on nn_array[i]->initial_mmap_addr
        }
    }


    // Free the data arrays (Data is still on the heap)
    for(int i=0;i<N_SAMPLES;i++) free(X_train[i]);
    free(X_train);
    for(int i=0;i<N_SAMPLES;i++) free(y_train[i]);
    free(y_train);
    free(y_train_temp);

    for(int i=0;i<N_TEST_SAMPLES;i++) free(X_test[i]);
    free(X_test);
    for(int i=0;i<N_TEST_SAMPLES;i++) free(y_test[i]);
    free(y_test);
    free(y_test_temp);

    return 0;
}
