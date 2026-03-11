#define _GNU_SOURCE 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#ifdef NUMA_API_ENABLED
#include <numaif.h>
#include <numa.h>
#include <sys/sysinfo.h>
#include <sys/mman.h>
#include <sched.h>       // Per cpu_set_t, CPU_ZERO, sched_setaffinity
#include <unistd.h>     // Per fork()
#include <sys/types.h>  // Per pid_t
#include <sys/wait.h>   // Per wait()
#endif
#include "read_data.h"
#include "activation.h"
#include "neural_network.h"



#ifdef NUMA_API_ENABLED


// Define global base address
unsigned long base_address = 0x1000000000; // Aligned to 2MB
int num_numa_nodes = 2;
#endif

// Initialize constants used in optimizers
const double epsilon = 1e-8;
const double beta = 0.9;
const double beta_1 = 0.9;
const double beta_2 = 0.999;




NeuralNet* newNetSharedAlloc(int n_layers, int n_neurons_per_layer[], size_t aligned_shared_size);

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

#ifdef NUMA_API_ENABLED
// -------- Utilities for size computation --------

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

/**
 * This functions computes the amount of memory needed for transient data
 * such as delta, in, out, targets → the pointers kept by each process in its own private memory zone
 * It also computes the memory needed for local gradients (weights and bias)
 * */
size_t calculate_private_workspace_size(int n_layers, const int n_neurons_per_layer[]) {
    size_t size = 0;

    // --- 1. PUNTATORI INTERMEDI (Shadow Pointers) ---
    // Questi devono essere privati per evitare che processi sullo stesso nodo 
    // sovrascrivano dove puntano le righe di dw, db, momentum.
    size = ALIGN_SIZE(size + (size_t)(n_layers - 1) * sizeof(double*)); // per db
    size = ALIGN_SIZE(size + (size_t)(n_layers - 1) * 2 * sizeof(double*)); // per momentum_b/2
    
    for (int i = 0; i < n_layers - 1; i++) {
        int n_in = n_neurons_per_layer[i] + 1;
        // Array di puntatori per ogni riga di dw, momentum_w, momentum2_w
        size = ALIGN_SIZE(size + (size_t)n_in * 3 * sizeof(double*)); 
    }

    // --- 2. ATTIVAZIONI E DELTA (Puntatori + Dati) ---
    size = ALIGN_SIZE(size + (size_t)n_layers * 3 * sizeof(double*)); // in, out, delta
    for (int i = 0; i < n_layers; i++) {
        int n = n_neurons_per_layer[i] + 1;
        size = ALIGN_SIZE(size + (size_t)n * 3 * sizeof(double));
    }

    // --- 3. DATI REALI (Gradienti e Momentum) ---
    for (int i = 0; i < n_layers - 1; i++) {
        int n_in  = n_neurons_per_layer[i] + 1;
        int n_out = n_neurons_per_layer[i+1] + 1;
        
        // Dati bias: db, mb, m2b
        size = ALIGN_SIZE(size + (size_t)n_out * 3 * sizeof(double));

        // Dati pesi: dw, mw, m2w per ogni riga
        for (int j = 0; j < n_in; j++) {
            size = ALIGN_SIZE(size + (size_t)n_out * 3 * sizeof(double));
        }
    }

    // Target
    size = ALIGN_SIZE(size + (size_t)(n_neurons_per_layer[n_layers-1] + 1) * sizeof(double));

    return size;
}


/**
 * This functions only computes the amount of memory needed for the shared part of the network
 * excluding all parameters that change at each iteration of training (momentum_w, momentum_b, in, out, delta and targets)
 * */
size_t calculate_shared_model_size(int n_layers, const int n_neurons_per_layer[]) {
    size_t size = 0;
    
    // 1. NeuralNet struct + architecture array
    size = ALIGN_SIZE(size + sizeof(struct NeuralNet));
    size = ALIGN_SIZE(size + (size_t)n_layers * sizeof(int));
    
    // 2. TUTTI i Top-level pointers (quelli che allochi prima del loop dei row pointers)
    // Devi contare ogni riga della tua newNetSharedAlloc che sposta current_ptr
    size += (n_layers - 1) * sizeof(double***); // w
    size += (n_layers - 1) * sizeof(double***); // dw
    size += (n_layers - 1) * sizeof(double**);  // b
    size += (n_layers - 1) * sizeof(double**);  // db
    size += (n_layers - 1) * sizeof(double***); // momentum_w
    size += (n_layers - 1) * sizeof(double***); // momentum2_w
    size += (n_layers - 1) * sizeof(double**);  // momentum_b
    size += (n_layers - 1) * sizeof(double**);  // momentum2_b
    size += n_layers * sizeof(double**);        // delta
    size += n_layers * sizeof(double**);        // in
    size += n_layers * sizeof(double**);        // out
    size = ALIGN_SIZE(size);

    // 3. Row pointers (dw[i], w[i], etc.)
    for (int i = 0; i < n_layers - 1; i++) {
        size_t n_in = (size_t)(n_neurons_per_layer[i] + 1);
        // Allineato alla tua allocazione: w[i], dw[i], momentum_w[i], momentum2_w[i]
        size += n_in * sizeof(double*) * 4; 
    }
    size = ALIGN_SIZE(size);

    // 4. Real Data (Solo Pesi e Bias)
    for (int i = 0; i < n_layers - 1; i++) {
        int n_in  = n_neurons_per_layer[i] + 1;
        int n_out = n_neurons_per_layer[i+1] + 1;

        // Bias data (nn->b[i])
        size += n_in * sizeof(double);
        size = ALIGN_SIZE(size);

        // Weights data (nn->w[i][j])
        for (int j = 0; j < n_in; j++) {
             size += n_out * sizeof(double);
        }
        size = ALIGN_SIZE(size);
    }
    return size;
}

// -------- Utilities for memory management --------


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
    
    void* addr = (void*)base_address;

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
#endif


/** Function to create a neural network and allocate memory
* @param n_layers: int, number of layers
* @param n_neurons_per_layer: int[], array of neurons per layers
* @return struct NeuralNet*, a neural network ptr 
*/
struct NeuralNet* newNet(int n_layers, int n_neurons_per_layer[]){

    struct NeuralNet* nn = malloc(sizeof(struct NeuralNet));
    nn->n_layers = n_layers;

    // alloc neurons per layer and c
    nn->n_neurons_per_layer = malloc(nn->n_layers * sizeof(int));
    for(int i=0;i<n_layers;i++){
        nn->n_neurons_per_layer[i] = n_neurons_per_layer[i];
    }

    // alloc weights, biases and momentum states
    nn->w = malloc((nn->n_layers-1)*sizeof(double**));
    nn->momentum_w = malloc((nn->n_layers-1)*sizeof(double**));
    nn->momentum2_w = malloc((nn->n_layers-1)*sizeof(double**));
    nn->b = malloc((nn->n_layers-1)*sizeof(double*));
    nn->momentum_b = malloc((nn->n_layers-1)*sizeof(double*));
    nn->momentum2_b = malloc((nn->n_layers-1)*sizeof(double*));
    for(int i=0;i<nn->n_layers-1;i++){
        nn->w[i] = malloc((nn->n_neurons_per_layer[i] + 1)*sizeof(double*));
        nn->momentum_w[i] = malloc((nn->n_neurons_per_layer[i] + 1)*sizeof(double*));
        nn->momentum2_w[i] = malloc((nn->n_neurons_per_layer[i] + 1)*sizeof(double*));
        nn->b[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        nn->momentum_b[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        nn->momentum2_b[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        for(int j=0;j<nn->n_neurons_per_layer[i]+1;j++){
            nn->w[i][j] = malloc((nn->n_neurons_per_layer[i+1] + 1)*sizeof(double));
            nn->momentum_w[i][j] = malloc((nn->n_neurons_per_layer[i+1] + 1)*sizeof(double));
            nn->momentum2_w[i][j] = malloc((nn->n_neurons_per_layer[i+1] + 1)*sizeof(double));
        }
    }

    // alloc of backpropagation parameters
    nn->delta = malloc((nn->n_layers)*sizeof(double*));
    nn->in = malloc((nn->n_layers)*sizeof(double*));
    nn->out = malloc((nn->n_layers)*sizeof(double*));
    for(int i=0;i<nn->n_layers;i++){
        nn->in[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        nn->out[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        nn->delta[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
    }

    // alloc target one-hot vector
    nn->targets = malloc((nn->n_neurons_per_layer[nn->n_layers-1]+1)*sizeof(double));
    return nn;
}


#ifdef NUMA_API_ENABLED
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
        
        //mbind
        bind_memory_to_numa_node(net_addr, aligned_size, numa_node);

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
        nn->b = (double**)current_ptr;
        current_ptr += bp_sz;
        nn->momentum_w = (double***)current_ptr;
        current_ptr += lp_sz;
        nn->momentum2_w = (double***)current_ptr;
        current_ptr += lp_sz;
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
        nn->private_zone_start = NULL;

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


// in this function the pointers to: in, out, delta, targets and momentum are set to NULL
// they are not allocated yet, each process will allocate them privately
NeuralNet* newNetSharedAlloc(int n_layers, int n_neurons_per_layer[], size_t aligned_shared_size) {
    size_t total_numa_map_size = (size_t)num_numa_nodes * aligned_shared_size;
    void* mmap_block = mmap_alloc(total_numa_map_size); 

    for (int numa_node = 0; numa_node < num_numa_nodes; numa_node++) {
        char * net_addr = (char *)mmap_block + (numa_node * aligned_shared_size);
        bind_memory_to_numa_node(net_addr, aligned_shared_size, numa_node);

        struct NeuralNet* nn = (struct NeuralNet*)net_addr;
        char * current_ptr = net_addr; 

        // metadata
        nn->n_layers = n_layers;
        current_ptr += sizeof(struct NeuralNet);
        current_ptr = ALIGN_BLOCK(current_ptr);

        nn->n_neurons_per_layer = (int*)current_ptr;
        for (int i = 0; i < n_layers; i++) nn->n_neurons_per_layer[i] = n_neurons_per_layer[i];
        current_ptr += (n_layers * sizeof(int));
        current_ptr = ALIGN_BLOCK(current_ptr);

        // top level pointers
        nn->w = (double***)current_ptr;           
        current_ptr += (n_layers - 1) * sizeof(double**);
        nn->dw = (double***)current_ptr;          
        current_ptr += (n_layers - 1) * sizeof(double**); 

        nn->b = (double**)current_ptr;            
        current_ptr += (n_layers - 1) * sizeof(double*);
        nn->db = (double**)current_ptr;           
        current_ptr += (n_layers - 1) * sizeof(double*);  
        
        nn->momentum_w = (double***)current_ptr;  
        current_ptr += (n_layers - 1) * sizeof(double**);
        nn->momentum2_w = (double***)current_ptr; 
        current_ptr += (n_layers - 1) * sizeof(double**);
        

        nn->momentum_b = (double**)current_ptr;   
        current_ptr += (n_layers - 1) * sizeof(double*);
        nn->momentum2_b = (double**)current_ptr;  
        current_ptr += (n_layers - 1) * sizeof(double*);
        
        nn->delta = (double**)current_ptr;        
        current_ptr += n_layers * sizeof(double*);
        nn->in = (double**)current_ptr;           
        current_ptr += n_layers * sizeof(double*);
        nn->out = (double**)current_ptr;          
        current_ptr += n_layers * sizeof(double*);
        current_ptr = ALIGN_BLOCK(current_ptr);

        // row pointers
        for (int i = 0; i < n_layers - 1; i++) {
            size_t n_in = (size_t)(nn->n_neurons_per_layer[i] + 1);
            
            nn->w[i] = (double**)current_ptr;           
            current_ptr += n_in * sizeof(double*);
            nn->dw[i] = (double**)current_ptr;          
            current_ptr += n_in * sizeof(double*);
            nn->momentum_w[i] = (double**)current_ptr;  
            current_ptr += n_in * sizeof(double*);
            nn->momentum2_w[i] = (double**)current_ptr; 
            current_ptr += n_in * sizeof(double*);
        }
        current_ptr = ALIGN_BLOCK(current_ptr);

        // real data -- most of it will be in private zone
        for (int i = 0; i < n_layers - 1; i++) {
            int n_in = nn->n_neurons_per_layer[i] + 1;
            int n_out = nn->n_neurons_per_layer[i+1] + 1;

            // bias
            nn->b[i] = (double*)current_ptr;            
            current_ptr += n_in * sizeof(double);
            nn->db[i] = NULL; // init in private zone
            nn->momentum_b[i] = NULL;                   // init in private zone
            nn->momentum2_b[i] = NULL;                  // init in private zone
            
            current_ptr = ALIGN_BLOCK(current_ptr);

            // weights
            for (int j = 0; j < n_in; j++) {
                // real weight in shared zone
                nn->w[i][j] = (double*)current_ptr;     
                current_ptr += n_out * sizeof(double);
                
                nn->dw[i][j] = NULL; // init in private zone
                // momentum will be in private zone
                nn->momentum_w[i][j] = NULL;            
                nn->momentum2_w[i][j] = NULL;           
            }
            current_ptr = ALIGN_BLOCK(current_ptr);
        }

        // activations and targets -- private zone
        for (int i = 0; i < n_layers; i++) {
            nn->delta[i] = NULL;
            nn->in[i] = NULL;
            nn->out[i] = NULL;
        }
        nn->targets = NULL;
        nn->private_zone_start = NULL;

        // Sanity Check
        if ((size_t)(current_ptr - net_addr) > aligned_shared_size) {
            fprintf(stderr, "FATAL: Shared allocation overflow\n");
            exit(EXIT_FAILURE);
        }
    }
    return (struct NeuralNet*)mmap_block;
}
#endif

/** Function to free the dynamically allocated memory
 * @param nn: struct NeuralNet*, the neural network to free
 * 
 * */
void free_NN(NeuralNet* nn) {
    //for each layer free from inside-out everything allocated in newNet
    for(int i=0;i<nn->n_layers-1;i++){
        for(int j=0;j<nn->n_neurons_per_layer[i]+1;j++){
            free(nn->w[i][j]);
            free(nn->momentum_w[i][j]);
            free(nn->momentum2_w[i][j]);
        }
        free(nn->w[i]);
        free(nn->momentum_w[i]);
        free(nn->momentum2_w[i]);
        free(nn->b[i]);
        free(nn->momentum_b[i]);
        free(nn->momentum2_b[i]);
    }
    free(nn->w);
    free(nn->momentum_w);
    free(nn->momentum2_w);
    free(nn->b);
    free(nn->momentum_b);
    free(nn->momentum2_b);
    
    // free backpropagation parameters
    for(int i=0;i<nn->n_layers;i++){
        free(nn->in[i]);
        free(nn->out[i]);
        free(nn->delta[i]);
    }

    free(nn->in);
    free(nn->out);
    free(nn->delta);
    
    free(nn->targets);
    free(nn->n_neurons_per_layer);

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


#ifdef NUMA_API_ENABLED
void unmap_nn(struct NeuralNet *nn) {

    if (nn && nn->initial_mmap_addr) {
        // Since we mapped a huge chunk for all NUMA nodes, 
        // a real free_NN would likely need to munmap the whole shared region.
        // For now, we munmap the portion associated with this instance.
        munmap(nn->initial_mmap_addr, nn->total_mmap_size);
    }
}


void init_nn_shared(struct NeuralNet* nn){
    // only init shared part of network
    for(int k=0; k < nn->n_layers - 1; k++){
        for(int i=1; i < nn->n_neurons_per_layer[k] + 1; i++){
            
            // Bias 
            nn->b[k][i] = 0.0; 
            
            for(int j=1; j < nn->n_neurons_per_layer[k+1] + 1; j++){
                // weights
                nn->w[k][i][j] = randn(); 
                
            }
        }
    }
}


// -------- Wrapped by pragma for allocation of new neural network --------

/**
 * Setup function to allocate and initialize a neural network 
 * this must be called by the annotation pragma
 */
NeuralNet* setup_numa_model(int n_layers, int n_neurons_per_layer[]) {

    size_t raw_shared_size = calculate_shared_model_size(n_layers, n_neurons_per_layer);
    
    size_t aligned_size = (raw_shared_size + PDE_ALIGN_SIZE - 1) & ~(PDE_ALIGN_SIZE - 1);

    // allocation
    NeuralNet* nn_array = newNetSharedAlloc(n_layers, n_neurons_per_layer, aligned_size);

    // init neural network for each numa node
    for (int i = 0; i < num_numa_nodes; i++) {
        // Calcolo l'offset per la i-esima copia (Slab)
        NeuralNet *nn = (NeuralNet*)((char*)nn_array + i * aligned_size);
        
        if (nn != NULL) {
            nn->numa_node_id = i;
            nn->initial_mmap_addr = nn_array;
            nn->total_mmap_size = aligned_size;

            // init neural network
            init_nn_shared(nn); 

            printf("[NODE %d] Base: %p | Weights[0][0]: %p\n", 
                    i, (void*)nn, (void*)nn->w[0][0]);
        } else {
            fprintf(stderr, "Error allocating network copy %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    return nn_array;
}
#endif

// -------- Training functions --------


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
            for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){ //check this loop it might be wrong
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


#ifdef NUMA_API_ENABLED
/* ------------- BELOW the new model train in parallel setting */

void back_propagation_accumulate(struct NeuralNet* nn, char* activation_fun, char* loss){
    int last = nn->n_layers-1;
    for(int i=1;i<nn->n_neurons_per_layer[last]+1;i++){
        if(strcmp(loss, "mse") == 0) nn->delta[last][i] = (nn->out[last][i] - nn->targets[i]) * sigmoid_d(nn->out[last][i]);
        else nn->delta[last][i] = nn->out[last][i] - nn->targets[i];
    }
    for(int k=nn->n_layers-2; k>0; k--){
        for(int i=1; i<nn->n_neurons_per_layer[k]+1; i++){
            double sum = 0.0;
            for(int j=1; j<nn->n_neurons_per_layer[k+1]+1; j++) sum += nn->w[k][i][j] * nn->delta[k+1][j];
            double grad;
            if(strcmp(activation_fun, "tanh") == 0) grad = tanh_d(nn->out[k][i]);
            else if(strcmp(activation_fun, "relu") == 0) grad = relu_d(nn->out[k][i]);
            else grad = sigmoid_d(nn->out[k][i]);
            nn->delta[k][i] = grad * sum;
        }
    }
    for(int k=0; k<nn->n_layers-1; k++){
        for(int j=1; j<nn->n_neurons_per_layer[k+1]+1; j++){
            nn->db[k][j] += nn->delta[k+1][j];
            for(int i=1; i<nn->n_neurons_per_layer[k]+1; i++) nn->dw[k][i][j] += nn->delta[k+1][j] * nn->out[k][i];
        }
    }
}

void update_weights_batch(struct NeuralNet* nn, double lr, char* opt, int itr, int batch_size) {
    
    for(int k=0; k < nn->n_layers-1; k++){
        for(int j=1; j < nn->n_neurons_per_layer[k+1]+1; j++){
            
            // update bias
            double grad_b = nn->db[k][j] / (double)batch_size;
            
            if(strcmp(opt, "sgd") == 0) {
                nn->b[k][j] -= lr * grad_b;
            } 
            else if(strcmp(opt, "adam") == 0) {
                nn->momentum_b[k][j] = beta_1 * nn->momentum_b[k][j] + (1.0-beta_1) * grad_b;
                nn->momentum2_b[k][j] = beta_2 * nn->momentum2_b[k][j] + (1.0-beta_2) * grad_b * grad_b;
                
                double m_cap = nn->momentum_b[k][j] / (1.0 - pow(beta_1, itr));
                double v_cap = nn->momentum2_b[k][j] / (1.0 - pow(beta_2, itr));
                
                nn->b[k][j] -= (lr * m_cap) / (sqrt(v_cap) + epsilon);
            }
            
            // zero bias for next batch
            nn->db[k][j] = 0.0;

            // update weights
            for(int i=1; i < nn->n_neurons_per_layer[k]+1; i++){
                double grad_w = nn->dw[k][i][j] / (double)batch_size;
                
                if(strcmp(opt, "sgd") == 0) {
                    nn->w[k][i][j] -= lr * grad_w;
                } 
                else if(strcmp(opt, "adam") == 0) {
                    nn->momentum_w[k][i][j] = beta_1 * nn->momentum_w[k][i][j] + (1.0-beta_1) * grad_w;
                    nn->momentum2_w[k][i][j] = beta_2 * nn->momentum2_w[k][i][j] + (1.0-beta_2) * grad_w * grad_w;
                    
                    double m_cap = nn->momentum_w[k][i][j] / (1.0 - pow(beta_1, itr));
                    double v_cap = nn->momentum2_w[k][i][j] / (1.0 - pow(beta_2, itr));
                    
                    nn->w[k][i][j] -= (lr * m_cap) / (sqrt(v_cap) + epsilon);
                }
                
                // zero weights for next batch
                nn->dw[k][i][j] = 0.0;
            }
        }
    }
}



/**
 * Init process private zones
 */
void init_private_workspace(struct NeuralNet* nn) {
    size_t priv_size = calculate_private_workspace_size(nn->n_layers, nn->n_neurons_per_layer);
    void* local_block = mmap(NULL, priv_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memset(local_block, 0, priv_size);
    char* ptr = (char*)local_block;

    // --- A. COSTRUIAMO GLI ARRAY DI PUNTATORI PRIVATI ---
    // Iniziamo con i puntatori di secondo livello (quelli che tengono le righe)
    for (int i = 0; i < nn->n_layers - 1; i++) {
        int n_in = nn->n_neurons_per_layer[i] + 1;
        
        nn->db[i] = (double*)ptr; // Questo è un puntatore a vettore (già pronto per i dati)
        ptr = (char*)ALIGN_BLOCK(ptr + (nn->n_neurons_per_layer[i+1]+1) * sizeof(double)); // Alloco i dati db qui direttamente

        nn->dw[i] = (double**)ptr; // Array di puntatori alle righe
        ptr = (char*)ALIGN_BLOCK(ptr + n_in * sizeof(double*));
        
        nn->momentum_w[i] = (double**)ptr; 
        ptr = (char*)ALIGN_BLOCK(ptr + n_in * sizeof(double*));

        nn->momentum2_w[i] = (double**)ptr; 
        ptr = (char*)ALIGN_BLOCK(ptr + n_in * sizeof(double*));
    }

    // --- B. COSTRUIAMO LE ATTIVAZIONI ---
    for (int i = 0; i < nn->n_layers; i++) {
        int n = nn->n_neurons_per_layer[i] + 1;
        nn->in[i] = (double*)ptr; ptr = (char*)ALIGN_BLOCK(ptr + n * sizeof(double));
        nn->out[i] = (double*)ptr; ptr = (char*)ALIGN_BLOCK(ptr + n * sizeof(double));
        nn->delta[i] = (double*)ptr; ptr = (char*)ALIGN_BLOCK(ptr + n * sizeof(double));
    }
    
    // --- C. RIEMPIAMO LE RIGHE DI DW E MOMENTUM ---
    for (int i = 0; i < nn->n_layers - 1; i++) {
        int n_in = nn->n_neurons_per_layer[i] + 1;
        int n_out = nn->n_neurons_per_layer[i+1] + 1;

        // Bias momentum (dati)
        nn->momentum_b[i] = (double*)ptr; ptr = (char*)ALIGN_BLOCK(ptr + n_out * sizeof(double));
        nn->momentum2_b[i] = (double*)ptr; ptr = (char*)ALIGN_BLOCK(ptr + n_out * sizeof(double));

        for (int j = 0; j < n_in; j++) {
            nn->dw[i][j] = (double*)ptr; ptr = (char*)ALIGN_BLOCK(ptr + n_out * sizeof(double));
            nn->momentum_w[i][j] = (double*)ptr; ptr = (char*)ALIGN_BLOCK(ptr + n_out * sizeof(double));
            nn->momentum2_w[i][j] = (double*)ptr; ptr = (char*)ALIGN_BLOCK(ptr + n_out * sizeof(double));
        }
    }
    
    nn->targets = (double*)ptr;
    nn->private_zone_start = local_block;
}



double* new_model_train(struct NeuralNet* nn, double** X_train, double** y_train, double* y_train_temp, 
                    char* activation_fun, char* loss, char* opt, double learning_rate,
                    int num_samples_to_train, int itr, int batch_size) {

    // 1. Setup indici (Shuffle)
    int arr[N_SAMPLES];
    for(int i = 0; i < N_SAMPLES; i++) arr[i] = i;
    shuffle(arr, N_SAMPLES);

    int correct = 0;
    double loss_val = 0.0;
    int last = nn->n_layers - 1;

    // --- CICLO MINI-BATCH ---
    for(int i = 0; i < num_samples_to_train; i += batch_size) {
        
        // Calcola la dimensione del batch corrente (gestisce l'avanzo finale)
        int cur_batch = (i + batch_size > num_samples_to_train) ? 
                        (num_samples_to_train - i) : batch_size;


        for(int b = 0; b < cur_batch; b++) {
            int idx = arr[i + b];

            // Caricamento Input e Targets
            for(int j = 1; j <= nn->n_neurons_per_layer[0]; j++)
                nn->out[0][j] = X_train[idx][j-1];
            
            for(int j = 1; j <= nn->n_neurons_per_layer[last]; j++)
                nn->targets[j] = y_train[idx][j-1];

            // Forward
            forward_propagation(nn, activation_fun, loss);

            // Accumulo metriche (Loss e Accuratezza)
            for(int j = 1; j <= nn->n_neurons_per_layer[last]; j++) {
                if(strcmp(loss, "mse") == 0) 
                    loss_val += 0.5 * pow(nn->out[last][j] - nn->targets[j], 2);
                else 
                    loss_val -= nn->targets[j] * log(nn->out[last][j] + 1e-15);
            }

            double max_v = -1.0; int pred = -1;
            for(int j = 1; j <= nn->n_neurons_per_layer[last]; j++) {
                if(nn->out[last][j] > max_v) { 
                    max_v = nn->out[last][j]; 
                    pred = j - 1; 
                }
            }
            if(pred == (int)y_train_temp[idx]) correct++;

            // Backprop: accumula i gradienti internamente alla struct (zona privata)
            back_propagation_accumulate(nn, activation_fun, loss);
        }

        // Fine Batch: Applica i gradienti accumulati ai pesi reali (zona Shared)
        // Sostituisce update_master_weights() usando direttamente la struct nn
        update_weights_batch(nn, learning_rate, opt, itr, cur_batch);
    }

    // Calcolo medie finali
    loss_val /= (double)num_samples_to_train;
    double accuracy = (double)correct / (double)num_samples_to_train;
    
    static double metrics[2];
    metrics[0] = loss_val;
    metrics[1] = accuracy;
    return metrics;
}


/**
 * Function to pin current process to a core belonging to a certain numa node
 * @param rank    Index of process
 * @param target_node NUMA node on which the process runs on
 */
void pin_process_to_node(int rank, int target_node) {

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    struct bitmask *cpus_in_node = numa_allocate_cpumask();
    if (numa_node_to_cpus(target_node, cpus_in_node) != 0) {
        perror("Error in numa_node_to_cpus");
        numa_free_cpumask(cpus_in_node);
        return;
    }

    int num_cpus = numa_num_possible_cpus();
    int num_numa_nodes = numa_num_configured_nodes();
    int core_in_node_count = 0;
    
    // count cores associated to this node
    for (int i = 0; i < num_cpus; i++) {
        if (numa_bitmask_isbitset(cpus_in_node, i)) {
            core_in_node_count++;
        }
    }

    if (core_in_node_count == 0) {
        fprintf(stderr, "[NUMA] Error: Node %d has no associated CPUs.\n", target_node);
        numa_free_cpumask(cpus_in_node);
        return;
    }

    // choose core based on rank
    int target_core_index = (rank / num_numa_nodes) % core_in_node_count;
    int current_match = 0;
    int core_found = -1;

    for (int i = 0; i < num_cpus; i++) {
        if (numa_bitmask_isbitset(cpus_in_node, i)) {
            if (current_match == target_core_index) {
                core_found = i;
                break;
            }
            current_match++;
        }
    }

    // set affinity
    if (core_found != -1) {
        CPU_SET(core_found, &cpuset);
        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
            perror("sched_setaffinity failed");
        } else {
            printf("[RANK %d] Pinning completed: CORE %d on NODO %d\n", rank, core_found, target_node);
        }
    }

    numa_free_cpumask(cpus_in_node);
}


/** this is the function that will wrap processes fork, 
 *  training set partitioning,
 *  private memory management
 *  and actual training with forward/back propagation
 * */
double* parallel_training(struct NeuralNet* base_nn, double** X_train, double** y_train, double* y_train_temp, char* activation_fun, char* loss, char* opt, double learning_rate, int total_samples, int itr, int batch_size, int n_workers) {

    printf("[DEBUG] NN: %p, X_train[0]: %p, y_train[0]: %p\n", (void*)base_nn, (void*)X_train[0], (void*)y_train[0]);
    // 1. Memoria condivisa per raccogliere Loss e Accuracy da ogni processo
    double* shared_metrics = mmap(NULL, n_workers * 2 * sizeof(double), 
                                  PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);


    for (int rank = 0; rank < n_workers; rank++) {

        pid_t pid = fork();
        if (pid < 0) {
            perror("fork failed");
            exit(EXIT_FAILURE);
        }

        if (pid == 0) {
            
            // pinning
            int my_node = rank % num_numa_nodes;
            // set affinity
            pin_process_to_node(rank, my_node);

            // pte switch on my_node memory view e.g. (base_addres, 0, my_node)
            
            
            // allocate private memory
            struct NeuralNet local_nn = *base_nn; //shadow memory zone
            init_private_workspace(&local_nn);

            printf("[DEBUG] neural network LOCAL ptr = %p\n", (void*)local_nn);

            // training set partitioning
            int samples_per_worker = total_samples / n_workers;
            int start = rank * samples_per_worker;
            // remainder to last worker
            int count = (rank == n_workers - 1) ? (total_samples - start) : samples_per_worker;

            // model training
            double* results = new_model_train(&local_nn, X_train, y_train, y_train_temp, 
                                                        activation_fun, loss, opt, learning_rate, 
                                                        count, itr, batch_size);

            // E. Scrittura risultati e uscita
            shared_metrics[rank * 2] = results[0];
            shared_metrics[rank * 2 + 1] = results[1];
            exit(0);
        }
    }

    // --- PROCESSO PADRE ---
    // Attende che tutti i worker finiscano
    for (int i = 0; i < n_workers; i++) {
        wait(NULL);
    }

    double* final_metrics = malloc(2 * sizeof(double));
    final_metrics[0] = 0; final_metrics[1] = 0;

    for (int i = 0; i < n_workers; i++) {
        final_metrics[0] += shared_metrics[i * 2];     // Somma Loss
        final_metrics[1] += shared_metrics[i * 2 + 1]; // Somma Acc
    }
    final_metrics[0] /= n_workers; // Media
    final_metrics[1] /= n_workers; // Media

    return final_metrics;


}
#endif


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

