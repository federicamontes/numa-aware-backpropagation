#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include <numaif.h>
#include <numa.h>
#include <sys/sysinfo.h>
#include <sys/mman.h>
#include <sched.h>       // Per cpu_set_t, CPU_ZERO, sched_setaffinity
#include <unistd.h>     // Per fork()
#include <sys/types.h>  // Per pid_t
#include <sys/wait.h>   // Per wait()

#include "neural_network.h"
#include "activation.h"

// In cima a numa_api.c o dentro neural_network.h
extern double* new_model_train(struct NNWorkerWorkspace* nn, double** X_train, double** y_train, double* y_train_temp, 
                               char* activation_fun, char* loss, char* opt, double learning_rate,
                               int num_samples_to_train, int itr, int batch_size);


/** these are the functions wrapped
 * parallel_training is the one creating the processes and applying parallel training
 * setup_numa_model is the one allocating memory in a compliant way for our LKM technique */
double* parallel_training(struct NeuralNet* base_nn, double** X_train, double** y_train, double* y_train_temp, char* activation_fun, char* loss, char* opt, double learning_rate, int total_samples, int itr, int batch_size, int n_workers);
NeuralNet* setup_numa_model(int n_layers, int n_neurons_per_layer[]);

// --- Memory & Alignment Settings ---
#define PAGE_SIZE 4096
// alignment for huge pages 2MB
#define PAGE_ALIGNMENT 2097152

#define PDE_ALIGN_SIZE (2 * 1024 * 1024)      // 2MB Alignment for PDE Switching

#define ALIGN_BLOCK(ptr) (char*)(((uintptr_t)(ptr) + 7) & ~7)
#define ALIGN_PAGE(sz) ({            \
    size_t __s = (sz);               \
    (__s + PAGE_ALIGNMENT - 1) & ~(PAGE_ALIGNMENT - 1); \
})



// Define global base address
unsigned long base_address = 0x1000000000; // Aligned to 2MB
static void* global_mmap_start = NULL;
static size_t global_mmap_total_size = 0;
int num_numa_nodes = 1;


#define DISPLACEMENT (1<<21)
#define SET_MEMORY(addr, value) \
    do { \
        *(addr) = (value); \
    for (int _i = 1; _i < (num_numa_nodes); ++_i){ \
            *((typeof(addr))((char*)(addr) + _i * (DISPLACEMENT))) = *(addr); \
    } \
    } while(0)

#define UPDATE_REPLICAS_GLOBAL(local_addr, delta) \
    do { \
        uintptr_t offset = (uintptr_t)(local_addr) - (uintptr_t)base_address; \
        /* 2. Applica a tutti i nodi partendo dal base_address reale */ \
        for (int _n = 0; _n < num_numa_nodes; ++_n) { \
            double* target = (double*)((char*)base_address + (_n * DISPLACEMENT) + offset); \
            *target -= (delta); \
        } \
    } while(0)

#define PES 156 //this depends on what the kernel tells you when mounting the vtpmo module

int pes(unsigned long x, int a, int b){
    printf("CALL SYSCALL\n");
    return syscall(PES,x,a,b);
}



// -------- Utilities for size computation --------


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

    size = ALIGN_SIZE(sizeof(NNWorkerWorkspace));

    // TOP-LEVEL POINTERS 
    // dw, momentum_w, momentum2_w -> 3 arrays of (n_layers-1) ptr double**
    size += (size_t)(n_layers - 1) * 3 * sizeof(double**); 
    // db, momentum_b, momentum2_b -> 3 arrays of (n_layers-1) ptr double*
    size += (size_t)(n_layers - 1) * 3 * sizeof(double*);
    // delta, in, out -> 3 arrays of (n_layers) ptr double*
    size += (size_t)n_layers * 3 * sizeof(double*); 
    size = ALIGN_SIZE(size);

    // ROW POINTERS 
    for (int i = 0; i < n_layers - 1; i++) {
        size_t n_in = (size_t)(n_neurons_per_layer[i] + 1);
        // Ogni riga di dw, momentum_w e momentum2_w ha bisogno di un puntatore
        size += n_in * 3 * sizeof(double*); 
    }
    size = ALIGN_SIZE(size);

    // REAL DATA (activation, delta and targets)
    for (int i = 0; i < n_layers; i++) {
        size_t n = (size_t)(n_neurons_per_layer[i] + 1);
        size += n * 3 * sizeof(double);
        size = ALIGN_SIZE(size);
    }
    // Target
    size += (size_t)(n_neurons_per_layer[n_layers-1] + 1) * sizeof(double);
    size = ALIGN_SIZE(size);

    // Gradients and Momentum
    for (int i = 0; i < n_layers - 1; i++) {
        size_t n_in  = (size_t)(n_neurons_per_layer[i] + 1);
        size_t n_out = (size_t)(n_neurons_per_layer[i+1] + 1);
        
        // Bias: db, momentum_b, momentum2_b
        size += n_out * 3 * sizeof(double);
        size = ALIGN_SIZE(size);

        // Weights: dw, momentum_w, momentum2_w
        size += (n_in * n_out) * 3 * sizeof(double);
        size = ALIGN_SIZE(size);
    }
    return size;
}


/**
 * This functions only computes the amount of memory needed for the shared part of the network
 * */
size_t calculate_shared_model_size(int n_layers, const int n_neurons_per_layer[]) {
    size_t size = 0;
    
    // 1. Struttura base + array architettura
    size += sizeof(struct NeuralNet);
    size = ALIGN_SIZE(size);
    size += (size_t)n_layers * sizeof(int);
    size = ALIGN_SIZE(size);
    
    // 2. Puntatori top-level (w, b, in, out)
    size += (size_t)(n_layers - 1) * sizeof(double**); // w
    size += (size_t)(n_layers - 1) * sizeof(double*);  // b
    size += (size_t)n_layers * sizeof(double*);        // in
    size += (size_t)n_layers * sizeof(double*);        // out
    size = ALIGN_SIZE(size);

    // 3. Puntatori di riga per i pesi (w[i][j])
    for (int i = 0; i < n_layers - 1; i++) {
        size_t n_in = (size_t)(n_neurons_per_layer[i] + 1);
        size += n_in * sizeof(double*); 
    }
    size = ALIGN_SIZE(size);

    // 4. Dati Reali: Pesi (w) e Bias (b)
    for (int i = 0; i < n_layers - 1; i++) {
        size_t n_in  = (size_t)(n_neurons_per_layer[i] + 1);
        size_t n_out = (size_t)(n_neurons_per_layer[i+1] + 1);

        size += n_in * sizeof(double);             // Bias b[i]
        size += n_in * n_out * sizeof(double);     // Pesi w[i]
    }
    size = ALIGN_SIZE(size);

    // 5. Dati Reali: Attivazioni (in, out)
    for (int i = 0; i < n_layers; i++) {
        size_t n = (size_t)(n_neurons_per_layer[i] + 1);
        size += n * sizeof(double); // in[i]
        size += n * sizeof(double); // out[i]
    }
    size = ALIGN_SIZE(size);

    // 6. Dati Reali: Targets
    size += (size_t)(n_neurons_per_layer[n_layers-1] + 1) * sizeof(double);
    size = ALIGN_SIZE(size);

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
    
    size_t aligned_size = (size + PDE_ALIGN_SIZE - 1) & ~(PDE_ALIGN_SIZE - 1);
    
    void* addr = (void*)base_address;

    if ((uintptr_t)addr % PDE_ALIGN_SIZE != 0) {
        fprintf(stderr, "FATAL: base_address non allineato a 2MB!\n");
    }

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
 * This function allocates the shared neural network (weights and bias + metadata)
 * */
NeuralNet* newNetSharedAlloc(int n_layers, int n_neurons_per_layer[], size_t aligned_shared_size) {
    size_t total_numa_map_size = (size_t)num_numa_nodes * aligned_shared_size;
    void* mmap_block = mmap_alloc(total_numa_map_size); 

    global_mmap_start = mmap_block;
    global_mmap_total_size = total_numa_map_size;

    char* virtual_base = (char*)mmap_block;

    for (int numa_node = 0; numa_node < num_numa_nodes; numa_node++) {
        char * net_addr = (char *)mmap_block + (numa_node * aligned_shared_size);
        bind_memory_to_numa_node(net_addr, aligned_shared_size, numa_node);

        // Touch memoria
        for (size_t i = 0; i < aligned_shared_size; i += 4096) {
            *((volatile char*)(net_addr + i)) = 0; 
        }
        
        struct NeuralNet* nn = (struct NeuralNet*)net_addr;
        char * curr = net_addr; 

        // Helper per ottenere l'indirizzo virtuale corrispondente a curr
        // (v_base + distanza dall'inizio della slab attuale)
        #define V_ADDR (virtual_base + (curr - net_addr))

        // --- Metadata ---
        nn->magic_test_value = (numa_node == 0) ? 123.45 : 678.90;
        nn->n_layers = n_layers;
        curr += sizeof(struct NeuralNet);

        // --- Architettura ---
        curr = (char*)ALIGN_BLOCK(curr);
        nn->n_neurons_per_layer = (int*)V_ADDR; 
        for (int i = 0; i < n_layers; i++) 
            ((int*)curr)[i] = n_neurons_per_layer[i];
        curr += (n_layers * sizeof(int));

        // --- Puntatori Top-Level ---
        curr = (char*)ALIGN_BLOCK(curr);
        nn->w = (double***)V_ADDR;   curr += (n_layers - 1) * sizeof(double**);
        nn->b = (double**)V_ADDR;    curr += (n_layers - 1) * sizeof(double*);
        nn->in = (double**)V_ADDR;   curr += n_layers * sizeof(double*);
        nn->out = (double**)V_ADDR;  curr += n_layers * sizeof(double*);

        // --- Tabelle Puntatori di Riga (W) ---
        curr = (char*)ALIGN_BLOCK(curr);
        for (int i = 0; i < n_layers - 1; i++) {
            int n_in = n_neurons_per_layer[i] + 1;
            // Scriviamo nella tabella fisica dei puntatori (nn->w è virtuale, non usiamolo!)
            // Troviamo la posizione fisica di nn->w[i]
            double*** w_top_phys = (double***)(net_addr + ((char*)nn->w - virtual_base));
            w_top_phys[i] = (double**)V_ADDR; 
            curr += n_in * sizeof(double*);
        }

        // --- Dati Reali: Bias e Pesi ---
        curr = (char*)ALIGN_BLOCK(curr);
        for (int i = 0; i < n_layers - 1; i++) {
            int n_in = n_neurons_per_layer[i] + 1;
            int n_out = n_neurons_per_layer[i+1] + 1;

            // Bias
            double** b_top_phys = (double**)(net_addr + ((char*)nn->b - virtual_base));
            b_top_phys[i] = (double*)V_ADDR;
            curr += n_in * sizeof(double);
            curr = (char*)ALIGN_BLOCK(curr);

            // Pesi (righe)
            double*** w_top_phys = (double***)(net_addr + ((char*)nn->w - virtual_base));
            double** row_table_phys = (double**)(net_addr + ((char*)w_top_phys[i] - virtual_base));
            for (int j = 0; j < n_in; j++) {
                row_table_phys[j] = (double*)V_ADDR;
                curr += n_out * sizeof(double);
            }
            curr = (char*)ALIGN_BLOCK(curr);
        }

        // --- Dati Reali: In, Out, Targets ---
        for (int i = 0; i < n_layers; i++) {
            int n = n_neurons_per_layer[i] + 1;
            
            double** in_top_phys = (double**)(net_addr + ((char*)nn->in - virtual_base));
            in_top_phys[i] = (double*)V_ADDR;
            curr += n * sizeof(double);
            curr = (char*)ALIGN_BLOCK(curr);

            double** out_top_phys = (double**)(net_addr + ((char*)nn->out - virtual_base));
            out_top_phys[i] = (double*)V_ADDR;
            curr += n * sizeof(double);
            curr = (char*)ALIGN_BLOCK(curr);
        }

        nn->targets = (double*)V_ADDR;
        // (Target space opzionale qui se serve)

        #undef V_ADDR
    }
    return (struct NeuralNet*)mmap_block;
}

void unmap_nn() {
    if (global_mmap_start != NULL) {
        if (munmap(global_mmap_start, global_mmap_total_size) < 0) {
            perror("munmap failed");
        } else {
            printf("INFO: Shared memory unmapped successfully.\n");
            global_mmap_start = NULL;
        }
    }
}


void init_nn_shared(struct NeuralNet* nn) {
    // init bias and weights
    for (int k = 0; k < nn->n_layers - 1; k++) {
        
        // Number of neurons + 1
        int n_in = nn->n_neurons_per_layer[k] + 1;
        int n_out = nn->n_neurons_per_layer[k+1] + 1;

        for (int i = 0; i < n_in; i++) {
            // int bias from 0
            nn->b[k][i] = 0.0;
            
            for (int j = 0; j < n_out; j++) {
                // init weights matrix with randn()
                nn->w[k][i][j] = randn();
            }
        }
    }

    for (int k = 0; k < nn->n_layers; k++) {
        int n = nn->n_neurons_per_layer[k] + 1;
        for (int i = 0; i < n; i++) {
            nn->in[k][i] = 0.0;
            nn->out[k][i] = 0.0;
        }
    }

    // Initialize targets
    int last_layer_size = nn->n_neurons_per_layer[nn->n_layers - 1] + 1;
    for (int i = 0; i < last_layer_size; i++) {
        nn->targets[i] = 0.0;
    }
}


// -------- Wrapped by pragma for allocation of new neural network --------

/**
 * Setup function to allocate and initialize a neural network 
 * this must be called by the annotation pragma
 */
NeuralNet* setup_numa_model(int n_layers, int n_neurons_per_layer[]) {

    size_t raw_shared_size = calculate_shared_model_size(n_layers, n_neurons_per_layer);
    
    size_t aligned_size = ALIGN_PAGE(raw_shared_size);

    // allocation
    NeuralNet* nn_mapped = newNetSharedAlloc(n_layers, n_neurons_per_layer, aligned_size);

    // init neural network for each numa node
    for (int i = 0; i < num_numa_nodes; i++) {
        // Calcolo l'offset per la i-esima copia (Slab)
        NeuralNet *nn = (NeuralNet*)((char*)nn_mapped + i * aligned_size);
        
        if (nn != NULL) {
            nn->magic_test_value = (i == 0) ? 123.45 : 678.90; // Per testare il PTE switcher
            nn->n_layers = n_layers;

            srand(42); //set seed to avoid each numa node having different weights

            // Prima di chiamare init_nn_shared, verifica i puntatori
            if (nn->out == NULL || ((void**)nn->out)[0] == NULL) {
                 printf("[ERROR] Puntatori già corrotti prima di init_nn per Nodo %d\n", i);
            }
            // init neural network
            init_nn_shared(nn); 

            if (nn->out[0] == NULL) {
                printf("[ERROR] init_nn_shared ha azzerato i puntatori del Nodo %d!\n", i);
            }

            printf("[GLOBAL SETUP] Node %d Slab at: %p | Weights[0] at: %p\n", 
                    i, (void*)nn, (void*)nn->w);
        } else {
            fprintf(stderr, "Error allocating network copy %d\n", i);
            exit(EXIT_FAILURE);
        }
    }

    return nn_mapped;
}



/* ------------- BELOW the new model train in parallel setting */


void back_propagation_accumulate(NNWorkerWorkspace* ws, char* activation_fun, char* loss) {
    // local variable for reading shared nn
    NeuralNet* nn = ws->shared_nn; 
    
    int last = nn->n_layers - 1;

    /* error at last layer */
    for (int i = 1; i < nn->n_neurons_per_layer[last] + 1; i++) {
        if (strcmp(loss, "mse") == 0) {
            ws->delta[last][i] = (ws->out[last][i] - ws->targets[i]) * sigmoid_d(ws->out[last][i]);
        } else {
            ws->delta[last][i] = ws->out[last][i] - ws->targets[i];
        }
    }

    /* backpropagation */
    for (int k = nn->n_layers - 2; k > 0; k--) {
        for (int i = 1; i < nn->n_neurons_per_layer[k] + 1; i++) {
            double sum = 0.0;
            for (int j = 1; j < nn->n_neurons_per_layer[k + 1] + 1; j++) {
                // READING shared weights
                sum += nn->w[k][i][j] * ws->delta[k + 1][j];
            }

            double grad;
            if (strcmp(activation_fun, "tanh") == 0)      grad = tanh_d(ws->out[k][i]);
            else if (strcmp(activation_fun, "relu") == 0) grad = relu_d(ws->out[k][i]);
            else                                          grad = sigmoid_d(ws->out[k][i]);

            ws->delta[k][i] = grad * sum;
        }
    }

    /* accumulate local gradients */
    for (int k = 0; k < nn->n_layers - 1; k++) {
        for (int j = 1; j < nn->n_neurons_per_layer[k + 1] + 1; j++) {
            ws->db[k][j] += ws->delta[k + 1][j];
            for (int i = 1; i < nn->n_neurons_per_layer[k] + 1; i++) {
                ws->dw[k][i][j] += ws->delta[k + 1][j] * ws->out[k][i];
            }
        }
    }
}

/**
 * Update weights utilizzando i gradienti accumulati nel Workspace
 * e applicandoli al modello condiviso (shared_nn).
 */
void update_weights_batch(NNWorkerWorkspace* ws, double lr, char* opt, int itr, int batch_size) {
    // Puntatore alla rete condivisa (dove risiedono i pesi reali)
    NeuralNet* nn = ws->shared_nn;
    
    for(int k=0; k < nn->n_layers-1; k++){
        for(int j=1; j < nn->n_neurons_per_layer[k+1]+1; j++){
            
            // 1. Update BIAS sul modello condiviso
            // Usiamo il gradiente locale (ws->db) ma modifichiamo il bias globale (nn->b)
            double grad_b = ws->db[k][j] / (double)batch_size;
            
            double delta_b = 0.0;

            if(strcmp(opt, "sgd") == 0) {
                delta_b = lr * grad_b;
                //nn->b[k][j] -= lr * grad_b;
            } 
            else if(strcmp(opt, "adam") == 0) {
                // I momenti di Adam devono essere salvati nel Workspace (sono locali al processo/nodo)
                ws->momentum_b[k][j] = beta_1 * ws->momentum_b[k][j] + (1.0-beta_1) * grad_b;
                ws->momentum2_b[k][j] = beta_2 * ws->momentum2_b[k][j] + (1.0-beta_2) * grad_b * grad_b;
                
                double m_cap = ws->momentum_b[k][j] / (1.0 - pow(beta_1, itr));
                double v_cap = ws->momentum2_b[k][j] / (1.0 - pow(beta_2, itr));
                
                delta_b = (lr * m_cap) / (sqrt(v_cap) + epsilon);
                //nn->b[k][j] -= (lr * m_cap) / (sqrt(v_cap) + epsilon);
            }
            
            // instead of nn->b[k][j] -= delta_b
            // apply delta to all replicas (subtraction is inside macro)
            UPDATE_REPLICAS_GLOBAL(&(nn->b[k][j]), delta_b);

            // Azzero il gradiente locale nel Workspace per il prossimo batch
            ws->db[k][j] = 0.0;

            // 2. Update WEIGHTS sul modello condiviso
            for(int i=1; i < nn->n_neurons_per_layer[k]+1; i++){
                double grad_w = ws->dw[k][i][j] / (double)batch_size;

                double delta_w = 0.0;
                
                if(strcmp(opt, "sgd") == 0) {
                    delta_w = lr * grad_w;
                    //nn->w[k][i][j] -= lr * grad_w;
                } 
                else if(strcmp(opt, "adam") == 0) {
                    ws->momentum_w[k][i][j] = beta_1 * ws->momentum_w[k][i][j] + (1.0-beta_1) * grad_w;
                    ws->momentum2_w[k][i][j] = beta_2 * ws->momentum2_w[k][i][j] + (1.0-beta_2) * grad_w * grad_w;
                    
                    double m_cap = ws->momentum_w[k][i][j] / (1.0 - pow(beta_1, itr));
                    double v_cap = ws->momentum2_w[k][i][j] / (1.0 - pow(beta_2, itr));
                    
                    delta_w = (lr * m_cap) / (sqrt(v_cap) + epsilon);
                    //nn->w[k][i][j] -= (lr * m_cap) / (sqrt(v_cap) + epsilon);
                }
                
                UPDATE_REPLICAS_GLOBAL(&(nn->w[k][i][j]), delta_w);
                // Azzero il gradiente locale nel Workspace
                ws->dw[k][i][j] = 0.0;
            }
        }
    }
}


/**
 * This does the setup of the process' workspace neural network
 * */
NNWorkerWorkspace* setup_worker_workspace(NeuralNet* shared_nn, int rank, int node_id) {
    int n_layers = shared_nn->n_layers;
    int* n_neurons = shared_nn->n_neurons_per_layer;

    // Compute total space needed
    size_t priv_size = calculate_private_workspace_size(n_layers, n_neurons);
    
    // map memory
    void* local_block = mmap(NULL, priv_size, PROT_READ | PROT_WRITE, 
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    if (local_block == MAP_FAILED) {
        perror("mmap private workspace");
        exit(EXIT_FAILURE);
    }

    // init struct
    NNWorkerWorkspace* ws = (NNWorkerWorkspace*)local_block;
    char* ptr = (char*)local_block;

    ws->shared_nn = shared_nn;
    ws->rank = rank;
    ws->local_numa_node = node_id;
    ws->private_zone_start = local_block;
    ws->private_zone_size = priv_size;

    ptr += ALIGN_SIZE(sizeof(NNWorkerWorkspace));

    // TOP-LEVEL POINTERS
    ws->dw = (double***)ptr;          ptr += (n_layers - 1) * sizeof(double**);
    ws->momentum_w = (double***)ptr;  ptr += (n_layers - 1) * sizeof(double**);
    ws->momentum2_w = (double***)ptr; ptr += (n_layers - 1) * sizeof(double**);
    
    ws->db = (double**)ptr;           ptr += (n_layers - 1) * sizeof(double*);
    ws->momentum_b = (double**)ptr;   ptr += (n_layers - 1) * sizeof(double*);
    ws->momentum2_b = (double**)ptr;  ptr += (n_layers - 1) * sizeof(double*);

    ws->in = (double**)ptr;           ptr += n_layers * sizeof(double*);
    ws->out = (double**)ptr;          ptr += n_layers * sizeof(double*);
    ws->delta = (double**)ptr;        ptr += n_layers * sizeof(double*);
    
    ptr = ALIGN_BLOCK(ptr);

    // ROW POINTERS
    for (int i = 0; i < n_layers - 1; i++) {
        size_t n_in = (size_t)(n_neurons[i] + 1);
        
        ws->dw[i] = (double**)ptr;          ptr += n_in * sizeof(double*);
        ws->momentum_w[i] = (double**)ptr;  ptr += n_in * sizeof(double*);
        ws->momentum2_w[i] = (double**)ptr; ptr += n_in * sizeof(double*);
    }
    ptr = ALIGN_BLOCK(ptr);

    // REAL DATA (Gradients, Momentum, Activation)
    for (int i = 0; i < n_layers - 1; i++) {
        int n_in = n_neurons[i] + 1;
        int n_out = n_neurons[i+1] + 1;

        // Bias (db, mb, m2b)
        ws->db[i] = (double*)ptr;          ptr += ALIGN_SIZE(n_out * sizeof(double));
        ws->momentum_b[i] = (double*)ptr;  ptr += ALIGN_SIZE(n_out * sizeof(double));
        ws->momentum2_b[i] = (double*)ptr; ptr += ALIGN_SIZE(n_out * sizeof(double));

        // Pesi (dw, mw, m2w)
        for (int j = 0; j < n_in; j++) {
            ws->dw[i][j] = (double*)ptr;          ptr += n_out * sizeof(double);
            ws->momentum_w[i][j] = (double*)ptr;  ptr += n_out * sizeof(double);
            ws->momentum2_w[i][j] = (double*)ptr; ptr += n_out * sizeof(double);
        }
        ptr = ALIGN_BLOCK(ptr);
    }

    // ACTIVATIONS, DELTA and TARGET 
    for (int i = 0; i < n_layers; i++) {
        int n = n_neurons[i] + 1;
        ws->in[i] = (double*)ptr;    ptr += ALIGN_SIZE(n * sizeof(double));
        ws->out[i] = (double*)ptr;   ptr += ALIGN_SIZE(n * sizeof(double));
        ws->delta[i] = (double*)ptr; ptr += ALIGN_SIZE(n * sizeof(double));
    }
    
    ws->targets = (double*)ptr; 

    return ws;
}


void forward_propagation_numa(NNWorkerWorkspace* ws, char* activation_fun, char* loss) {
    // Puntatore di comodo alla parte condivisa (Pesi e Bias)
    NeuralNet* shared = ws->shared_nn;

    // 1. Cleanup degli input (in) nel Workspace locale
    for(int i = 0; i < shared->n_layers; i++) {
        for(int j = 0; j < shared->n_neurons_per_layer[i] + 1; j++) {
            ws->in[i][j] = 0.0;
        }
    }

    // 2. Calcolo per ogni layer (dal primo nascosto in poi)
    for(int k = 1; k < shared->n_layers; k++) {

        /* Calcolo della somma pesata (z = W*a + b) */
        
        // Aggiunta dei Bias (presi dalla rete shared)
        for(int j = 1; j < shared->n_neurons_per_layer[k] + 1; j++) {
            ws->in[k][j] = shared->b[k-1][j]; // b[layer-1] collega layer-1 a layer
        }

        // Aggiunta del prodotto Pesi * Output layer precedente
        for(int i = 1; i < shared->n_neurons_per_layer[k-1] + 1; i++) {
            for(int j = 1; j < shared->n_neurons_per_layer[k] + 1; j++) {
                // Pesi presi da shared->w, output precedenti presi dal workspace locale
                ws->in[k][j] += ws->out[k-1][i] * shared->w[k-1][i][j];
            }
        }

        /* Applicazione funzione di attivazione */

        // Caso Ultimo Layer (Output)
        if(k == shared->n_layers - 1) {
            if(strcmp(loss, "mse") == 0) {
                for(int j = 1; j < shared->n_neurons_per_layer[k] + 1; j++) {
                    ws->out[k][j] = sigmoid(ws->in[k][j]);
                }
            }
            else if(strcmp(loss, "ce") == 0) {
                // Softmax con protezione numerica (LogSumExp trick semplificato)
                double max_in = -INFINITY;
                for(int j = 1; j < shared->n_neurons_per_layer[k] + 1; j++) {
                    if(ws->in[k][j] > max_in) max_in = ws->in[k][j];
                }

                double deno = 0.0;
                for(int j = 1; j < shared->n_neurons_per_layer[k] + 1; j++) {
                    // Sottraiamo il max per evitare overflow di exp()
                    deno += exp(ws->in[k][j] - max_in);
                }
                for(int j = 1; j < shared->n_neurons_per_layer[k] + 1; j++) {
                    ws->out[k][j] = exp(ws->in[k][j] - max_in) / deno;
                }
            }
        } 
        // Caso Hidden Layers
        else {
            for(int j = 1; j < shared->n_neurons_per_layer[k] + 1; j++) {
                if(strcmp(activation_fun, "sigmoid") == 0) {
                    ws->out[k][j] = sigmoid(ws->in[k][j]);
                }
                else if(strcmp(activation_fun, "tanh") == 0) {
                    ws->out[k][j] = tanh(ws->in[k][j]);
                }
                else if(strcmp(activation_fun, "relu") == 0) {
                    ws->out[k][j] = relu(ws->in[k][j]);
                }
                else {
                    ws->out[k][j] = sigmoid(ws->in[k][j]);
                }
            }
        }
    }
}



double* new_model_train(struct NNWorkerWorkspace* ws, double** X_train, double** y_train, double* y_train_temp, 
                    char* activation_fun, char* loss, char* opt, double learning_rate,
                    int num_samples_to_train, int itr, int batch_size) {

    // Puntatore di comodo alla rete condivisa
    NeuralNet* nn_shared = ws->shared_nn;

    // 1. Setup indici (Shuffle) - Assicurati che N_SAMPLES sia definito o usa num_samples_to_train
    int* arr = malloc(num_samples_to_train * sizeof(int));
    for(int i = 0; i < num_samples_to_train; i++) arr[i] = i;
    shuffle(arr, num_samples_to_train);

    int correct = 0;
    double loss_val = 0.0;
    int last = nn_shared->n_layers - 1;

    // --- CICLO MINI-BATCH ---
    for(int i = 0; i < num_samples_to_train; i += batch_size) {
        
        // Calcola la dimensione del batch corrente
        int cur_batch = (i + batch_size > num_samples_to_train) ? 
                        (num_samples_to_train - i) : batch_size;

        for(int b = 0; b < cur_batch; b++) {
            int idx = arr[i + b];

            // Caricamento Input e Targets nel Workspace privato (ws)
            for(int j = 1; j <= nn_shared->n_neurons_per_layer[0]; j++)
                ws->out[0][j] = X_train[idx][j-1];
            
            for(int j = 1; j <= nn_shared->n_neurons_per_layer[last]; j++)
                ws->targets[j] = y_train[idx][j-1];

            // Forward: Passiamo il workspace (che internamente ha il puntatore alla rete condivisa)
            forward_propagation(ws, activation_fun, loss);

            // Accumulo metriche usando ws->out
            for(int j = 1; j <= nn_shared->n_neurons_per_layer[last]; j++) {
                if(strcmp(loss, "mse") == 0) 
                    loss_val += 0.5 * pow(ws->out[last][j] - ws->targets[j], 2);
                else 
                    loss_val -= ws->targets[j] * log(ws->out[last][j] + 1e-15);
            }

            double max_v = -1.0; int pred = -1;
            for(int j = 1; j <= nn_shared->n_neurons_per_layer[last]; j++) {
                if(ws->out[last][j] > max_v) { 
                    max_v = ws->out[last][j]; 
                    pred = j - 1; 
                }
            }
            if(pred == (int)y_train_temp[idx]) correct++;

            // Backprop: accumula i gradienti internamente al workspace
            back_propagation_accumulate(ws, activation_fun, loss);
        }

        // Fine Batch: Applica i gradienti del workspace ai pesi reali (Shared)
        update_weights_batch(ws, learning_rate, opt, itr, cur_batch);
    }

    // Calcolo medie finali
    loss_val /= (double)num_samples_to_train;
    double accuracy = (double)correct / (double)num_samples_to_train;
    
    free(arr); // Non dimenticare di liberare la memoria dello shuffle

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


    size_t total_shared_size = (size_t)num_numa_nodes * ALIGN_PAGE(calculate_shared_model_size(base_nn->n_layers, base_nn->n_neurons_per_layer));
    //touch memory to create 3rd level PTE
    volatile char* ptr = (char*)base_nn;
    for (size_t i = 0; i < total_shared_size; i += 4096) {
        ptr[i] = ptr[i]; 
    }


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

            // memory touch to create ptes in child
            volatile char* touch_ptr = (char*)base_nn;
            for (int n = 0; n < num_numa_nodes; n++) {
                char c = touch_ptr[n * PAGE_ALIGNMENT]; 
                touch_ptr[n * PAGE_ALIGNMENT] = c;
            }

            printf("Valore prima della syscall: %f\n", base_nn->magic_test_value);
            // pte switch on my_node memory view e.g. (base_addres, 0, my_node)
            long ret = pes((unsigned long) base_nn, 0, my_node);
            if (ret < 0) {
                fprintf(stderr, "CRITICAL: pes syscall failed for node %d\n", my_node);
                exit(EXIT_FAILURE);
            }
            printf("Valore dopo la syscall: %f\n", base_nn->magic_test_value);

            
            // allocate private memory
            struct NNWorkerWorkspace *local_nn = setup_worker_workspace(base_nn, rank, my_node);

            printf("[DEBUG] neural network LOCAL ptr = %p\n", local_nn);
            printf("[DEBUG PID FIGLIO %d] nn->out[0] address: %p\n", getpid(), (NNWorkerWorkspace*)local_nn->out[0]);
            // training set partitioning
            int samples_per_worker = total_samples / n_workers;
            int start = rank * samples_per_worker;
            // remainder to last worker
            int count = (rank == n_workers - 1) ? (total_samples - start) : samples_per_worker;

            // model training
            double* results = new_model_train(local_nn, X_train, y_train, y_train_temp, 
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