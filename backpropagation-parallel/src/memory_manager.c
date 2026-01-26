#include "memory_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <numaif.h>
#include "utils.h"
#include <stdint.h>

// Define global variables
unsigned long base_address = 0x1000000000; // Aligned to 2MB


size_t align_block(size_t current_size) {
    return (current_size + PTR_ALIGNMENT - 1) & ~(PTR_ALIGNMENT - 1);
}

size_t align_page(size_t current_size) {
    return (current_size + PAGE_ALIGNMENT - 1) & ~(PAGE_ALIGNMENT - 1);
}


/**
 * It computes the amount of memory required for one instance of Neural Network struct
 * with all its field. It sums up the sizes of the metadata, the pointers, 
 * and the raw double data based on the network architecture (n_layers and neurons per layer)
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


/**
 * This functions computes the amount of memory needed for transient data
 * such as delta, in, out, targets → the pointers kept by each process in its own private memory zone
 * It also computes the memory needed for local gradients (weights and bias)
 * */
size_t calculate_private_workspace_size(int n_layers, const int n_neurons_per_layer[]) {
    size_t size = 0;

    // --- 1. ACTIVATION BUFFERS (in, out, delta) ---
    // Top-level pointers (double**)
    size = align_block(size + (size_t)n_layers * 3 * sizeof(double*));
    // Vector data for each layer
    for (int i = 0; i < n_layers; i++) {
        int n = n_neurons_per_layer[i] + 1;
        size = align_block(size + (size_t)n * 3 * sizeof(double));
    }

    // --- 2. TARGET BUFFER ---
    size = align_block(size + (size_t)(n_neurons_per_layer[n_layers-1] + 1) * sizeof(double));

    // --- 3. GRADIENT STORAGE (local_grad_w, local_grad_b) ---
    // Top-level pointers (w is double***, b is double**)
    size = align_block(size + (size_t)(n_layers - 1) * sizeof(double**)); // local_grad_w
    size = align_block(size + (size_t)(n_layers - 1) * sizeof(double*));  // local_grad_b

    for (int i = 0; i < n_layers - 1; i++) {
        int n_in  = n_neurons_per_layer[i] + 1;
        int n_out = n_neurons_per_layer[i+1] + 1;

        // Intermediate row pointers for local_grad_w[i]
        size = align_block(size + (size_t)n_in * sizeof(double*)); 
        
        // Bias gradient data for local_grad_b[i]
        size = align_block(size + (size_t)n_in * sizeof(double));

        // Weight gradient data rows for local_grad_w[i][j]
        // Following your allocSharedNN logic: align after every row
        for (int j = 0; j < n_in; j++) {
            size = align_block(size + (size_t)n_out * sizeof(double));
        }
    }

    return size;
}

/**
 * This functions only computes the amount of memory needed for the shared part of the network
 * excluding all parameters that change at each iteration of training (in, out, delta and targets)
 * */
size_t calculate_shared_model_size(int n_layers, const int n_neurons_per_layer[]) {
    size_t size = 0;
    
    // NeuralNet struct + architecture array
    size = align_block(size + sizeof(NeuralNet));
    size = align_block(size + (size_t)n_layers * sizeof(int));
    
    // Top-level weight/bias pointers
    size = align_block(size + (size_t)(n_layers - 1) * 3 * sizeof(double**)); // w, m_w, m2_w
    size = align_block(size + (size_t)(n_layers - 1) * 3 * sizeof(double*));  // b, m_b, m2_b

    for (int i = 0; i < n_layers - 1; i++) {
        int n_in  = n_neurons_per_layer[i] + 1;
        int n_out = n_neurons_per_layer[i+1] + 1;

        // 4. Intermediate row pointers (double**)
        size = align_block(size + (size_t)n_in * 3 * sizeof(double*)); 
        
        // 5. Bias data
        // Matches: nn->b[i] = (double*)curr; 
        //curr += n_in * sizeof(double); (repeated 3x)
        // Followed by: curr = (char*)align_block((uintptr_t)curr);
        size = align_block(size + (size_t)n_in * 3 * sizeof(double));  

        // 6. Weight data rows
        // Matches: for(j<n_in) { ... curr += n_out * sizeof(double) (x3); curr = align_block; }
        for (int j = 0; j < n_in; j++) {
             size = align_block(size + (size_t)n_out * 3 * sizeof(double));
        }
    }
    return size;
}

size_t calculate_total_system_memory(int n_layers, const int n_neurons_per_layer[], int n_nodes, int n_processes) {
    size_t shared_part = calculate_shared_model_size(n_layers, n_neurons_per_layer);
    size_t private_part = calculate_private_workspace_size(n_layers, n_neurons_per_layer);
    
    size_t private_step = (private_part > WORKER_SLOT_SIZE) ? align_page(private_part) : WORKER_SLOT_SIZE;

    // Total = (Shared Model * Nodes) + (Private Workspace * Processes)
    size_t total = (shared_part * n_nodes) + (private_step * n_processes);

    return align_page(total);
}


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
    // void* addr = (void*)base_address;

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
    // base_address = (uintptr_t)((char*)addr + aligned_size);

    return ret;
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


NeuralNet* newNetSingleAlloc(int n_layers, int n_neurons_per_layer[]) {
    size_t total_numa_map_size = (size_t)num_numa_nodes * PDE_ALIGN_SIZE;


    void* mmap_block = mmap_alloc(total_numa_map_size);

    for (int numa_node = 0; numa_node < num_numa_nodes; numa_node++) {
        char * net_addr = (char *)mmap_block + (numa_node * PDE_ALIGN_SIZE);
        
        // Materialize every page in the 2MB block
        for(size_t i=0; i < PDE_ALIGN_SIZE; i += 4096) net_addr[i] = 0;

        struct NeuralNet* nn = (struct NeuralNet*)net_addr;
        nn->magic_test_value = (numa_node == 0) ? 123.456 : 999.888;
        char * current_ptr = net_addr;

        nn->n_layers = n_layers;
        nn->total_mmap_size = PDE_ALIGN_SIZE;
        nn->initial_mmap_addr = net_addr;
        
        current_ptr += sizeof(struct NeuralNet);
        current_ptr = ALIGN_BLOCK(current_ptr);

        // 1. Metadata
        nn->n_neurons_per_layer = (int*)current_ptr;
        for (int i = 0; i < n_layers; i++) nn->n_neurons_per_layer[i] = n_neurons_per_layer[i];
        current_ptr += (n_layers * sizeof(int));
        current_ptr = ALIGN_BLOCK(current_ptr);

        // 2. Top-Level Pointers
        size_t lp_sz = (n_layers - 1) * sizeof(double**);
        size_t bp_sz = (n_layers - 1) * sizeof(double*);
        size_t back_sz = n_layers * sizeof(double*);

        nn->w = (double***)current_ptr;           current_ptr += lp_sz;
        nn->momentum_w = (double***)current_ptr;  current_ptr += lp_sz;
        nn->momentum2_w = (double***)current_ptr; current_ptr += lp_sz;
        nn->b = (double**)current_ptr;            current_ptr += bp_sz;
        nn->momentum_b = (double**)current_ptr;   current_ptr += bp_sz;
        nn->momentum2_b = (double**)current_ptr;  current_ptr += bp_sz;
        nn->delta = (double**)current_ptr;        current_ptr += back_sz;
        nn->in = (double**)current_ptr;           current_ptr += back_sz;
        nn->out = (double**)current_ptr;          current_ptr += back_sz;
        current_ptr = ALIGN_BLOCK(current_ptr);

        // 3. Row Pointers
        for (int i = 0; i < n_layers - 1; i++) {
            size_t rows = (size_t)(nn->n_neurons_per_layer[i] + 1);
            nn->w[i]           = (double**)current_ptr; current_ptr += rows * sizeof(double*);
            nn->momentum_w[i]  = (double**)current_ptr; current_ptr += rows * sizeof(double*);
            nn->momentum2_w[i] = (double**)current_ptr; current_ptr += rows * sizeof(double*);
        }
        current_ptr = ALIGN_BLOCK(current_ptr);

        // 4. Double Data
        for (int i = 0; i < n_layers - 1; i++) {
            int n_in = nn->n_neurons_per_layer[i] + 1;
            int n_out = nn->n_neurons_per_layer[i+1] + 1;
            size_t b_sz = (size_t)n_in * sizeof(double);
            size_t w_row_sz = (size_t)n_out * sizeof(double);

            nn->b[i] = (double*)current_ptr;           current_ptr += b_sz;
            nn->momentum_b[i] = (double*)current_ptr;  current_ptr += b_sz;
            nn->momentum2_b[i] = (double*)current_ptr; current_ptr += b_sz;
            current_ptr = ALIGN_BLOCK(current_ptr);

            for (int j = 0; j < n_in; j++) {
                nn->w[i][j] = (double*)current_ptr;           current_ptr += w_row_sz;
                nn->momentum_w[i][j] = (double*)current_ptr;  current_ptr += w_row_sz;
                nn->momentum2_w[i][j] = (double*)current_ptr; current_ptr += w_row_sz;
            }
            current_ptr = ALIGN_BLOCK(current_ptr);
        }

        // 5. Backprop Data
        for (int i = 0; i < n_layers; i++) {
            size_t sz = (size_t)(nn->n_neurons_per_layer[i] + 1) * sizeof(double);
            nn->delta[i] = (double*)current_ptr; current_ptr += sz;
            nn->in[i]    = (double*)current_ptr; current_ptr += sz;
            nn->out[i]   = (double*)current_ptr; current_ptr += sz;
            current_ptr = ALIGN_BLOCK(current_ptr);
        }

        nn->targets = (double*)current_ptr;
        current_ptr += (size_t)(nn->n_neurons_per_layer[n_layers-1] + 1) * sizeof(double);

        // Final sanity check
        if ((size_t)(current_ptr - net_addr) > PDE_ALIGN_SIZE) {
            fprintf(stderr, "FATAL: Node %d overflow: %zu/2097152\n", numa_node, (size_t)(current_ptr - net_addr));
            exit(EXIT_FAILURE);
        }
    }
    return (struct NeuralNet*)mmap_block;
}


/**
 * Function to create a neural network and allocate all its memory 
 * in a single contiguous block, aligned to 512 pages (2MB Huge Pages)
 * @param n_layers: int, number of layers
 * @param n_neurons_per_layer: int[], array of neurons per layer
 * @return struct NeuralNet*, pointer to the neural network
 */
/*NeuralNet** newNetSingleAlloc(int n_layers, int n_neurons_per_layer[]) {
    
    // Compute the total required size, aligned to the 2MB page boundary.
    // This size includes all data, pointers, and necessary padding for native alignment.
    size_t total_size = calculate_total_nn_size_for_single_mmap(n_layers, n_neurons_per_layer);
    // nn_array_size = num_numa_nodes * size of NeuralNet *
    size_t nn_array_size = (size_t)num_numa_nodes * sizeof(struct NeuralNet *);
    //size_t nn_array_size_aligned = (nn_array_size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    size_t nn_array_size_aligned = align_page(nn_array_size);
    // total_numa_map_size = total_size * num_numa_nodes + nn_array_size
    size_t total_numa_map_size = (total_size * (size_t)num_numa_nodes) + nn_array_size_aligned;


    // Map the entire memory block in a single call
    void* mmap_block = mmap_alloc(total_numa_map_size);
    
    struct NeuralNet **nn_array = (struct NeuralNet **) mmap_block;
    // char * current_start = (char *) mmap_block + nn_array_size
    char * current_start = (char *)mmap_block + nn_array_size_aligned;
    
    // Initialize the current pointer to traverse the allocated block.
    //char* current_ptr = (char*)mmap_block;


    for (int numa_node = 0; numa_node < num_numa_nodes; numa_node++) {


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

        char* master_base = (char*)current_start; 
        char* local_base = (char*)net_addr;

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
}*/


/**
 * Each child calls this after fork() to claim its slot.
 * They map in, out and delta pointers at specific fixed offset to
 * match the 2MB memory layout
 */
// Change return type from void to NeuralNet (the struct itself)
NNWorker assign_worker_zone(NeuralNet* shared_nn, int rank, unsigned long private_base, int total_samples, int n_processes) {
    NNWorker worker;
    
    // Identification & Data Chunking
    worker.process_id = rank;
    worker.shared_nn = shared_nn;
    
    // Calculate the workload distribution
    worker.chunk = total_samples / n_processes;
    worker.start_sample = rank * worker.chunk;
    // Handle the remainder if total_samples isn't perfectly divisible
    worker.end_sample = (rank == n_processes - 1) ? (total_samples - 1) : (worker.start_sample + worker.chunk - 1);

    // Memory Zone Initialization
    worker.my_zone_start = (char*)(private_base + (rank * WORKER_SLOT_SIZE));
    char* curr = worker.my_zone_start;
    int L = shared_nn->n_layers;

    // Map Private Pointer Arrays (at the start of the slot)
    worker.private_delta = (double**)curr; 
    curr += L * sizeof(double*);
    worker.private_in    = (double**)curr; 
    curr += L * sizeof(double*);
    worker.private_out   = (double**)curr; 
    curr += L * sizeof(double*);

    curr = (char*)align_block((uintptr_t)curr);

    // Map Activation Buffers
    for (int i = 0; i < L; i++) {
        int n = shared_nn->n_neurons_per_layer[i] + 1;
        size_t bytes = n * sizeof(double);

        worker.private_delta[i] = (double*)curr; 
        curr += bytes;
        worker.private_in[i]    = (double*)curr; 
        curr += bytes;
        worker.private_out[i]   = (double*)curr; 
        curr += bytes;
        
        curr = (char*)align_block((uintptr_t)curr);
    }
    worker.private_targets = (double*)curr; 
    curr += (shared_nn->n_neurons_per_layer[L-1] + 1) * sizeof(double);
    curr = (char*)align_block((uintptr_t)curr);

    // Map Gradient Storage (to avoid race conditions on shared weights)
    worker.local_grad_b = (double**)curr; 
    curr += (L - 1) * sizeof(double*);
    worker.local_grad_w = (double***)curr; 
    curr += (L - 1) * sizeof(double**);

    // Map Gradient Intermediate Row Pointers (Middle Level)
    // We need an array of double* for every row in every weight matrix
    for (int i = 0; i < L - 1; i++) {
        int n_in = shared_nn->n_neurons_per_layer[i] + 1;
        worker.local_grad_w[i] = (double**)curr;
        curr += n_in * sizeof(double*);
    }
    curr = (char*)align_block((uintptr_t)curr);


    // Map Actual Gradient Data (The double values)
    // This traverses layers and neurons to assign private data space
    for (int i = 0; i < L - 1; i++) {
        int n_in  = shared_nn->n_neurons_per_layer[i] + 1;
        int n_out = shared_nn->n_neurons_per_layer[i+1] + 1;

        // Assign memory for Bias Gradients (1D)
        worker.local_grad_b[i] = (double*)curr;
        curr += n_in * sizeof(double);
        curr = (char*)align_block((uintptr_t)curr);

        // Assign memory for Weight Gradients (2D Rows)
        for (int j = 0; j < n_in; j++) {
            worker.local_grad_w[i][j] = (double*)curr;
            curr += n_out * sizeof(double);
            // Row-level alignment for performance
            curr = (char*)align_block((uintptr_t)curr);
        }
    }

    // Safety Verification
    if ((uintptr_t)curr > (uintptr_t)worker.my_zone_start + WORKER_SLOT_SIZE) {
        fprintf(stderr, "[FATAL] Rank %d exceeded 256KB slot! Used: %zu\n", 
                rank, (size_t)(curr - worker.my_zone_start));
        exit(1);
    }

    return worker;
}


/**
 * Allocates a global memory slab for a Multi-Node, Multi-Process Neural Network.
 * Layout: [NN_Ptr_Array] + [NUMA_Replicas (2MB aligned)] + [Worker_Slots (256KB aligned)]
 */
NeuralNet** allocSharedNN(int n_layers, int n_neurons_per_layer[], int n_processes) {
    
    // shared model size, weights, bias,...
    size_t shared_raw = calculate_shared_model_size(n_layers, n_neurons_per_layer);
    
    // Align shared network to 2MB
    size_t shared_aligned = (shared_raw + PDE_ALIGN_SIZE - 1) & ~(PDE_ALIGN_SIZE - 1);
    
    // size of the neural network ptr
    size_t nn_ptr_array_size = (size_t)num_numa_nodes * sizeof(NeuralNet *);
    size_t nn_ptr_array_aligned = (nn_ptr_array_size + PDE_ALIGN_SIZE - 1) & ~(PDE_ALIGN_SIZE-1); 

    // Total Slab = Pointers + (shared_model * Nodes) + (private_network * Processes)
    size_t total_slab_size = nn_ptr_array_aligned + 
                             (shared_aligned * num_numa_nodes) + 
                             (WORKER_SLOT_SIZE * n_processes);

    // allocate global address space 
    void* mmap_block = mmap_alloc(total_slab_size);
    if (!mmap_block) return NULL;

    NeuralNet **nn_array = (NeuralNet **)mmap_block;
    char *shared_base = (char *)mmap_block + nn_ptr_array_aligned;

    // init shared copies on each numa node (must be revised)
    for (int node = 0; node < num_numa_nodes; node++) {
        char* net_addr = shared_base + (node * shared_aligned);
        
        // this must be only used as baseline vs LKM technique
        bind_memory_to_numa_node(net_addr, shared_aligned, node);

        struct NeuralNet* nn = (struct NeuralNet*)net_addr;
        nn_array[node] = nn;

        // --- Metadata ---
        nn->n_layers = n_layers;
        nn->numa_node_id = node;
        nn->initial_mmap_addr = mmap_block;
        nn->total_mmap_size = total_slab_size;

        char* curr = (char*)net_addr + sizeof(struct NeuralNet);
        curr = (char*)align_block((uintptr_t)curr);

        // --- Architecture Array ---
        // CRITICAL: The array of neurons MUST be copied into the shared MMAP
        nn->n_neurons_per_layer = (int*)curr;
        for (int i = 0; i < n_layers; i++) {
            nn->n_neurons_per_layer[i] = n_neurons_per_layer[i];
        }
        curr += n_layers * sizeof(int);
        curr = (char*)align_block((uintptr_t)curr);

        // --- Weight/Bias Top-Level Pointers ---
        size_t w_top_size = (n_layers - 1) * sizeof(double**);
        size_t b_top_size = (n_layers - 1) * sizeof(double*);

        nn->w = (double***)curr;           curr += w_top_size;
        nn->momentum_w = (double***)curr;  curr += w_top_size;
        nn->momentum2_w = (double***)curr; curr += w_top_size;

        nn->b = (double**)curr;            curr += b_top_size;
        nn->momentum_b = (double**)curr;   curr += b_top_size;
        nn->momentum2_b = (double**)curr;  curr += b_top_size;

        curr = (char*)align_block((uintptr_t)curr);

        // --- Intermediate Pointers (Rows) ---
        for (int i = 0; i < n_layers - 1; i++) {
            int n_in = nn->n_neurons_per_layer[i] + 1;
            size_t row_ptr_size = n_in * sizeof(double*);

            nn->w[i] = (double**)curr;           curr += row_ptr_size;
            nn->momentum_w[i] = (double**)curr;  curr += row_ptr_size;
            nn->momentum2_w[i] = (double**)curr; curr += row_ptr_size;
        }

        // --- Real Data (Weights and Biases) ---
        // curr now points to the start of the double values
        for (int i = 0; i < n_layers - 1; i++) {
            int n_in = nn->n_neurons_per_layer[i] + 1;
            int n_out = nn->n_neurons_per_layer[i+1] + 1;

            // Bias data
            nn->b[i] = (double*)curr;           curr += n_in * sizeof(double);
            nn->momentum_b[i] = (double*)curr;  curr += n_in * sizeof(double);
            nn->momentum2_b[i] = (double*)curr; curr += n_in * sizeof(double);
            curr = (char*)align_block((uintptr_t)curr);

            // Weight data
            for (int j = 0; j < n_in; j++) {
                nn->w[i][j] = (double*)curr;           curr += n_out * sizeof(double);
                nn->momentum_w[i][j] = (double*)curr;  curr += n_out * sizeof(double);
                nn->momentum2_w[i][j] = (double*)curr; curr += n_out * sizeof(double);
                curr = (char*)align_block((uintptr_t)curr);
            }
        }

        // backpropagation pointers are set to NULL because they live in the Private Pool
        // assigned after fork() based on process rank.
        nn->in = NULL;
        nn->out = NULL;
        nn->delta = NULL;
        nn->targets = NULL;

        // Verification: Ensure we didn't bleed into the next 2MB node slab
        size_t used = (size_t)(curr - net_addr);
        if (used > shared_aligned) {
            fprintf(stderr, "FATAL: Node %d data (%zu) exceeded 2MB slab!\n", node, used);
            exit(1);
        }
    }

    return nn_array;
}