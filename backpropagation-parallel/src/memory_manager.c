#include "memory_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <numaif.h>
#include "utils.h"

// Define global variables
unsigned long base_address = 0x1000000000; // Aligned to 2MB


size_t align_block(size_t current_size) {
    return (current_size + PTR_ALIGNMENT - 1) & ~(PTR_ALIGNMENT - 1);
}

size_t align_page(size_t current_size) {
    return (current_size + PAGE_ALIGNMENT - 1) & ~(PAGE_ALIGNMENT - 1);
}



size_t sum_all_mmap_allocations(int n_layers, const int n_neurons_per_layer[]) {
    size_t total_required_size = 0;
    
    // Space for the base struct itself
    total_required_size += sizeof(NeuralNet);
    

    // Stores the architecture configuration
    total_required_size += (size_t)n_layers * sizeof(int);
    
    // Top-Level Pointers (Weight and Bias Matrices)
    // nn->w, nn->momentum_w, nn->momentum2_w: 3 pointer arrays
    total_required_size += (size_t)(n_layers - 1) * 3 * sizeof(double***); 
    
    // nn->b, nn->momentum_b, nn->momentum2_b: 3 pointer arrays
    total_required_size += (size_t)(n_layers - 1) * 3 * sizeof(double**);  

    // Loop over Weight Layers (Intermediate Pointers and Data)
    for (int i = 0; i < n_layers - 1; i++) {
        int n_in  = n_neurons_per_layer[i] + 1;    // Current layer (plus bias)
        int n_out = n_neurons_per_layer[i+1] + 1;  // Next layer (plus bias)

        // Intermediate Pointers for W (w[i], m_w[i], m2_w[i])
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


/**
 * Function to create a neural network and allocate all its memory 
 * in a single contiguous block, aligned to 512 pages (2MB Huge Pages)
 * @param n_layers: int, number of layers
 * @param n_neurons_per_layer: int[], array of neurons per layer
 * @return struct NeuralNet*, pointer to the neural network
 */
NeuralNet** newNetSingleAlloc(int n_layers, int n_neurons_per_layer[]) {
    
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

