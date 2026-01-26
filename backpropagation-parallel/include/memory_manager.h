#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <stddef.h>
#include <sys/mman.h>
#include "neural_network.h"


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


extern unsigned long base_address;

size_t align_block(size_t current_size);
size_t align_page(size_t current_size);


/**
 * Initializes the neural network weights with random values and 
 * resets optimizer momentum buffers to zero.
 */
void init_nn(NeuralNet* nn);


/**
 * Calculates the exact sum of all memory dimensions required for the neural network.
 * This is used to determine the size of the single mmap block for shared memory.
 * * @param n_layers: Number of layers (L).
 * @param n_neurons_per_layer: Array specifying neurons per layer.
 * @return Total required size in bytes.
 */
size_t sum_all_mmap_allocations(int n_layers, const int n_neurons_per_layer[]);

/**
 * Calculates the total size required for a single contiguous mmap allocation.
 */
size_t calculate_total_nn_size_for_single_mmap(int n_layers, const int n_neurons_per_layer[]);
size_t calculate_total_system_memory(int n_layers, const int n_neurons_per_layer[], int n_nodes, int n_processes);


/**
 * Calculates the size of the PRIVATE workspace (in, out, delta, targets).
 * Each process needs exactly one of these blocks.
 */
size_t calculate_private_workspace_size(int n_layers, const int n_neurons_per_layer[]);

/**
 * Calculates the size of the SHARED parts only (Struct, Weights, Biases, Momentum).
 * This is mapped once and shared across all processes.
 */
size_t calculate_shared_model_size(int n_layers, const int n_neurons_per_layer[]);

/**
 * Creates an array of Neural Networks (one per NUMA node) in a single 
 * contiguous memory block. Each network is bound to its respective NUMA node.
 * * @return NeuralNet**, an array of pointers to the networks.
 */
//NeuralNet** newNetSingleAlloc(int n_layers, int n_neurons_per_layer[]);
NeuralNet* newNetSingleAlloc(int n_layers, int n_neurons_per_layer[]);


/**
 * Each child calls this after fork() to claim its 256KB slot.
 * They map in, out and delta pointers at specific fixed offset to
 * match the 2MB memory layout
 */
NNWorker assign_worker_zone(NeuralNet* shared_nn, int rank, unsigned long private_base, int total_samples, int n_processes);

/** Parent process maps the entire network (MAP_SHARED), then calls fork()
 *  each children mmaps its own in, out pointers (MAP_PRIVATE?)
 * 
 *  Allocates a global memory slab for a Multi-Node, Multi-Process Neural Network.
 *  Layout: [NN_Ptr_Array] + [NUMA_Replicas (2MB aligned)] + [Worker_Slots (256KB aligned)]
 * */
NeuralNet** allocSharedNN(int n_layers, int n_neurons_per_layer[], int n_processes);

/**
 * Releases the shared memory region allocated for the network.
 */
void free_NN(NeuralNet* nn);


/**
 * Binds a memory range to a specific NUMA node using MPOL_BIND.
 */
void bind_memory_to_numa_node(void* addr, size_t size, int node);

/**
 * Allocates a shared memory block at a fixed base address using mmap.
 * Increments the global base_address to prevent overlaps.
 */
void* mmap_alloc(size_t size);

#endif