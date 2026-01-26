#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <stdbool.h>
#include <string.h>
#include "neural_network.h"
#include "read_data.h"
#include "memory_manager.h"
#include "utils.h"

// --- CONSTANTS & MACROS ---
#define GET_PDE_INDEX(addr) (((uintptr_t)(addr) >> 21) & 0x1FF)
#define N_SAMPLES 60000
#define N_TEST_SAMPLES 10000
#define N_DIMS 784
#define N_CLASSES 10

// --- LOGGING & AUDIT HELPERS ---

void print_memory_plan(int n_layers, int n_neurons_per_layer[], int n_processes) {
    size_t shared_size  = calculate_shared_model_size(n_layers, n_neurons_per_layer);
    size_t private_work = calculate_private_workspace_size(n_layers, n_neurons_per_layer);
    size_t total_sys_mem = calculate_total_system_memory(n_layers, n_neurons_per_layer, num_numa_nodes, n_processes);

    printf("=== STEP 0: MEMORY PLANNING & DEBUGGING ===\n");
    printf("Shared Model Size (Data Only):    %zu bytes (~%.2f KB)\n", shared_size, (double)shared_size / 1024.0);
    printf("Private Workspace (Per Worker):   %zu bytes (~%.2f KB)\n", private_work, (double)private_work / 1024.0);
    printf("Total System Memory Requested:    %zu bytes (~%.2f MB)\n", total_sys_mem, (double)total_sys_mem / (1024.0 * 1024.0));

    printf("\n--- PDE BOUNDARY ANALYSIS ---\n");
    printf("Allocated per Shared PDE (2MB):   %d bytes\n", PDE_ALIGN_SIZE);
    printf("Wasted 'Slack' space in PDE:      %ld bytes\n", (long)PDE_ALIGN_SIZE - (long)shared_size);
    
    printf("\n--- PRIVATE SLOT ANALYSIS ---\n");
    printf("Allocated per Worker Slot (1MB):  %d bytes\n", WORKER_SLOT_SIZE);
    printf("Free 'Slack' space in Slot:       %ld bytes\n", (long)WORKER_SLOT_SIZE - (long)private_work);
}

bool perform_compliance_audit(NeuralNet** nn_array, int n_layers, int n_neurons_per_layer[]) {
    bool compliant = true;
    uintptr_t mmap_base = (uintptr_t)nn_array[0]->initial_mmap_addr;
    
    printf("\n--- LKM COMPLIANCE & MEMORY MAP ---\n");
    printf("%-20s | %-18s | %-10s | %s\n", "Segment", "Virtual Address", "PDE Index", "Status");
    printf("---------------------|--------------------|------------|----------\n");

    printf("%-20s | %p | %-10lu | %s\n", "NN Directory", (void*)nn_array, GET_PDE_INDEX(nn_array), "OK");

    for (int n = 0; n < num_numa_nodes; n++) {
        char label[24]; sprintf(label, "Node %d Weights", n);
        uintptr_t addr = (uintptr_t)nn_array[n];
        printf("%-20s | 0x%lx | %-10lu | %s\n", label, (unsigned long)addr, GET_PDE_INDEX(addr), 
               (addr % PDE_ALIGN_SIZE == 0) ? "PDE-ALIGN" : "MISALIGNED");
        if (addr % PDE_ALIGN_SIZE != 0) compliant = false;
    }

    uintptr_t arch_ptr = (uintptr_t)nn_array[0]->n_neurons_per_layer;
    printf("%-20s | 0x%lx | %-10lu | %s\n", "Metadata", (unsigned long)arch_ptr, GET_PDE_INDEX(arch_ptr), "SHARED");

    return compliant;
}

void log_worker_status(NNWorker* worker, NeuralNet* shared_nn, int my_node) {
    uintptr_t my_zone_addr = (uintptr_t)worker->my_zone_start;
    uintptr_t grad_w = (uintptr_t)worker->local_grad_w[0][0];

    printf("[Child %d] Node %d | Data Chunk: %d samples [%d to %d]\n", 
           worker->process_id, my_node, worker->chunk, worker->start_sample, worker->end_sample);
    
    printf("[Child %d] Mapping: Slot 0x%lx (PDE %lu) | Grad 0x%lx (PDE %lu)\n",
           worker->process_id, my_zone_addr, GET_PDE_INDEX(my_zone_addr), grad_w, GET_PDE_INDEX(grad_w));

    if (GET_PDE_INDEX(grad_w) == GET_PDE_INDEX(shared_nn)) {
        printf("  [!!] WARNING: Gradients are sharing a PDE with Weights!\n");
    } else {
        printf("  [OK] Isolation: Gradients are in a separate PDE from Shared Weights.\n");
    }
    fflush(stdout);
}

/**
 * Calculates the start of the Private Workspace pool.
 * It skips over the Pointer Directory and all NUMA Shared Model replicas.
 */
uintptr_t get_private_pool_base(void* mmap_start, int n_layers, int n_neurons_per_layer[]) {
    // 1. Skip the Pointer Directory (Block 0)
    size_t ptr_block = (num_numa_nodes * sizeof(struct NeuralNet *) + PDE_ALIGN_SIZE - 1) & ~(PDE_ALIGN_SIZE - 1);
    
    // 2. Calculate the size of one shared model replica
    size_t shared_raw = calculate_shared_model_size(n_layers, n_neurons_per_layer);
    
    // 3. Align that size to 2MB (PDE boundary)
    size_t shared_aligned = (shared_raw + PDE_ALIGN_SIZE - 1) & ~(PDE_ALIGN_SIZE - 1);
    
    // 4. Base = Start + Pointers + (Number of Nodes * Aligned Shared Size)
    return (uintptr_t)mmap_start + ptr_block + (num_numa_nodes * shared_aligned);
}

// --- MAIN TEST CASE ---

int main(int argc, const char* argv[]) {
    // 1. Initial configuration
    int n_layers = 4;
    int n_neurons_per_layer[] = {784, 64, 32, 10};
    int n_processes = 2;
    int num_samples_to_train = 10000;
    int epochs = 5;

    print_memory_plan(n_layers, n_neurons_per_layer, n_processes);

    // 2. Memory Allocation
    printf("\n=== STEP 1: ALLOCATION ===\n");
    NeuralNet** nn_array = allocSharedNN(n_layers, n_neurons_per_layer, n_processes);
    if (!nn_array) return 1;

    if (!perform_compliance_audit(nn_array, n_layers, n_neurons_per_layer)) {
        fprintf(stderr, "\n[FATAL] Memory layout is invalid for LKM. Aborting.\n");
        return 1;
    }

    // 3. Weight Initialization
    for (int i = 0; i < num_numa_nodes; i++) {
        if (nn_array[i] != NULL) {
            srand(42); // Ensure identical starting weights across all replicas
            init_nn(nn_array[i]);
            printf("Node %d: Base %p | Weights[0][0] %p (Initialized)\n",
                   nn_array[i]->numa_node_id, (void*)nn_array[i], (void*)nn_array[i]->w[0][0]);
        }
    }

    // 4. Data Loading and Pre-processing
    printf("\n=== STEP 1.5: DATA LOADING ===\n");
    double** X_train = malloc(N_SAMPLES * sizeof(double*));
    double** y_train = malloc(N_SAMPLES * sizeof(double*));
    double* y_train_temp = malloc(N_SAMPLES * sizeof(double));
    for(int i=0; i<N_SAMPLES; i++) {
        X_train[i] = malloc(N_DIMS * sizeof(double));
        y_train[i] = malloc(N_CLASSES * sizeof(double));
    }

    double** X_test = malloc(N_TEST_SAMPLES * sizeof(double*));
    double** y_test = malloc(N_TEST_SAMPLES * sizeof(double*));
    double* y_test_temp = malloc(N_TEST_SAMPLES * sizeof(double));
    for(int i=0; i<N_TEST_SAMPLES; i++) {
        X_test[i] = malloc(N_DIMS * sizeof(double));
        y_test[i] = malloc(N_CLASSES * sizeof(double));
    }

    printf("Loading dataset from CSV...\n");
    read_csv_file(X_train, y_train_temp, y_train, "train");
    scale_data(X_train, "train");
    read_csv_file(X_test, y_test_temp, y_test, "test");
    scale_data(X_test, "test");
    normalize_data(X_train, X_test);

    FILE* metrics_file = fopen("metrics_64_32.txt", "w");
    if (metrics_file) fprintf(metrics_file, "train_loss,train_acc,test_loss,test_acc\n");

    // 5. Parallel Worker Forking
    void* mmap_base = nn_array[0]->initial_mmap_addr;
    uintptr_t private_base = get_private_pool_base(mmap_base, n_layers, n_neurons_per_layer);
    size_t total_sys_mem = calculate_total_system_memory(n_layers, n_neurons_per_layer, num_numa_nodes, n_processes);

    printf("\n[SUCCESS] Private Base: 0x%lx\n", (unsigned long)private_base);
    printf("Total Managed MMAP Range: [%p - %p]\n", mmap_base, (char*)mmap_base + total_sys_mem);

    printf("\n=== STEP 2: FORKING PARALLEL WORKERS ===\n");
    fflush(stdout);

    for (int i = 0; i < n_processes; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child logic
            int my_node = i % num_numa_nodes;

            // Pin to physical CPUs of the NUMA node
            if (pin_to_numa_node(my_node) == 0) {
                printf("[Child %d] Pinned to NUMA node %d\n", i, my_node);
            }
            
            // Assign the memory slot
            NNWorker worker = assign_worker_zone(nn_array[my_node], i, private_base, num_samples_to_train, n_processes);

            // Log detailed mapping info
            log_worker_status(&worker, nn_array[my_node], my_node);

            // Here you would call: worker_train(&worker, X_train, y_train, epochs);
            
            exit(0); 
        }
    }

    // 6. Parent Wait & Cleanup
    for (int i = 0; i < n_processes; i++) wait(NULL);

    if (metrics_file) fclose(metrics_file);
    for(int i=0; i<N_SAMPLES; i++) { free(X_train[i]); free(y_train[i]); }
    for(int i=0; i<N_TEST_SAMPLES; i++) { free(X_test[i]); free(y_test[i]); }
    free(X_train); free(y_train); free(y_train_temp);
    free(X_test); free(y_test); free(y_test_temp);

    printf("\n=== TEST COMPLETE ===\n");
    return 0;
}