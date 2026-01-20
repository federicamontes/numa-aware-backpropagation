#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <stdbool.h>
#include "neural_network.h"
#include "read_data.h"
#include "memory_manager.h"
#include "utils.h"

// Helper to get the PDE index from a virtual address
#define GET_PDE_INDEX(addr) (((uintptr_t)(addr) >> 21) & 0x1FF)

// --- HELPER FOR UNIFIED MATH ---
uintptr_t get_private_pool_base(void* mmap_start, int n_layers, int n_neurons_per_layer[]) {
    size_t ptr_block = (num_numa_nodes * sizeof(NeuralNet *) + PDE_ALIGN_SIZE - 1) & ~(PDE_ALIGN_SIZE - 1);
    size_t shared_raw = calculate_shared_model_size(n_layers, n_neurons_per_layer);
    size_t shared_aligned = (shared_raw + PDE_ALIGN_SIZE - 1) & ~(PDE_ALIGN_SIZE - 1);
    return (uintptr_t)mmap_start + ptr_block + (num_numa_nodes * shared_aligned);
}


// --- UPDATED AUDIT FUNCTION WITH VISUALIZATION ---
bool perform_compliance_audit(NeuralNet** nn_array, int n_layers, int n_neurons_per_layer[]) {
    bool compliant = true;
    uintptr_t mmap_base = (uintptr_t)nn_array[0]->initial_mmap_addr;
    
    printf("\n--- LKM COMPLIANCE & MEMORY MAP ---\n");
    printf("%-20s | %-18s | %-10s | %s\n", "Segment", "Virtual Address", "PDE Index", "Status");
    printf("---------------------|--------------------|------------|----------\n");

    // 1. Pointer Array (Directory)
    printf("%-20s | %p | %-10lu | %s\n", 
           "NN Directory (Ptrs)", 
           (void*)nn_array, 
           GET_PDE_INDEX(nn_array),
           ((uintptr_t)nn_array % 4096 == 0) ? "OK" : "MISALIGNED");

    // 2. Node Replicas
    for (int n = 0; n < num_numa_nodes; n++) {
        char label[24];
        sprintf(label, "Node %d Weights", n);
        uintptr_t addr = (uintptr_t)nn_array[n];
        printf("%-20s | 0x%lx | %-10lu | %s\n", 
               label, 
               (unsigned long)addr, 
               GET_PDE_INDEX(addr),
               (addr % PDE_ALIGN_SIZE == 0) ? "PDE-ALIGN" : "SHARED-PDE");
        if (addr % PDE_ALIGN_SIZE != 0) compliant = false;
    }

    // 3. Metadata
    uintptr_t arch_ptr = (uintptr_t)nn_array[0]->n_neurons_per_layer;
    bool in_slab = (arch_ptr >= mmap_base && arch_ptr < (mmap_base + nn_array[0]->total_mmap_size));
    printf("%-20s | 0x%lx | %-10lu | %s\n", 
           "Metadata (Neurons)", 
           (unsigned long)arch_ptr, 
           GET_PDE_INDEX(arch_ptr),
           in_slab ? "SHARED" : "LOCAL-ERR");


    uintptr_t private_start = get_private_pool_base(nn_array[0]->initial_mmap_addr, n_layers, n_neurons_per_layer);
    printf("%-20s | 0x%lx | %-10lu | %s\n", 
       "Private Workspace", 
       (unsigned long)private_start, 
       GET_PDE_INDEX(private_start),
       (private_start % 4096 == 0) ? "OK" : "FAIL");

    return compliant;
}

int main(int argc, const char* argv[]) {
    int n_layers = 4;
    int n_neurons_per_layer[] = {784, 64, 32, 10};
    int n_processes = 2;
    int n_nodes = num_numa_nodes;

    
    // setup Hyper-parameters
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

    printf("=== STEP 0: MEMORY PLANNING & DEBUGGING ===\n");

    // 1. Inspect sizes of components
    size_t shared_size  = calculate_shared_model_size(n_layers, n_neurons_per_layer);
    size_t private_work = calculate_private_workspace_size(n_layers, n_neurons_per_layer);
    
    // 2. Inspect Aggregate allocations
    size_t single_nn_mmap = calculate_total_nn_size_for_single_mmap(n_layers, n_neurons_per_layer);
    size_t total_sys_mem  = calculate_total_system_memory(n_layers, n_neurons_per_layer, n_nodes, n_processes);

    printf("Shared Model Size (Data Only):    %zu bytes (~%.2f KB)\n", 
            shared_size, (double)shared_size / 1024.0);
    printf("Private Workspace (Per Worker):   %zu bytes (~%.2f KB)\n", 
            private_work, (double)private_work / 1024.0);
    printf("Total System Memory Requested:    %zu bytes (~%.2f MB)\n", 
            total_sys_mem, (double)total_sys_mem / (1024.0 * 1024.0));

    printf("\n--- PDE BOUNDARY ANALYSIS ---\n");
    printf("Required for 1 Shared Model:      %zu bytes\n", shared_size);
    printf("Allocated per Shared PDE (2MB):   %d bytes\n", PDE_ALIGN_SIZE);
    printf("Wasted 'Slack' space in PDE:      %ld bytes\n", (long)PDE_ALIGN_SIZE - (long)shared_size);
    
    printf("\n--- PRIVATE SLOT ANALYSIS ---\n");
    printf("Required for 1 Worker Data:       %zu bytes\n", private_work);
    printf("Allocated per Worker Slot (1MB):  %d bytes\n", WORKER_SLOT_SIZE);
    printf("Free 'Slack' space in Slot:       %ld bytes\n", (long)WORKER_SLOT_SIZE - (long)private_work);

    printf("\n=== STEP 1: ALLOCATION ===\n");
    NeuralNet** nn_array = allocSharedNN(n_layers, n_neurons_per_layer, n_processes);
    if (!nn_array) return 1;

    // Perform Audit (Visualizes the actual resulting pointers)
    if (!perform_compliance_audit(nn_array, n_layers, n_neurons_per_layer)) {
        fprintf(stderr, "\n[FATAL] Memory layout is invalid for LKM. Aborting.\n");
        return 1;
    }
    for (int i = 0; i < num_numa_nodes; i++) {
        if (nn_array[i] != NULL) {
            srand(42); // I want to have the same weights on each NUMA node 
            printf("Node %d: Base %p | NN %p | Weights[0][0] %p\n",
                   nn_array[i]->numa_node_id, 
                   nn_array[i]->initial_mmap_addr, 
                   (void*)nn_array[i], 
                   (void*)nn_array[i]->w[0][0]);
            
            // Initialize weights only for the master/base node
            if (i == 0) init_nn(nn_array[i]); 
        } else {
            fprintf(stderr, "Critical Error: Allocation failed on NUMA node %d\n", i);
            return EXIT_FAILURE;
        }
    }

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

    // --- CALCULATE BASE FOR FORK ---
    void* mmap_base = nn_array[0]->initial_mmap_addr;
    uintptr_t private_base = get_private_pool_base(mmap_base, n_layers, n_neurons_per_layer);

    printf("\n[SUCCESS] Private Base: 0x%lx\n", (unsigned long)private_base);
    printf("Total Managed MMAP Range: [0x%p - 0x%p]\n", 
            mmap_base, (char*)mmap_base + total_sys_mem);


    printf("=== STEP 2: FORKING PARALLEL WORKERS ===\n");
    fflush(stdout);
    for (int i = 0; i < n_processes; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            // Each child process chooses a NUMA nodint my_node = i % num_numa_nodes;
            int my_node = i % num_numa_nodes;

            // 2. Pin process to the physical hardware of that node
            if (pin_to_numa_node(my_node) == 0) {
                printf("[Child %d] Pinned to NUMA node %d\n", i, my_node);
            }
            
            // Initialize the Worker with all specific parameters
            NNWorker worker = assign_worker_zone(nn_array[my_node], i, private_base, num_samples_to_train, n_processes);

            // LOGGING THE INTEGRATION
            printf("[Child %d] Node %d | Data Chunk: %d samples [%d to %d] | Slot: %p\n", 
                    worker.process_id, 
                    my_node, 
                    worker.chunk,
                    worker.start_sample, 
                    worker.end_sample,
                    (void*)worker.my_zone_start);
            fflush(stdout);

            // Calculate boundaries for verification
            uintptr_t my_zone_addr = (uintptr_t)worker.my_zone_start;
            uintptr_t slot_end = my_zone_addr + WORKER_SLOT_SIZE;
            uintptr_t grad_w = (uintptr_t)worker.local_grad_w[0][0];

            // Print Gradient Info
            // We check the first weight gradient of the first layer: local_grad_w[0][0][0]
            printf("[Child %d] Gradient Base (grad_b[0]): %p\n", i, (void*)worker.local_grad_b[0]);
            printf("[Child %d] Gradient Data (grad_w[0][0]): %p\n", i, (void*)worker.local_grad_w[0][0]);

            // Logic Check: Is the gradient data inside the private slot?
            uintptr_t grad_ptr = (uintptr_t)worker.local_grad_w[0][0];
            if (grad_ptr >= my_zone_addr && grad_ptr < slot_end) {
                printf("[Child %d] VERIFIED: Gradients are inside the private process slot.\n", i);
            } else {
                fprintf(stderr, "[Child %d] ERROR: Gradients leaked outside slot!\n", i);
                exit(1);
            }

            printf("[Child %d] Mapping Summary:\n", i);
            printf("  -> Private Slot Base: 0x%lx (PDE: %lu)\n", my_zone_addr, GET_PDE_INDEX(my_zone_addr));
            printf("  -> Weight Gradients : 0x%lx (PDE: %lu)\n", grad_w, GET_PDE_INDEX(grad_w));
            
            // Check if Gradients are in the same PDE as Weights (They SHOULD NOT BE)
            if (GET_PDE_INDEX(grad_w) == GET_PDE_INDEX(nn_array[my_node])) {
                printf("  [!!] WARNING: Gradients are sharing a PDE with Weights!\n");
            } else {
                printf("  [OK] Isolation: Gradients are in a separate PDE from Shared Weights.\n");
            }
            
            fflush(stdout);

            // VERIFICATION OF CONTIGUITY
            uintptr_t expected_offset = private_base + (i * WORKER_SLOT_SIZE);
            if ((uintptr_t)worker.my_zone_start != expected_offset) {
                fprintf(stderr, "[Child %d] ERROR: Misaligned slot start!\n", i);
                exit(1);
            }

            // At this point, you would call your training function:
            // worker_train(&worker, X_train, y_train, ...);

            exit(0); 
        }
    }
    // Parent waits
    for (int i = 0; i < n_processes; i++) wait(NULL);
    printf("\n=== TEST COMPLETE ===\n");

    return 0;
}