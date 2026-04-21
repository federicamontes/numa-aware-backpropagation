#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <numa.h>
#include "neural_network.h" 

// syscall number (check it in your system with dmesg after mounting modules)
#define SYS_PES 156 

extern NeuralNet* setup_numa_model(int n_layers, int* neurons);
extern void pin_process_to_node(int rank, int node);
extern NNWorkerWorkspace* setup_worker_workspace(NeuralNet* shared_nn, int rank, int node_id);

int main() {
    int n_layers = 3;
    int neurons[] = {10, 5, 2}; // Input, Hidden, Output
    int n_workers = 2;          // processes
    int num_numa_nodes = 2;     // simulate numa nodes

    printf("[TEST] Init numa model...\n");
    
    NeuralNet* base_nn = setup_numa_model(n_layers, neurons);

    if (!base_nn) {
        fprintf(stderr, "Errore allocazione setup_numa_model\n");
        return 1;
    }

    printf("[PARENT] Base address: %p\n", (void*)base_nn);
    printf("[PARENT] Magic Value (Slab 0): %.2f\n", base_nn->magic_test_value);
    printf("--------------------------------------------------\n");

    size_t slab_size = ALIGN_PAGE(calculate_shared_model_size(base_nn->n_layers, base_nn->n_neurons_per_layer));
    

    printf("[PARENT] Force materialization...\n");
    for (int n = 0; n < num_numa_nodes; n++) {
        char* slab_ptr = (char*)base_nn + (n * PAGE_ALIGNMENT);
        
        for (size_t i = 0; i < PAGE_ALIGNMENT; i += 4096) {
            slab_ptr[i] = (char)(n + 1);
        }
        
        NeuralNet* nn = (NeuralNet*)slab_ptr;
        nn->magic_test_value = (n == 0) ? 123.45 : 678.90;
        
        printf("  [SLAB %d] Touch completed a %p (Magic: %.2f)\n", n, (void*)slab_ptr, nn->magic_test_value);
    }

    // Fork processes
    for (int rank = 0; rank < n_workers; rank++) {
        pid_t pid = fork();
        
        if (pid < 0) {
            perror("fork failed");
            exit(EXIT_FAILURE);
        }

        if (pid == 0) { // FIGLIO
            int my_node = rank % num_numa_nodes;

            // Pinning 
            pin_process_to_node(rank, my_node);

            // touchn memory in child to force pte materialization
            volatile char* touch_ptr = (char*)base_nn;
            for (int n = 0; n < num_numa_nodes; n++) {
                char c = touch_ptr[n * PAGE_ALIGNMENT]; 
                touch_ptr[n * PAGE_ALIGNMENT] = c;
            }
            // -----------------------------------------

            double val_pre = base_nn->magic_test_value;
            printf("[RANK %d] Indirizzo Virtuale base_nn: %p\n", rank, (void*)base_nn);

            // call syscall to switch memory view for child
            long ret = syscall(SYS_PES, (unsigned long)base_nn, 0, my_node);

            if (ret < 0) {
                fprintf(stderr, "[RANK %d] CRITICAL: pes syscall failed for node %d\n", rank, my_node);
                exit(EXIT_FAILURE);
            }

            NNWorkerWorkspace *local_nn = setup_worker_workspace(base_nn, rank, my_node);
    
            if (!local_nn) {
                printf("[RANK %d] Errore allocazione Workspace\n", rank);
                exit(1);
            }

            // --- STAMPE DI DEBUG ---
            printf("[RANK %d] Workspace allocato a: %p\n", rank, (void*)local_nn);
            printf("[RANK %d] Workspace->shared_nn punta a: %p\n", rank, (void*)local_nn->shared_nn);

            double val_post = base_nn->magic_test_value;
            double val_via_ws = local_nn->shared_nn->magic_test_value;

            void* internal_ptr = (void*)base_nn->n_neurons_per_layer;

            
            printf("[RANK %d] --- VERIFICA POST-PES (Nodo %d) ---\n", rank, my_node);
            printf("[RANK %d] Magic PRE: %.2f | POST: %.2f\n", rank, val_pre, val_post);
            printf("[RANK %d] Magic LETTO VIA WORKSPACE: %.2f\n", rank, val_via_ws);
            printf("[RANK %d] Internal Ptr (n_neurons): %p\n", rank, internal_ptr);

            if (my_node > 0 && val_pre == val_post) {
                printf("[RANK %d] RESULT: FAILURE (Vista non cambiata)\n", rank);
            } else if (val_post != val_via_ws) {
                printf("[RANK %d] RESULT: FAILURE (Incoerenza Workspace/Shared)\n", rank);
            } else {
                printf("[RANK %d] RESULT: SUCCESS\n", rank);
            }

            exit(0);
        }
    }

    // Wait
    for (int i = 0; i < n_workers; i++) {
        wait(NULL);
    }

    printf("--------------------------------------------------\n");
    printf("[TEST] Completato.\n");

    return 0;
}