#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <numa.h>
#include "neural_network.h"

typedef struct {
    int rank;
    int target_node;
    double canary_read;
    double expected_canary;
    double local_write_val;
    double expected_write_val;
    int neuron_dim_0;
    bool switch_success;
} WorkerReport;

int main(int argc, char* argv[]) {
    int nodes_to_test = 2;
    int workers_to_test = 4;

    if (argc > 1) nodes_to_test = atoi(argv[1]);
    if (argc > 2) workers_to_test = atoi(argv[2]);

    num_numa_nodes = nodes_to_test;

    printf("====================================================================\n");
    printf("     NUMA Memory-View Sharing & Process Isolation Test Suite         \n");
    printf("====================================================================\n");
    printf("Physical NUMA Slabs: %d (Exactly %.2f MB)\n", 
           num_numa_nodes, (double)(num_numa_nodes * 2097152) / (1024.0 * 1024.0));
    printf("Total Workers:       %d (%d workers per NUMA node)\n", 
           workers_to_test, workers_to_test / num_numa_nodes);

    int n_layers = 3;
    int neurons[] = {784, 64, 10};

    NeuralNet* base_nn = setup_numa_model(n_layers, neurons);
    if (!base_nn) {
        fprintf(stderr, "[ERROR] setup_numa_model failed!\n");
        return 1;
    }

    size_t slab_size = global_aligned_slab_size;
    printf("[PARENT] Virtual Base:      %p\n", (void*)base_nn);
    printf("[PARENT] Total Shared Mmap: %zu bytes (%.2f MB)\n", 
           global_mmap_total_size, (double)global_mmap_total_size / (1024.0 * 1024.0));
    printf("--------------------------------------------------------------------\n");

    WorkerReport* reports = mmap(NULL, workers_to_test * sizeof(WorkerReport),
                                 PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    for (int rank = 0; rank < workers_to_test; rank++) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork failed");
            exit(EXIT_FAILURE);
        }

        if (pid == 0) {
            int my_node = rank % num_numa_nodes;
            pin_process_to_node(rank, my_node);

            reports[rank].rank           = rank;
            reports[rank].target_node    = my_node;
            reports[rank].switch_success = false;

            // 1. Activate shared NUMA view for this socket
            if (activate_numa_view(my_node, base_nn, slab_size) < 0) {
                fprintf(stderr, "[WORKER %d] FAILED to execute PDE switch!\n", rank);
                exit(1);
            }

            // 2. Setup transparent single-struct view
            struct NeuralNet local_nn;
            local_nn.magic_test_value    = base_nn->magic_test_value;
            local_nn.n_layers            = base_nn->n_layers;
            local_nn.n_neurons_per_layer = base_nn->n_neurons_per_layer;
            local_nn.w                   = base_nn->w; 
            local_nn.b                   = base_nn->b; 

            // 3. Initialize process-private working buffers
            init_process_working_buffers(&local_nn);

            double post_canary = local_nn.magic_test_value;
            double expected_canary = 100.0 * (double)(my_node + 1);

            double test_val = 1000.0 + (double)rank * 50.0 + (double)my_node;
            local_nn.out[0][1] = test_val;

            usleep(25000);

            reports[rank].canary_read        = post_canary;
            reports[rank].expected_canary    = expected_canary;
            reports[rank].local_write_val    = local_nn.out[0][1];
            reports[rank].expected_write_val = test_val;
            reports[rank].neuron_dim_0       = local_nn.n_neurons_per_layer[0];
            reports[rank].switch_success     = true;

            printf("[WORKER %2d | Node %d] Shared Canary: %.2f | Private out[0][1]: %.2f\n", 
                   rank, my_node, post_canary, reports[rank].local_write_val);

            free_process_working_buffers(&local_nn);
            exit(0);
        }
    }

    for (int i = 0; i < workers_to_test; i++) {
        wait(NULL);
    }

    printf("\n====================================================================\n");
    printf("                    VERIFICATION SUMMARY MATRIX                     \n");
    printf("====================================================================\n");
    printf(" Rank | Node | Shared Canary (Expected) | Private Write (Expected) | Status \n");
    printf("------+------+--------------------------+--------------------------+--------\n");

    bool all_passed = true;

    for (int r = 0; r < workers_to_test; r++) {
        bool canary_ok = (fabs(reports[r].canary_read - reports[r].expected_canary) < 1e-4);
        bool write_ok  = (fabs(reports[r].local_write_val - reports[r].expected_write_val) < 1e-4);
        bool ptr_ok    = (reports[r].neuron_dim_0 == 784);
        bool pass      = canary_ok && write_ok && ptr_ok && reports[r].switch_success;

        if (!pass) all_passed = false;

        printf("  %2d  |  %2d  | %7.2f (%7.2f)        | %7.2f (%7.2f)        |  %s  \n",
               reports[r].rank,
               reports[r].target_node,
               reports[r].canary_read, reports[r].expected_canary,
               reports[r].local_write_val, reports[r].expected_write_val,
               pass ? "PASS" : "FAIL");
    }

    printf("====================================================================\n");
    if (all_passed) {
        printf(">>> SUITE RESULT: SUCCESS (%d NUMA Slabs, %d Workers Isolated) <<<\n", 
               nodes_to_test, workers_to_test);
    } else {
        printf(">>> SUITE RESULT: FAILED <<<\n");
    }
    printf("====================================================================\n");

    munmap(reports, workers_to_test * sizeof(WorkerReport));
    unmap_nn();

    return all_passed ? 0 : 1;
}