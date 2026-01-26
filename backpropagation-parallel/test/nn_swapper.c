#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <stdint.h>
#include <fcntl.h>
#include <time.h>

#include "neural_network.h"
#include "memory_manager.h"
#include "read_data.h"
#include "utils.h"

#define PES 156 
#define PDE_SIZE (2 * 1024 * 1024)

// Syscall wrapper
int pes(unsigned long vaddr, int node_a, int node_b) {
    return syscall(PES, vaddr, node_a, node_b);
}


int main(int argc, char** argv) {
    srand(time(NULL));

    // 1. ARCHITECTURE & HYPERPARAMETERS
    int n_processes = 2;
    int n_layers = 4;
    int n_neurons_per_layer[] = {784, 64, 32, 10};

    double learning_rate = 1e-4;
    double init_lr = 1e-4;
    char* activation_fun = "relu";
    char* loss = "ce";
    char* opt = "adam";
    int num_samples_to_train = 10000;
    int epochs = 5;

    // CLI Parsing
    if (argc > 1) learning_rate = atof(argv[1]);
    if (argc > 2) init_lr = atof(argv[2]);
    if (argc > 3) activation_fun = argv[3];
    if (argc > 4) loss = argv[4];
    if (argc > 5) opt = argv[5];
    if (argc > 6) num_samples_to_train = atoi(argv[6]);
    if (argc > 7) epochs = atoi(argv[7]);
    if (argc > 8) n_processes = atoi(argv[8]);

    // 2. DATA ALLOCATION
    double** X_train = malloc(N_SAMPLES * sizeof(double*));
    double** y_train = malloc(N_SAMPLES * sizeof(double*));
    double* y_train_temp = malloc(N_SAMPLES * sizeof(double));
    for(int i=0; i<N_SAMPLES; i++) {
        X_train[i] = malloc(N_DIMS * sizeof(double));
        y_train[i] = malloc(N_CLASSES * sizeof(double));
    }

    printf("Loading dataset...\n");
    read_csv_file(X_train, y_train_temp, y_train, "train");
    scale_data(X_train, "train");

    // 3. MANUAL NUMA ALLOCATION (Matches your swap test)
    printf("=== STEP 1: NUMA-AWARE ALLOCATION ===\n");
    // n_neurons_per_layer fixed here
    struct NeuralNet* n0 = newNetSingleAlloc(n_layers, n_neurons_per_layer);
    if (n0 == NULL) return EXIT_FAILURE;
    
    struct NeuralNet* n1 = (struct NeuralNet*)((char*)n0 + PDE_SIZE);

    init_nn(n0);
    init_nn(n1);

    n0->magic_test_value = 123.456;
    n1->magic_test_value = 999.888;

 // 4. PES SWAP TEST
    /*printf("\n=== BEFORE SWAP ===\n");
    printf("N0 (Vaddr %p): Magic=%f\n", (void*)n0, n0->magic_test_value);
    printf("N1 (Vaddr %p): Magic=%f\n", (void*)n1, n1->magic_test_value);

    printf("\n[!!!] Calling PES Syscall to swap Physical Node 0 and Node 1...\n");
    // This tells the kernel to swap the physical pages backing the 2MB region starting at n0
    pes((unsigned long)n0, 0, 1); 

    printf("\n=== AFTER SWAP ===\n");
    // If the swap worked, n0's virtual address now looks at n1's old physical data
    printf("N0 (Vaddr %p): Magic=%f (Expected: 999.888)\n", (void*)n0, n0->magic_test_value);
    printf("N1 (Vaddr %p): Magic=%f (Expected: 123.456)\n", (void*)n1, n1->magic_test_value);

    if (n0->magic_test_value == 999.888) {
        printf("\nSUCCESS: Physical pages swapped successfully!\n");
    } else {
        printf("\nFAILURE: Magic values did not change. Check kernel logs (dmesg).\n");
    }

    // Resetting the state before the workers start
    printf("\nResetting swap to original state...\n");
    pes((unsigned long)n0, 1, 0);*/


    // 5. WORKER LOOP
    int samples_per_process = num_samples_to_train / n_processes;
    
    for (int i = 0; i < n_processes; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            // Select clone based on process index
            // i=0,2,4... use n0 | i=1,3,5... use n1
            struct NeuralNet* shared_nn = (i % 2 == 0) ? n0 : n1;
            int current_node = (i % 2);
          
            int start = i * samples_per_process;
            int end = (i == n_processes - 1) ? num_samples_to_train : (i + 1) * samples_per_process;
            
            NNWorker worker;
            setup_worker(&worker, shared_nn, i, start, end);

            // 1. WARM UP: Touch the memory to ensure Page Tables are populated
            // This ensures the 'present' bit is set so the kernel doesn't skip the swap
            volatile double warm0 = n0->magic_test_value;
            volatile double warm1 = n1->magic_test_value;
            (void)warm0; 
            (void)warm1;

            printf("[Child %d] Before PES: %f\n", i, n0->magic_test_value);

            // 2. The Swap
            int ret = pes((unsigned long)n0, 0, 1);
            if (ret != 0) perror("PES syscall failed");

            // 3. Read back
            volatile double after = n0->magic_test_value;
            printf("[Child %d] After PES: %f\n", i, after);

            // 4. Remove the immediate swap back for this test
            // Let's see if it holds the value without the race condition
            sleep(1);

            pes((unsigned long) n0, 1, 0);

            printf("[Child %d] After Last PES: %f\n", i, n0->magic_test_value);

            worker.private_delta[0][0] = (double)getpid();
            printf("Child %d | Samples [%d-%d] | PID-based Delta: %.0f\n", 
                   i, start, end, worker.private_delta[0][0]);

            exit(0);
        }
    }

    // Parent waits
    for (int i = 0; i < n_processes; i++) {
        wait(NULL);
    }

    printf("All processes reconciled.\n");

    printf("Parent: n0->delta[0][0] is %f (Should be 0.0 or original)\n", n0->delta[0][0]);


    return 0;
}