#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include "neural_network.h"
#include "memory_manager.h"
#include <stdint.h>
#include <fcntl.h>

#define PES 156 
#define PDE_SIZE (2 * 1024 * 1024)

int pes(unsigned long vaddr, int node_a, int node_b) {
    return syscall(PES, vaddr, node_a, node_b);
}

void print_physical_address(const char* label, void* vaddr) {
    uintptr_t virtual_addr = (uintptr_t)vaddr;
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) {
        perror("pagemap open failed (run with sudo)");
        return;
    }

    off_t offset = (virtual_addr / 4096) * 8;
    uint64_t entry;
    
    if (pread(fd, &entry, sizeof(entry), offset) != sizeof(entry)) {
        close(fd);
        return;
    }
    close(fd);

    if (entry & (1ULL << 63)) {
        uint64_t pfn = entry & ((1ULL << 55) - 1);
        printf("%s: Virtual %p -> Physical PFN: 0x%lx\n", label, vaddr, pfn);
    } else {
        printf("%s: Virtual %p -> Page not present in RAM\n", label, vaddr);
    }
}

int main(int argc, char** argv) {
    int n_layers = 4;
    int n_neurons[] = {784, 64, 32, 10};
    
    printf("=== STEP 1: NUMA-AWARE ALLOCATION ===\n");
    struct NeuralNet* n0 = newNetSingleAlloc(n_layers, n_neurons);
    struct NeuralNet* n1 = (struct NeuralNet*)((char*)n0 + PDE_SIZE);

    if (n0 == NULL) return EXIT_FAILURE;

    // Initialize both networks
    init_nn(n0);
    init_nn(n1);

    // Set unique identifiers in the DIRECT struct memory
    // These are values, not pointers, so they MUST move when the page moves
    n0->magic_test_value = 123.456;
    n1->magic_test_value = 999.888;

    // Set weight values
    n0->w[0][0][0] = 1.111;
    n1->w[0][0][0] = 2.222;

    printf("\n=== BEFORE SWAP ===\n");
    printf("N0 (Vaddr %p): Magic=%f, Weight=%f\n", (void*)n0, n0->magic_test_value, n0->w[0][0][0]);
    printf("N1 (Vaddr %p): Magic=%f, Weight=%f\n", (void*)n1, n1->magic_test_value, n1->w[0][0][0]);
    
    print_physical_address("PFN N0", n0);
    print_physical_address("PFN N1", n1);

    printf("\n[!!!] Calling PES Syscall to swap PDE 0 and PDE 1...\n");
    pes((unsigned long)n0, 0, 1); 

    printf("\n=== AFTER SWAP ===\n");
    // Magic values WILL swap because they are plain data in the struct
    // Weights WILL NOT change their local value because the pointers were swapped too!
    printf("N0 (Vaddr %p): Magic=%f (Should be 999.888), Weight=%f\n", (void*)n0, n0->magic_test_value, n0->w[0][0][0]);
    printf("N1 (Vaddr %p): Magic=%f (Should be 123.456), Weight=%f\n", (void*)n1, n1->magic_test_value, n1->w[0][0][0]);

    print_physical_address("PFN N0", n0);
    print_physical_address("PFN N1", n1);

    if (n0->magic_test_value == 999.888) {
        printf("\nSUCCESS: Physical pages swapped! Node 0 virtual address now points to Node 1's physical RAM.\n");
    } else {
        printf("\nFAILURE: Magic values did not swap.\n");
    }

    // Reset for safety
    printf("\nResetting swap...\n");
    pes((unsigned long)n0, 1, 0);

    return 0;
}