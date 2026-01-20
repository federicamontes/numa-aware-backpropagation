#define _GNU_SOURCE  
#include "utils.h"
#include <stdlib.h>
#include <numa.h>
#include <unistd.h>


int seed; // Actual definition

double randn() {
    int a = 1103515245;
    int m = 2147483647;
    int c = 12345;
    seed = (a * seed + c) % m;
    return (double)seed / (double)m;
}

void shuffle(int* arr, size_t n) {
    if (n > 1) {
        for (size_t i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = arr[j];
            arr[j] = arr[i];
            arr[i] = t;
        }
    }
}


/**
 * Pins the calling process to all CPU cores associated with a specific NUMA node.
 * @param node_id: The ID of the NUMA node (0, 1, etc.)
 * @return: 0 on success, -1 on failure
 */
int pin_to_numa_node(int node_id) {
    cpu_set_t mask;
    CPU_ZERO(&mask);

    // Get the bitmask of CPUs for this NUMA node from libnuma
    struct bitmask* cpus = numa_allocate_cpumask();
    if (numa_node_to_cpus(node_id, cpus) != 0) {
        perror("numa_node_to_cpus");
        numa_free_cpumask(cpus);
        return -1;
    }

    // Transfer the bits from the bitmask to the cpu_set_t
    for (unsigned int i = 0; i < cpus->size; i++) {
        if (numa_bitmask_isbitset(cpus, i)) {
            CPU_SET(i, &mask);
        }
    }

    // Apply the affinity to the calling process (pid 0)
    if (sched_setaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
        perror("sched_setaffinity");
        numa_free_cpumask(cpus);
        return -1;
    }

    numa_free_cpumask(cpus);
    return 0;
}