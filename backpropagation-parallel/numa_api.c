#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include <numaif.h>
#include <numa.h>
#include <sys/sysinfo.h>
#include <sys/mman.h>
#include <sched.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/syscall.h>

#include "neural_network.h"
#include "activation.h"

#include "ipc_lock.h"


unsigned long base_address = 0x1000000000;
void* global_mmap_start = NULL;
size_t global_mmap_total_size = 0;
size_t global_aligned_slab_size = 0;
int num_numa_nodes = 1;

#define DEBUG if(0)
#define PES 156

static inline int pes_syscall(unsigned long vaddr, int index_a, int index_b) {
    return (int)syscall(PES, vaddr, index_a, index_b);
}

#if defined(ATOMIC_UPDATE)
/**
 * Atomic Fetch-And-Subtract for 64-bit double precision floats.
 * Implemented via a lock-free 64-bit CAS (Compare-And-Swap) loop.
 *
 * @param target Pointer to the double in shared memory.
 * @param delta  Value to subtract (*target -= delta).
 * @return The old value before subtraction.
 */
static inline double atomic_sub_double(double *target, double delta) {
    union {
        double   d;
        uint64_t u;
    } old_val, new_val;

    // Load current value as 64-bit integer
    old_val.u = __atomic_load_n((uint64_t*)target, __ATOMIC_RELAXED);

    do {
        // Compute subtraction using standard floating-point arithmetic
        new_val.d = old_val.d - delta;

        // Atomically replace *target if its memory still matches old_val.u.
        // If another core modified *target in the meantime, CAS fails,
        // reloads the new value into old_val.u, and retries the loop.
    } while (!__atomic_compare_exchange_n((uint64_t*)target, 
                                          &old_val.u, 
                                          new_val.u, 
                                          true,                // Weak CAS for loop efficiency
                                          __ATOMIC_RELEASE,   // Publish updated weight
                                          __ATOMIC_RELAXED)); // Relaxed on reload failure

    return old_val.d;
}
#endif

#if defined(ATOMIC_UPDATE)
    // Mode 3: Atomic Lock-Free FAD (Orders updates per float without global locks)
    #define APPLY_REPLICA_DELTA(target_ptr, delta) atomic_sub_double((target_ptr), (delta))
#else
    // Mode 1: Default Asynchronous (Hogwild! style, direct write)
    #define APPLY_REPLICA_DELTA(target_ptr, delta) (*(target_ptr) -= (delta))
#endif

#define UPDATE_REPLICAS_GLOBAL(local_addr, delta_val, slab_size, num_nodes) do {       \
    uintptr_t _l_addr = (uintptr_t)(local_addr);                                      \
    uintptr_t _base   = (uintptr_t)global_mmap_start;                                \
    size_t    _tot    = global_mmap_total_size;                                       \
    if (_base && _l_addr >= _base && _l_addr < _base + _tot) {                        \
        uintptr_t _offset = (_l_addr - _base) % (slab_size);                          \
        for (int _n = 0; _n < (num_nodes); _n++) {                                    \
            uintptr_t _target_addr = _base + ((uintptr_t)_n * (slab_size)) + _offset; \
            double* _target = (double*)_target_addr;                                  \
            APPLY_REPLICA_DELTA(_target, delta_val);                                  \
        }                                                                             \
    } else {                                                                          \
        APPLY_REPLICA_DELTA((double*)_l_addr, delta_val);                             \
    }                                                                                 \
} while (0)

// ============================================================================
// init_node_ipc_context
// Allocates NodeIPCContext and shared reduction buffers in a single mmap region.
// Configures process-shared attributes for mutex and barriers.
// ============================================================================
NodeIPCContext* init_node_ipc_context(int n_workers, int n_layers, const int n_neurons_per_layer[], int total_samples) {
    // 1. Calculate reduction buffer footprint
    size_t size = sizeof(NodeIPCContext);
    size = ALIGN_CACHELINE(size);

    // Shuffle index buffer
    size += total_samples * sizeof(int);
    size = ALIGN_CACHELINE(size);

    // Top-level pointers for shared_dw and shared_db
    size += (size_t)(n_layers - 1) * sizeof(double**); // shared_dw
    size += (size_t)(n_layers - 1) * sizeof(double*);   // shared_db
    size = ALIGN_CACHELINE(size);

    // Row pointers for shared_dw
    for (int i = 0; i < n_layers - 1; i++) {
        size_t n_in = (size_t)(n_neurons_per_layer[i] + 1);
        size += n_in * sizeof(double*);
    }
    size = ALIGN_CACHELINE(size);

    // Flat data buffers for shared_db
    for (int i = 0; i < n_layers - 1; i++) {
        size_t n_out = (size_t)(n_neurons_per_layer[i + 1] + 1);
        size += ALIGN_CACHELINE(n_out * sizeof(double));
    }

    // Flat data buffers for shared_dw
    for (int i = 0; i < n_layers - 1; i++) {
        size_t n_in  = (size_t)(n_neurons_per_layer[i] + 1);
        size_t n_out = (size_t)(n_neurons_per_layer[i + 1] + 1);
        size += ALIGN_CACHELINE(n_in * n_out * sizeof(double));
    }

    size_t total_alloc = ALIGN_PAGE(size);

    // 2. Allocate shared memory region
    void* mem = mmap(NULL, total_alloc, 
                     PROT_READ | PROT_WRITE, 
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        perror("[IPC-INIT] mmap failed for NodeIPCContext");
        exit(EXIT_FAILURE);
    }
    memset(mem, 0, total_alloc);

    NodeIPCContext* ipc = (NodeIPCContext*)mem;
    ipc->total_mmap_bytes = total_alloc;
    ipc->total_samples = total_samples;

    char* curr = (char*)mem + sizeof(NodeIPCContext);
    curr = ALIGN_CACHELINE_PTR(curr);

    // Bind shared shuffle array
    ipc->shared_sample_order = (int*)curr;
    curr += (total_samples * sizeof(int));
    curr = ALIGN_CACHELINE_PTR(curr);

    // Bind top-level pointer tables
    ipc->shared_dw = (double***)curr; curr += (n_layers - 1) * sizeof(double**);
    ipc->shared_db = (double**)curr;  curr += (n_layers - 1) * sizeof(double*);
    curr = ALIGN_CACHELINE_PTR(curr);

    // Bind row pointer tables for shared_dw
    for (int i = 0; i < n_layers - 1; i++) {
        int n_in = n_neurons_per_layer[i] + 1;
        ipc->shared_dw[i] = (double**)curr;
        curr += n_in * sizeof(double*);
    }
    curr = ALIGN_CACHELINE_PTR(curr);

    // Bind flat reduction buffers for shared_db
    for (int i = 0; i < n_layers - 1; i++) {
        int n_out = n_neurons_per_layer[i + 1] + 1;
        ipc->shared_db[i] = (double*)curr;
        curr += ALIGN_CACHELINE(n_out * sizeof(double));
    }

    // Bind flat reduction buffers for shared_dw
    for (int i = 0; i < n_layers - 1; i++) {
        int n_in  = n_neurons_per_layer[i] + 1;
        int n_out = n_neurons_per_layer[i + 1] + 1;
        for (int j = 0; j < n_in; j++) {
            ipc->shared_dw[i][j] = (double*)curr;
            curr += ALIGN_CACHELINE(n_out * sizeof(double));
        }
    }

    // 3. Initialize Process-Shared Mutex
    pthread_mutexattr_t m_attr;
    pthread_mutexattr_init(&m_attr);
    pthread_mutexattr_setpshared(&m_attr, PTHREAD_PROCESS_SHARED);
    if (pthread_mutex_init(&ipc->lock, &m_attr) != 0) {
        perror("[IPC-INIT] pthread_mutex_init failed");
        exit(EXIT_FAILURE);
    }
    pthread_mutexattr_destroy(&m_attr);

    // 4. Initialize Process-Shared Barriers
    pthread_barrierattr_t b_attr;
    pthread_barrierattr_init(&b_attr);
    pthread_barrierattr_setpshared(&b_attr, PTHREAD_PROCESS_SHARED);
    if (pthread_barrier_init(&ipc->barrier_accumulate, &b_attr, n_workers) != 0 ||
        pthread_barrier_init(&ipc->barrier_update, &b_attr, n_workers) != 0) {
        perror("[IPC-INIT] pthread_barrier_init failed");
        exit(EXIT_FAILURE);
    }
    pthread_barrierattr_destroy(&b_attr);

    return ipc;
}

// ============================================================================
// destroy_node_ipc_context
// Destroys synchronization primitives and unmaps shared memory.
// ============================================================================
void destroy_node_ipc_context(NodeIPCContext* ipc) {
    if (!ipc) return;
    pthread_mutex_destroy(&ipc->lock);
    pthread_barrier_destroy(&ipc->barrier_accumulate);
    pthread_barrier_destroy(&ipc->barrier_update);
    munmap(ipc, ipc->total_mmap_bytes);
}


size_t calculate_shared_slab_size(int n_layers, const int n_neurons_per_layer[]) {
    size_t size = 0;

    size += sizeof(struct NeuralNet);
    size = ALIGN_CACHELINE(size);
    size += (size_t)n_layers * sizeof(int);
    size = ALIGN_CACHELINE(size);

    size += (size_t)(n_layers - 1) * 3 * sizeof(double***);
    size += (size_t)(n_layers - 1) * 3 * sizeof(double*);
    size = ALIGN_CACHELINE(size);

    for (int i = 0; i < n_layers - 1; i++) {
        size_t n_in = (size_t)(n_neurons_per_layer[i] + 1);
        size += n_in * 3 * sizeof(double*);
    }
    size = ALIGN_CACHELINE(size);

    for (int i = 0; i < n_layers - 1; i++) {
        size_t n_out = (size_t)(n_neurons_per_layer[i+1] + 1);
        size += ALIGN_CACHELINE(n_out * 3 * sizeof(double));
    }

    for (int i = 0; i < n_layers - 1; i++) {
        size_t n_in  = (size_t)(n_neurons_per_layer[i] + 1);
        size_t n_out = (size_t)(n_neurons_per_layer[i+1] + 1);
        size += ALIGN_CACHELINE(n_in * n_out * 3 * sizeof(double));
    }

    return size;
}

// ============================================================================
// detect_numa_topology
// Dynamically discovers the hardware NUMA topology from the Linux kernel.
// ============================================================================
void detect_numa_topology(void) {
    // Only auto-detect if not already explicitly set (e.g., by test_pes emulation)
    if (num_numa_nodes <= 1) {
        if (numa_available() >= 0) {
            int detected = numa_num_configured_nodes();
            if (detected > 1) {
                num_numa_nodes = detected;
            } else {
                num_numa_nodes = 1;
            }
        } else {
            num_numa_nodes = 1;
        }
    }
}

void bind_memory_to_numa_node(void* addr, size_t size, int node) {
    int max_node = numa_max_node();
    if (max_node < 0 || node > max_node) return;

    unsigned long nodemask = (1UL << node);
    // MPOL_MF_MOVE ensures the kernel moves any pre-faulted pages to the target NUMA socket
    if (mbind(addr, size, MPOL_BIND, &nodemask, sizeof(unsigned long) * 8, MPOL_MF_STRICT | MPOL_MF_MOVE) < 0) {
        perror("[NUMA] mbind fallback");
    } else {
        printf("[NUMA] Memory range %p-%p bound to NUMA Node %d\n", addr, (char*)addr + size, node);
    }
}

void* mmap_alloc(size_t size) {
    size_t aligned_size = ALIGN_PAGE(size);
    void* addr = (void*)base_address;

    if ((uintptr_t)addr % PDE_ALIGN_SIZE != 0) {
        fprintf(stderr, "FATAL: base_address is not 2MB aligned!\n");
        exit(EXIT_FAILURE);
    }

    void* ret = mmap(addr, aligned_size,
                     PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE | MAP_POPULATE,
                     -1, 0);

    if (ret == MAP_FAILED) {
        perror("mmap fixed allocation failed");
        exit(EXIT_FAILURE);
    }

    return ret;
}

NeuralNet* newNetSharedAlloc(int n_layers, int n_neurons_per_layer[], size_t aligned_slab_size) {
    size_t total_map_size = (size_t)num_numa_nodes * aligned_slab_size;
    void* mmap_block = mmap_alloc(total_map_size);

    global_mmap_start = mmap_block;
    global_mmap_total_size = total_map_size;
    global_aligned_slab_size = aligned_slab_size;

    char* virtual_base = (char*)mmap_block;

    for (int numa_node = 0; numa_node < num_numa_nodes; numa_node++) {
        char* net_addr = (char*)mmap_block + (numa_node * aligned_slab_size);
        bind_memory_to_numa_node(net_addr, aligned_slab_size, numa_node);

        for (size_t i = 0; i < aligned_slab_size; i += PAGE_SIZE) {
            *((volatile char*)(net_addr + i)) = 0;
        }

        struct NeuralNet* nn = (struct NeuralNet*)net_addr;
        char* curr = net_addr;

        #define V_ADDR (virtual_base + (curr - net_addr))

        nn->magic_test_value = 100.0 * (double)(numa_node + 1);
        nn->n_layers = n_layers;
        curr += sizeof(struct NeuralNet);

        curr = ALIGN_CACHELINE_PTR(curr);
        nn->n_neurons_per_layer = (int*)V_ADDR;
        for (int i = 0; i < n_layers; i++) {
            ((int*)curr)[i] = n_neurons_per_layer[i];
        }
        curr += (n_layers * sizeof(int));

        curr = ALIGN_CACHELINE_PTR(curr);
        nn->w           = (double***)V_ADDR; curr += (n_layers - 1) * sizeof(double**);
        nn->momentum_w  = (double***)V_ADDR; curr += (n_layers - 1) * sizeof(double**);
        nn->momentum2_w = (double***)V_ADDR; curr += (n_layers - 1) * sizeof(double**);

        nn->b           = (double**)V_ADDR;  curr += (n_layers - 1) * sizeof(double*);
        nn->momentum_b  = (double**)V_ADDR;  curr += (n_layers - 1) * sizeof(double*);
        nn->momentum2_b = (double**)V_ADDR;  curr += (n_layers - 1) * sizeof(double*);

        curr = ALIGN_CACHELINE_PTR(curr);
        for (int i = 0; i < n_layers - 1; i++) {
            int n_in = n_neurons_per_layer[i] + 1;
            double*** w_phys   = (double***)(net_addr + ((char*)nn->w - virtual_base));
            double*** mw_phys  = (double***)(net_addr + ((char*)nn->momentum_w - virtual_base));
            double*** mw2_phys = (double***)(net_addr + ((char*)nn->momentum2_w - virtual_base));

            w_phys[i]   = (double**)V_ADDR; curr += n_in * sizeof(double*);
            mw_phys[i]  = (double**)V_ADDR; curr += n_in * sizeof(double*);
            mw2_phys[i] = (double**)V_ADDR; curr += n_in * sizeof(double*);
        }

        curr = ALIGN_CACHELINE_PTR(curr);
        for (int i = 0; i < n_layers - 1; i++) {
            int n_out = n_neurons_per_layer[i+1] + 1;
            double** b_phys   = (double**)(net_addr + ((char*)nn->b - virtual_base));
            double** mb_phys  = (double**)(net_addr + ((char*)nn->momentum_b - virtual_base));
            double** mb2_phys = (double**)(net_addr + ((char*)nn->momentum2_b - virtual_base));

            b_phys[i]   = (double*)V_ADDR; curr += ALIGN_CACHELINE(n_out * sizeof(double));
            mb_phys[i]  = (double*)V_ADDR; curr += ALIGN_CACHELINE(n_out * sizeof(double));
            mb2_phys[i] = (double*)V_ADDR; curr += ALIGN_CACHELINE(n_out * sizeof(double));
        }

        for (int i = 0; i < n_layers - 1; i++) {
            int n_in  = n_neurons_per_layer[i] + 1;
            int n_out = n_neurons_per_layer[i+1] + 1;

            double*** w_phys   = (double***)(net_addr + ((char*)nn->w - virtual_base));
            double*** mw_phys  = (double***)(net_addr + ((char*)nn->momentum_w - virtual_base));
            double*** mw2_phys = (double***)(net_addr + ((char*)nn->momentum2_w - virtual_base));

            double** w_rows   = (double**)(net_addr + ((char*)w_phys[i] - virtual_base));
            double** mw_rows  = (double**)(net_addr + ((char*)mw_phys[i] - virtual_base));
            double** mw2_rows = (double**)(net_addr + ((char*)mw2_phys[i] - virtual_base));

            for (int j = 0; j < n_in; j++) {
                w_rows[j]   = (double*)V_ADDR; curr += ALIGN_CACHELINE(n_out * sizeof(double));
                mw_rows[j]  = (double*)V_ADDR; curr += ALIGN_CACHELINE(n_out * sizeof(double));
                mw2_rows[j] = (double*)V_ADDR; curr += ALIGN_CACHELINE(n_out * sizeof(double));
            }
        }

        #undef V_ADDR
    }

    return (struct NeuralNet*)mmap_block;
}

void unmap_nn(void) {
    if (global_mmap_start != NULL) {
        if (munmap(global_mmap_start, global_mmap_total_size) < 0) {
            perror("munmap failed");
        } else {
            printf("[NUMA] Memory unmapped successfully.\n");
            global_mmap_start = NULL;
        }
    }
}

void init_nn_shared(struct NeuralNet* nn) {
    for (int k = 0; k < nn->n_layers - 1; k++) {
        int n_in  = nn->n_neurons_per_layer[k] + 1;
        int n_out = nn->n_neurons_per_layer[k+1] + 1;

        for (int j = 0; j < n_out; j++) {
            nn->b[k][j] = 0.0;
            nn->momentum_b[k][j] = 0.0;
            nn->momentum2_b[k][j] = 0.0;
        }

        for (int i = 0; i < n_in; i++) {
            for (int j = 0; j < n_out; j++) {
                if (i >= 1 && j >= 1) {
                    nn->w[k][i][j] = randn();
                } else {
                    nn->w[k][i][j] = 0.0;
                }
                nn->momentum_w[k][i][j] = 0.0;
                nn->momentum2_w[k][i][j] = 0.0;
            }
        }
    }
}

struct NeuralNet* setup_numa_model(int n_layers, int n_neurons_per_layer[]) {

    // Ensure hardware nodes are discovered
    detect_numa_topology();

    size_t raw_slab_size = calculate_shared_slab_size(n_layers, n_neurons_per_layer);
    size_t aligned_size  = ALIGN_PAGE(raw_slab_size);

    DEBUG printf("[NUMA-SETUP] NUMA Sockets: %d | Total Mmap: %.2f MB\n", 
                 num_numa_nodes, (double)(aligned_size * num_numa_nodes) / (1024.0 * 1024.0));

    NeuralNet* nn_mapped = newNetSharedAlloc(n_layers, n_neurons_per_layer, aligned_size);

    for (int i = 0; i < num_numa_nodes; i++) {
        NeuralNet *nn = (NeuralNet*)((char*)nn_mapped + i * aligned_size);
        nn->magic_test_value = 100.0 * (double)(i + 1);
        nn->n_layers = n_layers;
        seed = 12345; //must match seed in main.c
        init_nn_shared(nn);
    }

    return nn_mapped;
}

void init_process_working_buffers(struct NeuralNet* nn) {
    int n_layers = nn->n_layers;

    nn->dw    = malloc((n_layers - 1) * sizeof(double**));
    nn->db    = malloc((n_layers - 1) * sizeof(double*));
    nn->in    = malloc(n_layers * sizeof(double*));
    nn->out   = malloc(n_layers * sizeof(double*));
    nn->delta = malloc(n_layers * sizeof(double*));

    for (int i = 0; i < n_layers - 1; i++) {
        int n_in  = nn->n_neurons_per_layer[i] + 1;
        int n_out = nn->n_neurons_per_layer[i+1] + 1;

        nn->dw[i] = malloc(n_in * sizeof(double*));
        nn->db[i] = calloc(n_out, sizeof(double));

        for (int j = 0; j < n_in; j++) {
            nn->dw[i][j] = calloc(n_out, sizeof(double));
        }
    }

    for (int i = 0; i < n_layers; i++) {
        int n = nn->n_neurons_per_layer[i] + 1;
        nn->in[i]    = calloc(n, sizeof(double));
        nn->out[i]   = calloc(n, sizeof(double));
        nn->delta[i] = calloc(n, sizeof(double));
    }

    nn->targets = calloc(nn->n_neurons_per_layer[n_layers - 1] + 1, sizeof(double));
}

void free_process_working_buffers(struct NeuralNet* nn) {
    for (int i = 0; i < nn->n_layers - 1; i++) {
        int n_in = nn->n_neurons_per_layer[i] + 1;
        for (int j = 0; j < n_in; j++) {
            free(nn->dw[i][j]);
        }
        free(nn->dw[i]);
        free(nn->db[i]);
    }
    free(nn->dw);
    free(nn->db);

    for (int i = 0; i < nn->n_layers; i++) {
        free(nn->in[i]);
        free(nn->out[i]);
        free(nn->delta[i]);
    }
    free(nn->in);
    free(nn->out);
    free(nn->delta);
    free(nn->targets);
}

int activate_numa_view(int numa_node_id, void* base_ptr, size_t base_nn_size) {
    if (base_ptr == NULL || (uintptr_t)base_ptr % PDE_ALIGN_SIZE != 0) {
        fprintf(stderr, "[NUMA-ACTIVATE] Invalid base address %p\n", base_ptr);
        return -1;
    }

    int pde_index_target = (numa_node_id * (int)base_nn_size) / PDE_ALIGN_SIZE;

    // Node 0 already maps to physical frame 0 by default mmap
    if (pde_index_target == 0) {
        return 0;
    }

    volatile char* touch_ptr = (char*)base_ptr;
    for (size_t offset = 0; offset < base_nn_size; offset += PAGE_SIZE) {
        touch_ptr[offset] = touch_ptr[offset];
        touch_ptr[(pde_index_target * base_nn_size) + offset] = 
            touch_ptr[(pde_index_target * base_nn_size) + offset];
    }

    int ret = pes_syscall((unsigned long)base_ptr, 0, pde_index_target);
    if (ret < 0) {
        perror("[NUMA-ACTIVATE] pes_syscall failed");
        return -1;
    }

    return 0;
}


// ============================================================================
// accumulate_gradients_locked
// Invoked while holding ipc->lock. Accumulates private worker gradients into
// the shared reduction buffer and resets private accumulators to 0.0.
// ============================================================================
void accumulate_gradients_locked(NodeIPCContext* ipc, struct NeuralNet* local_nn) {
    for (int k = 0; k < local_nn->n_layers - 1; k++) {
        int n_in  = local_nn->n_neurons_per_layer[k];
        int n_out = local_nn->n_neurons_per_layer[k + 1];

        for (int j = 1; j <= n_out; j++) {
            ipc->shared_db[k][j] += local_nn->db[k][j];
            local_nn->db[k][j] = 0.0;
        }

        for (int i = 1; i <= n_in; i++) {
            for (int j = 1; j <= n_out; j++) {
                ipc->shared_dw[k][i][j] += local_nn->dw[k][i][j];
                local_nn->dw[k][i][j] = 0.0;
            }
        }
    }
}

// ============================================================================
// update_weights_from_shared
// Executed strictly by Worker 0 between Barrier 1 and Barrier 2.
// Normalizes accumulated gradients by the total mini-batch size, executes the
// optimizer step, and clears the shared reduction buffers to 0.0.
// ============================================================================
static void update_weights_from_shared(struct NeuralNet* nn, NodeIPCContext* ipc, 
                                      double lr, char* opt, int itr, int cur_batch) {
    double inv_batch = 1.0 / (double)cur_batch;
    int is_adam = (strcmp(opt, "adam") == 0);
    int is_sgd  = (strcmp(opt, "sgd") == 0);

    double bias_corr1 = 1.0;
    double bias_corr2 = 1.0;
    if (is_adam) {
        bias_corr1 = 1.0 - pow(beta_1, itr);
        bias_corr2 = 1.0 - pow(beta_2, itr);
        if (bias_corr1 <= 0.0) bias_corr1 = 1e-15;
        if (bias_corr2 <= 0.0) bias_corr2 = 1e-15;
    }

    for (int k = 0; k < nn->n_layers - 1; k++) {
        int n_in  = nn->n_neurons_per_layer[k];
        int n_out = nn->n_neurons_per_layer[k + 1];

        // 1. Biases Update
        for (int j = 1; j <= n_out; j++) {
            double grad_b  = ipc->shared_db[k][j] * inv_batch;
            double delta_b = 0.0;

            if (is_sgd) {
                delta_b = lr * grad_b;
            } else if (is_adam) {

                nn->momentum_b[k][j] = beta_1 * nn->momentum_b[k][j] + (1.0 - beta_1) * grad_b;
                nn->momentum2_b[k][j] = beta_2 * nn->momentum2_b[k][j] + (1.0 - beta_2) * grad_b * grad_b;

                double m_cap = nn->momentum_b[k][j] / bias_corr1;
                double v_cap = nn->momentum2_b[k][j] / bias_corr2;
                if (v_cap < 0.0) v_cap = 0.0;

                delta_b = (lr * m_cap) / (sqrt(v_cap) + epsilon);
            }

            UPDATE_REPLICAS_GLOBAL(&(nn->b[k][j]), delta_b, global_aligned_slab_size, num_numa_nodes);
            ipc->shared_db[k][j] = 0.0; // Reset shared buffer
        }

        // 2. Weights Update
        for (int i = 1; i <= n_in; i++) {
            for (int j = 1; j <= n_out; j++) {
                double grad_w  = ipc->shared_dw[k][i][j] * inv_batch;
                double delta_w = 0.0;

                if (is_sgd) {
                    delta_w = lr * grad_w;
                } else if (is_adam) {
                    nn->momentum_w[k][i][j] = beta_1 * nn->momentum_w[k][i][j] + (1.0 - beta_1) * grad_w;
                    nn->momentum2_w[k][i][j] = beta_2 * nn->momentum2_w[k][i][j] + (1.0 - beta_2) * grad_w * grad_w;

                    double m_cap = nn->momentum_w[k][i][j] / bias_corr1;
                    double v_cap = nn->momentum2_w[k][i][j] / bias_corr2;
                    if (v_cap < 0.0) v_cap = 0.0;

                    delta_w = (lr * m_cap) / (sqrt(v_cap) + epsilon);
                }

                UPDATE_REPLICAS_GLOBAL(&(nn->w[k][i][j]), delta_w, global_aligned_slab_size, num_numa_nodes);
                ipc->shared_dw[k][i][j] = 0.0; // Reset shared buffer
            }
        }
    }
}


void back_propagation_accumulate(struct NeuralNet* nn, char* activation_fun, char* loss) {
    int last = nn->n_layers - 1;

    for (int i = 1; i <= nn->n_neurons_per_layer[last]; i++) {
        if (strcmp(loss, "mse") == 0) {
            nn->delta[last][i] = (nn->out[last][i] - nn->targets[i]) * sigmoid_d(nn->out[last][i]);
        } else {
            nn->delta[last][i] = nn->out[last][i] - nn->targets[i];
        }
    }

    for (int k = nn->n_layers - 2; k > 0; k--) {
        for (int i = 1; i <= nn->n_neurons_per_layer[k]; i++) {
            double sum = 0.0;
            for (int j = 1; j <= nn->n_neurons_per_layer[k + 1]; j++) {
                sum += nn->w[k][i][j] * nn->delta[k + 1][j];
            }

            double grad;
            if (strcmp(activation_fun, "tanh") == 0)      grad = tanh_d(nn->out[k][i]);
            else if (strcmp(activation_fun, "relu") == 0) grad = relu_d(nn->out[k][i]);
            else                                          grad = sigmoid_d(nn->out[k][i]);

            nn->delta[k][i] = grad * sum;
        }
    }

    for (int k = 0; k < nn->n_layers - 1; k++) {
        for (int j = 1; j <= nn->n_neurons_per_layer[k + 1]; j++) {
            nn->db[k][j] += nn->delta[k + 1][j];
            for (int i = 1; i <= nn->n_neurons_per_layer[k]; i++) {
                nn->dw[k][i][j] += nn->delta[k + 1][j] * nn->out[k][i];
            }
        }
    }
}

void update_weights_batch(struct NeuralNet* nn, double lr, char* opt, int itr, int batch_size) {
    double inv_batch = 1.0 / (double)batch_size;
    int is_adam = (strcmp(opt, "adam") == 0);
    int is_sgd  = (strcmp(opt, "sgd") == 0);

    double bias_corr1 = 1.0;
    double bias_corr2 = 1.0;
    if (is_adam) {
        bias_corr1 = 1.0 - pow(beta_1, itr);
        bias_corr2 = 1.0 - pow(beta_2, itr);
        if (bias_corr1 <= 0.0) bias_corr1 = 1e-15;
        if (bias_corr2 <= 0.0) bias_corr2 = 1e-15;
    }

    for (int k = 0; k < nn->n_layers - 1; k++) {
        int n_in  = nn->n_neurons_per_layer[k];
        int n_out = nn->n_neurons_per_layer[k + 1];

        // 1. Biases
        for (int j = 1; j <= n_out; j++) {
            double grad_b  = nn->db[k][j] * inv_batch;
            double delta_b = 0.0;

            if (is_sgd) {
                delta_b = lr * grad_b;
            } else if (is_adam) {
                // todo: non si possono fare atomic, capire l'impatto della perdita di update qui
                // bisogna capire se ha senso tenere momentum intra-nodo oppure replicarla sui vari nodi numa
                // todo: forse momentum vanno allocate nell'heap dei processi e non condivise tra i processi intra-nodo
                nn->momentum_b[k][j] = beta_1 * nn->momentum_b[k][j] + (1.0 - beta_1) * grad_b;
                nn->momentum2_b[k][j] = beta_2 * nn->momentum2_b[k][j] + (1.0 - beta_2) * grad_b * grad_b;
                // memory fence qui?
                double m_cap = nn->momentum_b[k][j] / bias_corr1;
                double v_cap = nn->momentum2_b[k][j] / bias_corr2;
                if (v_cap < 0.0) v_cap = 0.0;

                delta_b = (lr * m_cap) / (sqrt(v_cap) + epsilon);
            }

            UPDATE_REPLICAS_GLOBAL(&(nn->b[k][j]), delta_b, 
                                   global_aligned_slab_size, num_numa_nodes);
            nn->db[k][j] = 0.0;
        }

        // 2. Weights
        for (int i = 1; i <= n_in; i++) {
            for (int j = 1; j <= n_out; j++) {
                double grad_w  = nn->dw[k][i][j] * inv_batch;
                double delta_w = 0.0;

                if (is_sgd) {
                    delta_w = lr * grad_w;
                } else if (is_adam) {
                    nn->momentum_w[k][i][j] = beta_1 * nn->momentum_w[k][i][j] + (1.0 - beta_1) * grad_w;
                    nn->momentum2_w[k][i][j] = beta_2 * nn->momentum2_w[k][i][j] + (1.0 - beta_2) * grad_w * grad_w;

                    double m_cap = nn->momentum_w[k][i][j] / bias_corr1;
                    double v_cap = nn->momentum2_w[k][i][j] / bias_corr2;
                    if (v_cap < 0.0) v_cap = 0.0;

                    delta_w = (lr * m_cap) / (sqrt(v_cap) + epsilon);
                }

                UPDATE_REPLICAS_GLOBAL(&(nn->w[k][i][j]), delta_w, 
                                       global_aligned_slab_size, num_numa_nodes);
                nn->dw[k][i][j] = 0.0;
            }
        }
    }
}

double* new_model_train(struct NeuralNet* nn, double** X_train, double** y_train, 
                        double* y_train_temp, char* activation_fun, char* loss, 
                        char* opt, double learning_rate, int num_samples_to_train, 
                        int itr, int batch_size) {

    int* arr = malloc(num_samples_to_train * sizeof(int));
    if (!arr) {
        perror("malloc failed in new_model_train");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_samples_to_train; i++) arr[i] = i;
    shuffle(arr, num_samples_to_train);

    int correct = 0;
    double loss_val = 0.0;
    int last = nn->n_layers - 1;
    int n_out_last = nn->n_neurons_per_layer[last];
    int n_in_first = nn->n_neurons_per_layer[0];

    for (int i = 0; i < num_samples_to_train; i += batch_size) {
        int cur_batch = (i + batch_size > num_samples_to_train) ? 
                        (num_samples_to_train - i) : batch_size;

        for (int b = 0; b < cur_batch; b++) {
            int idx = arr[i + b];

            for (int j = 1; j <= n_in_first; j++)
                nn->out[0][j] = X_train[idx][j - 1];

            for (int j = 1; j <= n_out_last; j++)
                nn->targets[j] = y_train[idx][j - 1];

            forward_propagation(nn, activation_fun, loss);

            for (int j = 1; j <= n_out_last; j++) {
                if (strcmp(loss, "mse") == 0) {
                    loss_val += 0.5 * pow(nn->out[last][j] - nn->targets[j], 2);
                } else {
                    // Safe Cross-Entropy: only accumulate when target > 0
                    if (nn->targets[j] > 0.0) {
                        double p = nn->out[last][j];
                        if (p < 1e-15) p = 1e-15;
                        loss_val -= nn->targets[j] * log(p);
                    }
                }
            }

            double max_v = (double)INT_MIN;
            int pred = -1;
            for (int j = 1; j <= n_out_last; j++) {
                if (nn->out[last][j] > max_v) {
                    max_v = nn->out[last][j];
                    pred = j - 1;
                }
            }
            if (pred == (int)y_train_temp[idx]) correct++;

            back_propagation_accumulate(nn, activation_fun, loss);
        }

        update_weights_batch(nn, learning_rate, opt, itr, cur_batch);
    }

    free(arr);

    double* metrics = malloc(2 * sizeof(double));
    metrics[0] = loss_val / (double)num_samples_to_train;
    metrics[1] = (double)correct / (double)num_samples_to_train;
    return metrics;
}

void pin_process_to_node(int rank, int target_node) {
    int max_node = numa_max_node();
    if (max_node < 0 || target_node > max_node) {
        int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        if (num_cpus > 0) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(rank % num_cpus, &cpuset);
            sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        }
        return;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    struct bitmask *cpus_in_node = numa_allocate_cpumask();
    if (numa_node_to_cpus(target_node, cpus_in_node) != 0) {
        numa_free_cpumask(cpus_in_node);
        return;
    }

    int num_cpus = numa_num_possible_cpus();
    int core_in_node_count = 0;
    for (int i = 0; i < num_cpus; i++) {
        if (numa_bitmask_isbitset(cpus_in_node, i)) core_in_node_count++;
    }

    if (core_in_node_count == 0) {
        numa_free_cpumask(cpus_in_node);
        return;
    }

    int configured_nodes = numa_num_configured_nodes();
    if (configured_nodes < 1) configured_nodes = 1;
    int target_core = (rank / configured_nodes) % core_in_node_count;
    int current_match = 0, core_found = -1;

    for (int i = 0; i < num_cpus; i++) {
        if (numa_bitmask_isbitset(cpus_in_node, i)) {
            if (current_match == target_core) {
                core_found = i;
                break;
            }
            current_match++;
        }
    }

    if (core_found != -1) {
        CPU_SET(core_found, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    }
    numa_free_cpumask(cpus_in_node);
}


// ============================================================================
// parallel_training_sync_lock
// Deterministic synchronous mini-batch training with lock-based reduction.
// ============================================================================
double* parallel_training_sync_lock(struct NeuralNet* base_nn, double** X_train, double** y_train, 
                                    double* y_train_temp, char* activation_fun, char* loss, 
                                    char* opt, double learning_rate, int total_samples, 
                                    int itr, int batch_size, int n_workers) {

    // 1. Initialize auxiliary shared IPC context & reduction buffers
    NodeIPCContext* ipc = init_node_ipc_context(n_workers, base_nn->n_layers, 
                                                base_nn->n_neurons_per_layer, total_samples);

    // 2. Master generates deterministic shuffle in shared memory
    for (int i = 0; i < total_samples; i++) ipc->shared_sample_order[i] = i;
    shuffle(ipc->shared_sample_order, total_samples);

    pid_t* pids = malloc(n_workers * sizeof(pid_t));
    if (!pids) {
        perror("malloc pids failed");
        exit(EXIT_FAILURE);
    }

    for (int rank = 0; rank < n_workers; rank++) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork failed");
            exit(EXIT_FAILURE);
        }

        if (pid == 0) {
            free(pids);

            // Bind process to node
            int my_node = rank % num_numa_nodes;
            pin_process_to_node(rank, my_node);

            if (activate_numa_view(my_node, base_nn, global_aligned_slab_size) < 0) {
                fprintf(stderr, "[WORKER %d] Failed to activate NUMA view\n", rank);
                exit(EXIT_FAILURE);
            }

            // Construct local struct pointing to shared 2MB slab
            struct NeuralNet local_nn;
            memset(&local_nn, 0, sizeof(struct NeuralNet));
            local_nn.magic_test_value    = base_nn->magic_test_value;
            local_nn.n_layers            = base_nn->n_layers;
            local_nn.n_neurons_per_layer = base_nn->n_neurons_per_layer;
            local_nn.w                   = base_nn->w;
            local_nn.b                   = base_nn->b;
            local_nn.momentum_w          = base_nn->momentum_w;
            local_nn.momentum2_w         = base_nn->momentum2_w;
            local_nn.momentum_b          = base_nn->momentum_b;
            local_nn.momentum2_b         = base_nn->momentum2_b;

            // Private heap working buffers (dw, db, in, out, delta)
            init_process_working_buffers(&local_nn);

            int last = local_nn.n_layers - 1;
            int n_out_last = local_nn.n_neurons_per_layer[last];
            int n_in_first = local_nn.n_neurons_per_layer[0];

            // ----------------------------------------------------------------
            // Synchronous Mini-Batch Step Loop
            // ----------------------------------------------------------------
            for (int b_start = 0; b_start < total_samples; b_start += batch_size) {
                int cur_batch = (b_start + batch_size > total_samples) ? 
                                (total_samples - b_start) : batch_size;

                // Evenly partition current global mini-batch among workers
                int base_count = cur_batch / n_workers;
                int rem        = cur_batch % n_workers;
                int my_offset  = rank * base_count + (rank < rem ? rank : rem);
                int my_count   = base_count + (rank < rem ? 1 : 0);

                double local_loss = 0.0;
                int local_correct = 0;

                // 1. Forward & Private Backward Pass on local partition
                for (int s = 0; s < my_count; s++) {
                    int idx = ipc->shared_sample_order[b_start + my_offset + s];

                    for (int j = 1; j <= n_in_first; j++)
                        local_nn.out[0][j] = X_train[idx][j - 1];

                    for (int j = 1; j <= n_out_last; j++)
                        local_nn.targets[j] = y_train[idx][j - 1];

                    forward_propagation(&local_nn, activation_fun, loss);

                    // Compute loss
                    for (int j = 1; j <= n_out_last; j++) {
                        if (strcmp(loss, "mse") == 0) {
                            local_loss += 0.5 * pow(local_nn.out[last][j] - local_nn.targets[j], 2);
                        } else {
                            if (local_nn.targets[j] > 0.0) {
                                double p = local_nn.out[last][j];
                                if (p < 1e-15) p = 1e-15;
                                local_loss -= local_nn.targets[j] * log(p);
                            }
                        }
                    }

                    // Compute argmax prediction
                    double max_v = (double)INT_MIN;
                    int pred = -1;
                    for (int j = 1; j <= n_out_last; j++) {
                        if (local_nn.out[last][j] > max_v) {
                            max_v = local_nn.out[last][j];
                            pred = j - 1;
                        }
                    }
                    if (pred == (int)y_train_temp[idx]) local_correct++;

                    // Backpropagate into private dw and db
                    back_propagation_accumulate(&local_nn, activation_fun, loss);
                }

                // 2. Lock-Based Reduction Phase
                pthread_mutex_lock(&ipc->lock);
                accumulate_gradients_locked(ipc, &local_nn);
                ipc->epoch_loss += local_loss;
                ipc->epoch_correct += (double)local_correct;
                pthread_mutex_unlock(&ipc->lock);

                // 3. Barrier 1: Wait until all W workers contribute gradients
                pthread_barrier_wait(&ipc->barrier_accumulate);

                // 4. Parameter Update: Worker 0 applies step to shared slab
                if (rank == 0) {
                    update_weights_from_shared(&local_nn, ipc, learning_rate, opt, itr, cur_batch);
                }

                // 5. Barrier 2: Wait until Worker 0 finishes weight update
                pthread_barrier_wait(&ipc->barrier_update);
            }

            free_process_working_buffers(&local_nn);
            exit(0);
        }

        pids[rank] = pid;
    }

    // Master collects workers
    int child_failed = 0;
    for (int i = 0; i < n_workers; i++) {
        int status;
        waitpid(pids[i], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "[MASTER] Worker PID %d failed\n", pids[i]);
            child_failed = 1;
        }
    }
    free(pids);

    if (child_failed) {
        destroy_node_ipc_context(ipc);
        exit(EXIT_FAILURE);
    }

    double* final_metrics = malloc(2 * sizeof(double));
    final_metrics[0] = ipc->epoch_loss / (double)total_samples;
    final_metrics[1] = ipc->epoch_correct / (double)total_samples;

    destroy_node_ipc_context(ipc);
    return final_metrics;
}


double* parallel_training(struct NeuralNet* base_nn, double** X_train, double** y_train, 
                         double* y_train_temp, char* activation_fun, char* loss, 
                         char* opt, double learning_rate, int total_samples, 
                         int itr, int batch_size, int n_workers) {

    double* shared_metrics = mmap(NULL, n_workers * 2 * sizeof(double),
                                  PROT_READ | PROT_WRITE, 
                                  MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (shared_metrics == MAP_FAILED) {
        perror("mmap shared_metrics failed");
        exit(EXIT_FAILURE);
    }

    pid_t* pids = malloc(n_workers * sizeof(pid_t));
    if (!pids) {
        perror("malloc pids failed");
        exit(EXIT_FAILURE);
    }

    volatile char* ptr = (char*)base_nn;
    for (size_t i = 0; i < global_mmap_total_size; i += PAGE_SIZE) {
        ptr[i] = ptr[i];
    }

    for (int rank = 0; rank < n_workers; rank++) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork failed");
            exit(EXIT_FAILURE);
        }

        if (pid == 0) {
            free(pids);

             // If 1 worker, keep seed 12345 so shuffle matches serial nn_app_std exactly
            if (n_workers == 1) {
                srand(12345);
            } else {
                unsigned int worker_seed = 12345 + (rank * 10007) + (itr * 31);
                srand(worker_seed);
            }

            int my_node = rank % num_numa_nodes;
            pin_process_to_node(rank, my_node);

            if (activate_numa_view(my_node, base_nn, global_aligned_slab_size) < 0) {
                fprintf(stderr, "[WORKER %d] Failed to activate NUMA view for Node %d\n", rank, my_node);
                exit(EXIT_FAILURE);
            }

            struct NeuralNet local_nn;
            memset(&local_nn, 0, sizeof(struct NeuralNet));
            local_nn.magic_test_value    = base_nn->magic_test_value;
            local_nn.n_layers            = base_nn->n_layers;
            local_nn.n_neurons_per_layer = base_nn->n_neurons_per_layer;
            local_nn.w                   = base_nn->w;
            local_nn.b                   = base_nn->b;
            local_nn.momentum_w          = base_nn->momentum_w;
            local_nn.momentum2_w         = base_nn->momentum2_w;
            local_nn.momentum_b          = base_nn->momentum_b;
            local_nn.momentum2_b         = base_nn->momentum2_b;

            init_process_working_buffers(&local_nn);

            int samples_per_worker = total_samples / n_workers;
            int start = rank * samples_per_worker;
            int count = (rank == n_workers - 1) ? (total_samples - start) : samples_per_worker;

            double* results = new_model_train(&local_nn, 
                                             X_train + start, 
                                             y_train + start, 
                                             y_train_temp + start, 
                                             activation_fun, loss, opt, learning_rate, 
                                             count, itr, batch_size);

            shared_metrics[rank * 2]     = results[0] * (double)count;
            shared_metrics[rank * 2 + 1] = results[1] * (double)count;

            free(results);
            free_process_working_buffers(&local_nn);
            exit(0);
        }

        pids[rank] = pid;
    }

    int child_failed = 0;
    for (int i = 0; i < n_workers; i++) {
        int status;
        waitpid(pids[i], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "[MASTER] Worker PID %d failed\n", pids[i]);
            child_failed = 1;
        }
    }
    free(pids);

    if (child_failed) {
        exit(EXIT_FAILURE);
    }

    double total_loss_sum = 0.0, total_correct_sum = 0.0;
    for (int i = 0; i < n_workers; i++) {
        total_loss_sum    += shared_metrics[i * 2];
        total_correct_sum += shared_metrics[i * 2 + 1];
    }

    double* final_metrics = malloc(2 * sizeof(double));
    final_metrics[0] = total_loss_sum / (double)total_samples;
    final_metrics[1] = total_correct_sum / (double)total_samples;

    munmap(shared_metrics, n_workers * 2 * sizeof(double));
    return final_metrics;
}