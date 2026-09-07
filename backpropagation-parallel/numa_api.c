#define _GNU_SOURCE

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

#define UPDATE_REPLICAS_GLOBAL(local_addr, delta_val, slab_size, num_nodes) do {    \
    uintptr_t _l_addr = (uintptr_t)(local_addr);                                   \
    uintptr_t _base   = (uintptr_t)global_mmap_start;                             \
    uintptr_t _offset = _l_addr - _base;                                          \
    for (int _n = 0; _n < (num_nodes); _n++) {                                    \
        uintptr_t _target_addr = _base + ((uintptr_t)_n * (slab_size)) + _offset;  \
        double* _target = (double*)_target_addr;                                  \
        *_target -= (delta_val);                                                  \
    }                                                                             \
} while (0)

size_t calculate_shared_slab_size(int n_layers, const int n_neurons_per_layer[]) {
    size_t size = 0;

    size += sizeof(struct NeuralNet);
    size = ALIGN_CACHELINE(size);
    size += (size_t)n_layers * sizeof(int);
    size = ALIGN_CACHELINE(size);

    size += (size_t)(n_layers - 1) * 3 * sizeof(double***); // w, mw, mw2
    size += (size_t)(n_layers - 1) * 3 * sizeof(double*);   // b, mb, mb2
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

void bind_memory_to_numa_node(void* addr, size_t size, int node) {
    int max_node = numa_max_node();
    if (max_node < 0 || node > max_node) return;

    unsigned long nodemask = (1UL << node);
    if (mbind(addr, size, MPOL_BIND, &nodemask, sizeof(unsigned long) * 8, MPOL_MF_STRICT) < 0) {
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

/**
 * FIXED: Uses the exact same loop indexing as init_nn() in neural_network.c
 * so that randn() calls match identically.
 */
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
    size_t raw_slab_size = calculate_shared_slab_size(n_layers, n_neurons_per_layer);
    size_t aligned_size  = ALIGN_PAGE(raw_slab_size);

    DEBUG printf("[NUMA-SETUP] NUMA Sockets: %d | Total Mmap: %.2f MB\n", 
                 num_numa_nodes, (double)(aligned_size * num_numa_nodes) / (1024.0 * 1024.0));

    NeuralNet* nn_mapped = newNetSharedAlloc(n_layers, n_neurons_per_layer, aligned_size);

    for (int i = 0; i < num_numa_nodes; i++) {
        NeuralNet *nn = (NeuralNet*)((char*)nn_mapped + i * aligned_size);
        nn->magic_test_value = 100.0 * (double)(i + 1);
        nn->n_layers = n_layers;
        seed = 42;
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
    for (int k = 0; k < nn->n_layers - 1; k++) {
        for (int j = 1; j <= nn->n_neurons_per_layer[k + 1]; j++) {
            
            double grad_b = nn->db[k][j] / (double)batch_size;
            double delta_b = 0.0;

            if (strcmp(opt, "sgd") == 0) {
                delta_b = lr * grad_b;
            } else if (strcmp(opt, "adam") == 0) {
                nn->momentum_b[k][j] = beta_1 * nn->momentum_b[k][j] + (1.0 - beta_1) * grad_b;
                nn->momentum2_b[k][j] = beta_2 * nn->momentum2_b[k][j] + (1.0 - beta_2) * grad_b * grad_b;

                double m_cap = nn->momentum_b[k][j] / (1.0 - pow(beta_1, itr));
                double v_cap = nn->momentum2_b[k][j] / (1.0 - pow(beta_2, itr));
                delta_b = (lr * m_cap) / (sqrt(v_cap) + epsilon);
            }

            UPDATE_REPLICAS_GLOBAL(&(nn->b[k][j]), delta_b, global_aligned_slab_size, num_numa_nodes);
            nn->db[k][j] = 0.0;

            for (int i = 1; i <= nn->n_neurons_per_layer[k]; i++) {
                double grad_w = nn->dw[k][i][j] / (double)batch_size;
                double delta_w = 0.0;

                if (strcmp(opt, "sgd") == 0) {
                    delta_w = lr * grad_w;
                } else if (strcmp(opt, "adam") == 0) {
                    nn->momentum_w[k][i][j] = beta_1 * nn->momentum_w[k][i][j] + (1.0 - beta_1) * grad_w;
                    nn->momentum2_w[k][i][j] = beta_2 * nn->momentum2_w[k][i][j] + (1.0 - beta_2) * grad_w * grad_w;

                    double m_cap = nn->momentum_w[k][i][j] / (1.0 - pow(beta_1, itr));
                    double v_cap = nn->momentum2_w[k][i][j] / (1.0 - pow(beta_2, itr));
                    delta_w = (lr * m_cap) / (sqrt(v_cap) + epsilon);
                }

                UPDATE_REPLICAS_GLOBAL(&(nn->w[k][i][j]), delta_w, global_aligned_slab_size, num_numa_nodes);
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
    for (int i = 0; i < num_samples_to_train; i++) arr[i] = i;
    shuffle(arr, num_samples_to_train);

    int correct = 0;
    double loss_val = 0.0;
    int last = nn->n_layers - 1;

    for (int i = 0; i < num_samples_to_train; i += batch_size) {
        int cur_batch = (i + batch_size > num_samples_to_train) ? 
                        (num_samples_to_train - i) : batch_size;

        for (int b = 0; b < cur_batch; b++) {
            int idx = arr[i + b];

            for (int j = 1; j <= nn->n_neurons_per_layer[0]; j++)
                nn->out[0][j] = X_train[idx][j-1];

            for (int j = 1; j <= nn->n_neurons_per_layer[last]; j++)
                nn->targets[j] = y_train[idx][j-1];

            forward_propagation(nn, activation_fun, loss);

            for (int j = 1; j <= nn->n_neurons_per_layer[last]; j++) {
                if (strcmp(loss, "mse") == 0)
                    loss_val += 0.5 * pow(nn->out[last][j] - nn->targets[j], 2);
                else
                    loss_val -= nn->targets[j] * log(nn->out[last][j]);
            }

            double max_v = (double)INT_MIN; int pred = -1;
            for (int j = 1; j <= nn->n_neurons_per_layer[last]; j++) {
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

    static double metrics[2];
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

double* parallel_training(struct NeuralNet* base_nn, double** X_train, double** y_train, 
                         double* y_train_temp, char* activation_fun, char* loss, 
                         char* opt, double learning_rate, int total_samples, 
                         int itr, int batch_size, int n_workers) {

    double* shared_metrics = mmap(NULL, n_workers * 2 * sizeof(double),
                                  PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

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
            int my_node = rank % num_numa_nodes;
            pin_process_to_node(rank, my_node);

            if (activate_numa_view(my_node, base_nn, global_aligned_slab_size) < 0) {
                fprintf(stderr, "[WORKER %d] Failed to activate NUMA view for Node %d\n", rank, my_node);
                exit(EXIT_FAILURE);
            }

            struct NeuralNet local_nn;
            local_nn.magic_test_value    = base_nn->magic_test_value;
            local_nn.n_layers            = base_nn->n_layers;
            local_nn.n_neurons_per_layer = base_nn->n_neurons_per_layer;
            local_nn.w                   = base_nn->w;            // 0x1000000000 (Shared Node k)
            local_nn.b                   = base_nn->b;            // 0x1000000000 (Shared Node k)
            local_nn.momentum_w          = base_nn->momentum_w;   // 0x1000000000 (Persistent Node k)
            local_nn.momentum2_w         = base_nn->momentum2_w;  // 0x1000000000 (Persistent Node k)
            local_nn.momentum_b          = base_nn->momentum_b;   // 0x1000000000 (Persistent Node k)
            local_nn.momentum2_b         = base_nn->momentum2_b;  // 0x1000000000 (Persistent Node k)

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

            free_process_working_buffers(&local_nn);
            exit(0);
        }
    }

    for (int i = 0; i < n_workers; i++) wait(NULL);

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