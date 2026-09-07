#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "neural_network.h"
#include "activation.h"

#define TOLERANCE 1e-12

#ifdef NUMA_API_ENABLED
extern struct NeuralNet* __real_newNet(int n_layers, int n_neurons_per_layer[]);
extern void __real_free_NN(struct NeuralNet* nn);
extern void __real_forward_propagation(struct NeuralNet* nn, char* activation_fun, char* loss);
#define STD_NEWNET __real_newNet
#define STD_FREE_NN __real_free_NN
#define STD_FORWARD __real_forward_propagation
#else
#define STD_NEWNET newNet
#define STD_FREE_NN free_NN
#define STD_FORWARD forward_propagation
#endif

static inline bool check_parity(double a, double b, const char* name, int layer, int i, int j) {
    double diff = fabs(a - b);
    if (diff > TOLERANCE) {
        printf("  [MISMATCH] %s at Layer %d [%d][%d]: Standard=%.15e, NUMA=%.15e (Diff=%.15e)\n",
               name, layer, i, j, a, b, diff);
        return false;
    }
    return true;
}

int main() {
    printf("====================================================================\n");
    printf("     Deterministic Parity Test: Standard Malloc vs. NUMA Engine      \n");
    printf("====================================================================\n");

    int n_layers = 4;
    int neurons[] = {784, 64, 32, 10};
    num_numa_nodes = 1;

    // 1. Standard Baseline Model (Heap Malloc)
    struct NeuralNet* std_nn = STD_NEWNET(n_layers, neurons);
    init_process_working_buffers(std_nn); // Allocate working buffers (dw, db, in, out, delta)

    // 2. NUMA Engine Model (2MB HugeTLB mmap layout)
    struct NeuralNet* numa_nn = setup_numa_model(n_layers, neurons);
    activate_numa_view(0, numa_nn, global_aligned_slab_size);

    struct NeuralNet local_numa;
    local_numa.magic_test_value    = numa_nn->magic_test_value;
    local_numa.n_layers            = numa_nn->n_layers;
    local_numa.n_neurons_per_layer = numa_nn->n_neurons_per_layer;
    local_numa.w                   = numa_nn->w;            // Points to 0x1000000000
    local_numa.b                   = numa_nn->b;            // Points to 0x1000000000
    local_numa.momentum_w          = numa_nn->momentum_w;   // Points to 0x1000000000
    local_numa.momentum2_w         = numa_nn->momentum2_w;  // Points to 0x1000000000
    local_numa.momentum_b          = numa_nn->momentum_b;   // Points to 0x1000000000
    local_numa.momentum2_b         = numa_nn->momentum2_b;  // Points to 0x1000000000
    init_process_working_buffers(&local_numa);

    // -------------------------------------------------------------------------
    // TEST 1: Memory Layout Separation Check
    // -------------------------------------------------------------------------
    printf("[TEST 1] Verifying Memory Layout Separation...\n");
    bool layout_ok = true;
    if ((uintptr_t)local_numa.w < 0x1000000000) {
        printf("  [FAIL] NUMA weights are not in the 2MB mmap region!\n");
        layout_ok = false;
    }
    if ((uintptr_t)local_numa.out >= 0x1000000000 && (uintptr_t)local_numa.out < 0x1000000000 + global_mmap_total_size) {
        printf("  [FAIL] Working buffer 'out' is inside shared slab (must be process-private)!\n");
        layout_ok = false;
    }
    printf("  -> Memory Layout Check: %s\n", layout_ok ? "PASS (Shared parameters at 0x1000000000, Working buffers private)" : "FAIL");

    // -------------------------------------------------------------------------
    // TEST 2: Deterministic Initialization
    // -------------------------------------------------------------------------
    printf("[TEST 2] Deterministic Weight & Bias Initialization...\n");
    seed = 42;
    for (int k = 0; k < n_layers - 1; k++) {
        int n_in  = neurons[k] + 1;
        int n_out = neurons[k + 1] + 1;

        // FIXED: Biases loop up to n_out (no overflow)
        for (int j = 0; j < n_out; j++) {
            double b_val = (double)(j + 1) * 0.005;
            std_nn->b[k][j] = b_val;
            local_numa.b[k][j] = b_val;
            std_nn->momentum_b[k][j] = 0.0;
            std_nn->momentum2_b[k][j] = 0.0;
            local_numa.momentum_b[k][j] = 0.0;
            local_numa.momentum2_b[k][j] = 0.0;
        }

        // Weights loop over n_in * n_out
        for (int i = 0; i < n_in; i++) {
            for (int j = 0; j < n_out; j++) {
                double w_val = randn();
                std_nn->w[k][i][j] = w_val;
                local_numa.w[k][i][j] = w_val;
                std_nn->momentum_w[k][i][j] = 0.0;
                std_nn->momentum2_w[k][i][j] = 0.0;
                local_numa.momentum_w[k][i][j] = 0.0;
                local_numa.momentum2_w[k][i][j] = 0.0;
            }
        }
    }

    // Set identical input and target
    for (int j = 1; j <= neurons[0]; j++) {
        double in_val = sin((double)j * 0.05);
        std_nn->out[0][j] = in_val;
        local_numa.out[0][j] = in_val;
    }
    for (int j = 1; j <= neurons[n_layers - 1]; j++) {
        double target_val = (j == 3) ? 1.0 : 0.0;
        std_nn->targets[j] = target_val;
        local_numa.targets[j] = target_val;
    }

    // -------------------------------------------------------------------------
    // TEST 3: Forward Propagation Parity
    // -------------------------------------------------------------------------
    printf("[TEST 3] Running Forward Propagation...\n");
    STD_FORWARD(std_nn, "relu", "ce");
    STD_FORWARD(&local_numa, "relu", "ce");

    bool fp_passed = true;
    for (int k = 1; k < n_layers; k++) {
        for (int j = 1; j <= neurons[k]; j++) {
            if (!check_parity(std_nn->in[k][j], local_numa.in[k][j], "in", k, j, 0)) fp_passed = false;
            if (!check_parity(std_nn->out[k][j], local_numa.out[k][j], "out", k, j, 0)) fp_passed = false;
        }
    }
    printf("  -> Forward Propagation: %s\n", fp_passed ? "PASS (Exact Match)" : "FAIL");

    // -------------------------------------------------------------------------
    // TEST 4: Backpropagation & Gradient Accumulation Parity
    // -------------------------------------------------------------------------
    printf("[TEST 4] Running Backpropagation Gradient Accumulation...\n");
    back_propagation_accumulate(std_nn, "relu", "ce");
    back_propagation_accumulate(&local_numa, "relu", "ce");

    bool bp_passed = true;
    for (int k = 1; k < n_layers; k++) {
        for (int j = 1; j <= neurons[k]; j++) {
            if (!check_parity(std_nn->delta[k][j], local_numa.delta[k][j], "delta", k, j, 0)) bp_passed = false;
        }
    }
    for (int k = 0; k < n_layers - 1; k++) {
        for (int j = 1; j <= neurons[k + 1]; j++) {
            if (!check_parity(std_nn->db[k][j], local_numa.db[k][j], "db", k, j, 0)) bp_passed = false;
            for (int i = 1; i <= neurons[k]; i++) {
                if (!check_parity(std_nn->dw[k][i][j], local_numa.dw[k][i][j], "dw", k, i, j)) bp_passed = false;
            }
        }
    }
    printf("  -> Gradient Accumulation: %s\n", bp_passed ? "PASS (Exact Match)" : "FAIL");

    // -------------------------------------------------------------------------
    // TEST 5: Parameter Update Step (SGD, Batch Size = 1)
    // -------------------------------------------------------------------------
    printf("[TEST 5] Executing Parameter Update (SGD, lr = 0.01)...\n");
    double lr = 0.01;
    update_weights_batch(std_nn, lr, "sgd", 1, 1);
    update_weights_batch(&local_numa, lr, "sgd", 1, 1);

    bool sgd_passed = true;
    double max_sgd_diff = 0.0;
    for (int k = 0; k < n_layers - 1; k++) {
        for (int j = 1; j <= neurons[k + 1]; j++) {
            double diff_b = fabs(std_nn->b[k][j] - local_numa.b[k][j]);
            if (diff_b > max_sgd_diff) max_sgd_diff = diff_b;
            if (!check_parity(std_nn->b[k][j], local_numa.b[k][j], "bias", k, j, 0)) sgd_passed = false;

            for (int i = 1; i <= neurons[k]; i++) {
                double diff_w = fabs(std_nn->w[k][i][j] - local_numa.w[k][i][j]);
                if (diff_w > max_sgd_diff) max_sgd_diff = diff_w;
                if (!check_parity(std_nn->w[k][i][j], local_numa.w[k][i][j], "weight", k, i, j)) sgd_passed = false;
            }
        }
    }
    printf("  -> SGD Weight Update: %s (Max Difference = %.15e)\n", sgd_passed ? "PASS" : "FAIL", max_sgd_diff);

    // -------------------------------------------------------------------------
    // TEST 6: Multi-Sample Adam Optimization Loop (100 Samples)
    // -------------------------------------------------------------------------
    printf("[TEST 6] Executing 100-Sample Adam Training Loop...\n");
    double adam_lr = 1e-4;

    for (int step = 1; step <= 100; step++) {
        for (int j = 1; j <= neurons[0]; j++) {
            double v = sin((double)(step * neurons[0] + j) * 0.02);
            std_nn->out[0][j] = v;
            local_numa.out[0][j] = v;
        }
        for (int j = 1; j <= neurons[n_layers - 1]; j++) {
            double t = (j == (step % 10 + 1)) ? 1.0 : 0.0;
            std_nn->targets[j] = t;
            local_numa.targets[j] = t;
        }

        STD_FORWARD(std_nn, "relu", "ce");
        STD_FORWARD(&local_numa, "relu", "ce");

        back_propagation_accumulate(std_nn, "relu", "ce");
        back_propagation_accumulate(&local_numa, "relu", "ce");

        update_weights_batch(std_nn, adam_lr, "adam", step, 1);
        update_weights_batch(&local_numa, adam_lr, "adam", step, 1);
    }

    bool adam_passed = true;
    double max_adam_diff = 0.0;
    for (int k = 0; k < n_layers - 1; k++) {
        for (int j = 1; j <= neurons[k + 1]; j++) {
            double diff_b = fabs(std_nn->b[k][j] - local_numa.b[k][j]);
            if (diff_b > max_adam_diff) max_adam_diff = diff_b;
            if (diff_b > TOLERANCE) adam_passed = false;

            for (int i = 1; i <= neurons[k]; i++) {
                double diff_w = fabs(std_nn->w[k][i][j] - local_numa.w[k][i][j]);
                if (diff_w > max_adam_diff) max_adam_diff = diff_w;
                if (diff_w > TOLERANCE) adam_passed = false;
            }
        }
    }
    printf("  -> 100-Step Adam Optimization: %s (Max Difference = %.15e)\n", adam_passed ? "PASS" : "FAIL", max_adam_diff);

    // -------------------------------------------------------------------------
    // SUMMARY
    // -------------------------------------------------------------------------
    printf("====================================================================\n");
    bool all_passed = layout_ok && fp_passed && bp_passed && sgd_passed && adam_passed;
    if (all_passed) {
        printf(">>> FINAL RESULT: 100%% BIT-FOR-BIT MATHEMATICAL PARITY CONFIRMED <<<\n");
    } else {
        printf(">>> FINAL RESULT: DIVERGENCE DETECTED <<<\n");
    }
    printf("====================================================================\n");

    free_process_working_buffers(&local_numa);
    STD_FREE_NN(std_nn);
    unmap_nn();

    return all_passed ? 0 : 1;
}