#include "neural_network.h"
#include "read_data.h"
#include "utils.h"
#include "memory_manager.h"


int main(int argc, char *argv[]) {
    // 1. Seed the random number generator
    srand(time(NULL));

    // 2. Define Neural Network Architecture
    int n_layers = 4;
    int n_neurons_per_layer[] = {784, 64, 32, 10};

    // 3. Memory Calculation and Reporting
    size_t raw_size = sum_all_mmap_allocations(n_layers, n_neurons_per_layer);
    size_t aligned_size = calculate_total_nn_size_for_single_mmap(n_layers, n_neurons_per_layer);

    printf("--- Network Configuration ---\n");
    printf("Architecture: [%d] -> [%d] -> [%d] -> [%d]\n", 
           n_neurons_per_layer[0], n_neurons_per_layer[1], 
           n_neurons_per_layer[2], n_neurons_per_layer[3]);
    printf("Raw Data Size: %.3f MB\n", (double)raw_size / (1024 * 1024));
    printf("Aligned Mmap Size (incl. Huge Pages): %.3f MB\n\n", (double)aligned_size / (1024 * 1024));

    // 4. Initialize Network with Single Allocation (Hogwild/NUMA style)
    printf("Initializing NUMA-aware shared memory allocation...\n");
    struct NeuralNet** nn_array = newNetSingleAlloc(n_layers, n_neurons_per_layer);

    for (int i = 0; i < num_numa_nodes; i++) {
        if (nn_array[i] != NULL) {
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

    // 5. Setup Hyper-parameters
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

    // 6. Data Loading and Pre-processing
    // Training Data
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
    
    struct NeuralNet* base_nn = nn_array[0];

    for(int itr = 0; itr < epochs; itr++) {
        double* train_m = model_train(base_nn, X_train, y_train, y_train_temp, activation_fun, loss, opt, learning_rate, num_samples_to_train, itr+1);
        double* test_m = model_test(base_nn, X_test, y_test, y_test_temp, activation_fun, loss);

        fprintf(file, "%lf,%lf,%lf,%lf\n", train_m[0], train_m[1], test_m[0], test_m[1]);

        printf("Epoch: %02d | Train Loss: %.4f | Acc: %.4f | Test Loss: %.4f | Acc: %.4f\n", 
                itr+1, train_m[0], train_m[1], test_m[0], test_m[1]);

        // Learning Rate Decay
        learning_rate = init_lr * exp(-0.1 * (itr+1));
    }

    // 8. Cleanup
    fclose(file);
    for (int i = 0; i < num_numa_nodes; i++) {
        if (nn_array[i]) free_NN(nn_array[i]);
    }

    // Free dataset
    for(int i=0; i<N_SAMPLES; i++) { free(X_train[i]); free(y_train[i]); }
    free(X_train); free(y_train); free(y_train_temp);
    for(int i=0; i<N_TEST_SAMPLES; i++) { free(X_test[i]); free(y_test[i]); }
    free(X_test); free(y_test); free(y_test_temp);

    return 0;
}