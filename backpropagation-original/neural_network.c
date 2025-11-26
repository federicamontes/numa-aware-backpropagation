#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <string.h>
#include "read_data.h"
#include "activation.h"


// Initialize constants used in optimizers
const double epsilon = 1e-8;
const double beta = 0.9;
const double beta_1 = 0.9;
const double beta_2 = 0.999;


// Pseudo-random number generator 
int seed;
double randn(){
  int a = 1103515245;
  int m = 2147483647;
  int c = 12345;
  seed = (a * seed + c) % m;
  double x = (double)seed/(double)m;
  return x;
}


// Neural Network struct definition

struct NeuralNet{
    int n_layers; // Stores the number of layers
    int* n_neurons_per_layer; // Stores the number of neurons in each layer
    
    /* weights and bias */
    double*** w; // Stores the weights between each neurons in each pair of layers, 3D: [layer][prev_neuron][curr_neuron]
    double** b; // Stores the bias weights from bias unit to neurons in the next layer, 2D: [layer][curr_neuron]
    
    /* optimzer variables */
    double*** momentum_w; // Stores the first order moment for the weights (mobile avg for derivatives)
    double*** momentum2_w; // Stores the second order moment for the weights (mobile avg for quadratic derivatives)
    double** momentum_b; // Stores the first order moment for the bias
    double** momentum2_b; // Stores the second order moment for the weights
    
    /* backpropagation parameters */
    double** delta; // Stores the errors computed for each neuron in each layer, 2D: [layer][neuron]
    double** in; // Stores the input to the activation function in each layer, 2D: [layer][neuron]
    double** out; // Stores the output of the activation function in each layer, 2D: [layer][neuron]

    double* targets; // Stores the actual output for a given sample. It is a one-hot vector
};


/** Function to create a neural network and allocate memory
* @param n_layers: int, number of layers
* @param n_neurons_per_layer: int[], array of neurons per layers
* @return struct NeuralNet*, a neural network ptr 
*/
struct NeuralNet* newNet(int n_layers, int n_neurons_per_layer[]){

    struct NeuralNet* nn = malloc(sizeof(struct NeuralNet));
    nn->n_layers = n_layers;

    // alloc neurons per layer and c
    nn->n_neurons_per_layer = malloc(nn->n_layers * sizeof(int));
    for(int i=0;i<n_layers;i++){
        nn->n_neurons_per_layer[i] = n_neurons_per_layer[i];
    }

    // alloc weights, biases and momentum states
    nn->w = malloc((nn->n_layers-1)*sizeof(double**));
    nn->momentum_w = malloc((nn->n_layers-1)*sizeof(double**));
    nn->momentum2_w = malloc((nn->n_layers-1)*sizeof(double**));
    nn->b = malloc((nn->n_layers-1)*sizeof(double*));
    nn->momentum_b = malloc((nn->n_layers-1)*sizeof(double*));
    nn->momentum2_b = malloc((nn->n_layers-1)*sizeof(double*));
    for(int i=0;i<nn->n_layers-1;i++){
        nn->w[i] = malloc((nn->n_neurons_per_layer[i] + 1)*sizeof(double*));
        nn->momentum_w[i] = malloc((nn->n_neurons_per_layer[i] + 1)*sizeof(double*));
        nn->momentum2_w[i] = malloc((nn->n_neurons_per_layer[i] + 1)*sizeof(double*));
        nn->b[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        nn->momentum_b[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        nn->momentum2_b[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        for(int j=0;j<nn->n_neurons_per_layer[i]+1;j++){
            nn->w[i][j] = malloc((nn->n_neurons_per_layer[i+1] + 1)*sizeof(double));
            nn->momentum_w[i][j] = malloc((nn->n_neurons_per_layer[i+1] + 1)*sizeof(double));
            nn->momentum2_w[i][j] = malloc((nn->n_neurons_per_layer[i+1] + 1)*sizeof(double));
        }
    }

    // alloc of backpropagation parameters
    nn->delta = malloc((nn->n_layers)*sizeof(double*));
    nn->in = malloc((nn->n_layers)*sizeof(double*));
    nn->out = malloc((nn->n_layers)*sizeof(double*));
    for(int i=0;i<nn->n_layers;i++){
        nn->in[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        nn->out[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
        nn->delta[i] = malloc((nn->n_neurons_per_layer[i]+1)*sizeof(double));
    }

    // alloc target one-hot vector
    nn->targets = malloc((nn->n_neurons_per_layer[nn->n_layers-1]+1)*sizeof(double));
    return nn;
}


/** Function to free the dynamically allocated memory
 * @param nn: struct NeuralNet*, the neural network to free
 * 
 * */
void free_NN(struct NeuralNet* nn){

    //for each layer free from inside-out everything allocated in newNet
    for(int i=0;i<nn->n_layers-1;i++){
        for(int j=0;j<nn->n_neurons_per_layer[i]+1;j++){
            free(nn->w[i][j]);
            free(nn->momentum_w[i][j]);
            free(nn->momentum2_w[i][j]);
        }
        free(nn->w[i]);
        free(nn->momentum_w[i]);
        free(nn->momentum2_w[i]);
        free(nn->b[i]);
        free(nn->momentum_b[i]);
        free(nn->momentum2_b[i]);
    }
    free(nn->w);
    free(nn->momentum_w);
    free(nn->momentum2_w);
    free(nn->b);
    free(nn->momentum_b);
    free(nn->momentum2_b);
    
    // free backpropagation parameters
    for(int i=0;i<nn->n_layers;i++){
        free(nn->in[i]);
        free(nn->out[i]);
        free(nn->delta[i]);
    }

    free(nn->in);
    free(nn->out);
    free(nn->delta);
    
    free(nn->targets);
    free(nn->n_neurons_per_layer);
}


/** Initialize the neural network
 * @param nn: struct NeuralNet* the neural network
 * */
void init_nn(struct NeuralNet* nn){

    // for each layer with out weights
    for(int k=0;k<nn->n_layers-1;k++){
        // for each neuron in layer k
        for(int i=1;i<nn->n_neurons_per_layer[k]+1;i++){
            nn->b[k][i] = 0.0;
            nn->momentum_b[k][i] = 0.0;
            nn->momentum2_b[k][i] = 0.0;
            //for each neuron on layer k+1
            for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){
                nn->w[k][i][j] = randn();
                nn->momentum_w[k][i][j] = 0.0;
                nn->momentum2_w[k][i][j] = 0.0;
            }
        }
    }
}


// Function to shuffle elements of an array
void shuffle(int* arr, size_t n){
    if(n > 1){
        for(size_t i=0;i<n-1;i++){
        size_t j = i+rand()/(RAND_MAX/(n-i)+1);
          int t = arr[j];
          arr[j] = arr[i];
          arr[i] = t;
        }
    }
}


/** Function for forward propagation step
 * @param nn: struct NeuralNet *, ptr to the neural network
 * @param activation_fun: char *, activation function
 * @param loss: char *, loss function
 * */
void forward_propagation(struct NeuralNet* nn, char* activation_fun, char* loss){

    // cleanup of input for each layer
    for(int i=0;i<nn->n_layers;i++){
        for(int j=0;j<nn->n_neurons_per_layer[i]+1;j++){
            nn->in[i][j] = 0.0;
        }
    }

    //for each layer
    for(int k=1;k<nn->n_layers;k++){

        /* Compute the weighted sum */
        // add bias
        for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
            nn->in[k][j] += 1.0 * nn->b[k-1][j];
        }

        // add weighed sum of outputs of prev layer
        for(int i=1;i<nn->n_neurons_per_layer[k-1]+1;i++){
            for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
                nn->in[k][j] += nn->out[k-1][i] * nn->w[k-1][i][j];
            }
        }

        /* Apply non-linear activation function to the weighted sums */

        //if last layer apply one of the loss functions
        if(k == nn->n_layers-1){
            if(strcmp(loss, "mse") == 0){ // mean square error
                for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
                    nn->out[k][j] = sigmoid(nn->in[k][j]); // apply sigmoid on in to obtain out
                }
            }
            else if(strcmp(loss, "ce") == 0){ // if cross-entropy
                double max_input_to_softmax = (double)INT_MIN; //use softmax

                for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){ // find max input
                    if(fabs(nn->in[k][j]) > max_input_to_softmax){
                        max_input_to_softmax = fabs(nn->in[k][j]);
                    }
                }
                double deno = 0.0;
                for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){ //compute denominator
                    nn->in[k][j] /= max_input_to_softmax;
                    deno += exp(nn->in[k][j]);
                }
                for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){ //apply softmax
                    nn->out[k][j] = (double)exp(nn->in[k][j])/(double)deno;
    
                }
            }
        } else{ // if hidden layer
            //for each neuron per layer apply activation function specified among sigmoid, tanh or relu
            for(int j=1;j<nn->n_neurons_per_layer[k]+1;j++){
                if(strcmp(activation_fun, "sigmoid") == 0){
                    nn->out[k][j] = sigmoid(nn->in[k][j]);
                }
                else if(strcmp(activation_fun, "tanh") == 0){
                    nn->out[k][j] = tanh(nn->in[k][j]);
                }
                else if(strcmp(activation_fun, "relu") == 0){
                    nn->out[k][j] = relu(nn->in[k][j]);
                }
                else{
                    nn->out[k][j] = sigmoid(nn->in[k][j]);
                }
            }
        }
    }
}


// Function to calculate loss
double calc_loss(struct NeuralNet* nn, char* loss){
    double loss_val = 0.0;
    int last_layer = nn->n_layers-1;

    for(int i=1;i<nn->n_neurons_per_layer[last_layer]+1;i++){
        if(strcmp(loss, "mse") == 0){
            loss_val += (0.5)*(nn->out[last_layer][i] - nn->targets[i]) * (nn->out[last_layer][i] - nn->targets[i]);
        }
        else if(strcmp(loss, "ce") == 0){
            loss_val -= nn->targets[i]*(log(nn->out[last_layer][i]));
        }
	}
    return loss_val;
}


/** Function for back propagation step
 * @param nn: struct NeuralNet*, ptr to the neural network
 * @param activation_fun: char *, activation function
 * @param learning_rate: double, the learning rate
 * @param loss: char *, the loss function
 * @param opt: char *, optimizer used
 * @param itr: int, current iteration
 * 
 * */
void back_propagation(struct NeuralNet* nn, char* activation_fun, double learning_rate, char* loss, char* opt, int itr){

    int last_layer = nn->n_layers-1;

    /* Calculate the error in the output layer */
    for(int i=1;i<nn->n_neurons_per_layer[last_layer]+1;i++){
        if(strcmp(loss, "mse") == 0){
            double grad = sigmoid_d(nn->out[last_layer][i]);
            nn->delta[last_layer][i] = grad*(nn->out[last_layer][i] - nn->targets[i]);
        }
        else if(strcmp(loss, "ce") == 0){
            nn->delta[last_layer][i] = nn->out[last_layer][i] - nn->targets[i];
        }
    }

    /* Backpropagate the error from the last layer to the first layer */
    for(int k=nn->n_layers-2;k>0;k--){
        
        for(int i=1;i<nn->n_neurons_per_layer[k]+1;i++){

            // weighted sum of deltas of next layer
            double sum = 0.0;
            for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){
                sum += nn->b[k][j] * nn->delta[k+1][j];
            }
            for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){
                sum += nn->w[k][i][j] * nn->delta[k+1][j];
            }

            // compute gradient of activation function
            double grad;
            if(strcmp(activation_fun, "sigmoid") == 0){
                grad = sigmoid_d(nn->out[k][i]);
            }
            else if(strcmp(activation_fun, "tanh") == 0){
                grad = tanh_d(nn->out[k][i]);
            }
            else if(strcmp(activation_fun, "relu") == 0){
                grad = relu_d(nn->out[k][i]);
            }
            else{
                grad = sigmoid_d(nn->out[k][i]);
            }
            // delta of hidden layer
            nn->delta[k][i] = grad * sum;
        }
    }

    /* Update the weights according to the given optimization technique */
    //for each layer to update
    for(int k=0;k<nn->n_layers-1;k++){

        for(int i=1;i<nn->n_neurons_per_layer[k]+1;i++){
            for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){

                // dw is partial gradient of weight w
                double dw = nn->delta[k+1][j] * nn->out[k][i];

                if(strcmp(opt, "sgd") == 0){ //stochastic gradient descent
                    nn->w[k][i][j] -= learning_rate * dw;
                }
                else if(strcmp(opt, "momentum") == 0){
                    nn->momentum_w[k][i][j] = beta * nn->momentum_w[k][i][j] + (1.0-beta) * dw * learning_rate;
                    nn->w[k][i][j] -= nn->momentum_w[k][i][j];
                }
                else if(strcmp(opt, "rmsprop") == 0){
                    nn->momentum_w[k][i][j] = beta * nn->momentum_w[k][i][j] + (1.0-beta) * dw * dw;
                    nn->w[k][i][j] -= (learning_rate * dw)/(sqrt(nn->momentum_w[k][i][j]) + 1e-6);
                }
                else if(strcmp(opt, "adam") == 0){
                    nn->momentum_w[k][i][j] = beta_1 * nn->momentum_w[k][i][j] + (1.0-beta_1) * dw;
                    nn->momentum2_w[k][i][j] = beta_2 * nn->momentum2_w[k][i][j] + (1.0-beta_2) * dw * dw;
                    double m_cap = (double)nn->momentum_w[k][i][j]/(double)(1.0 - pow(beta_1, itr));
                    double v_cap = (double)nn->momentum2_w[k][i][j]/(double)(1.0 - pow(beta_2, itr));
                    nn->w[k][i][j] -= (learning_rate * m_cap)/(sqrt(v_cap) + epsilon);
                }
            }
        }

        /* Update the bias weights */
        for(int j=1;j<nn->n_neurons_per_layer[k+1]+1;j++){

            // db is partial gradient for bias b
            double db = nn->delta[k+1][j] * 1.0;

            if(strcmp(opt, "sgd") == 0){
                nn->b[k][j] -= learning_rate * db;
            }
            else if(strcmp(opt, "momentum") == 0){
                nn->momentum_b[k][j] = beta * nn->momentum_b[k][j] + (1.0-beta) * db * learning_rate;
                nn->b[k][j] -= nn->momentum_b[k][j];
            }
            else if(strcmp(opt, "rmsprop") == 0){
                nn->momentum_b[k][j] = beta * nn->momentum_b[k][j] + (1.0-beta) * db * db;
                nn->b[k][j] -= (learning_rate * db)/(sqrt(nn->momentum_b[k][j]) + 1e-6);
            }
            else if(strcmp(opt, "adam") == 0){
                nn->momentum_b[k][j] = beta_1 * nn->momentum_b[k][j] + (1.0-beta_1) * db;
                nn->momentum2_b[k][j] = beta_2 * nn->momentum2_b[k][j] + (1.0-beta_2) * db * db;
                double m_cap = (double)nn->momentum_b[k][j]/(double)(1.0 - pow(beta_1, itr));
                double v_cap = (double)nn->momentum2_b[k][j]/(double)(1.0 - pow(beta_2, itr));
                nn->b[k][j] -= (learning_rate * m_cap)/(sqrt(v_cap) + epsilon);
            }
        }
    }
}


/** Function to train the model for 1 epoch
 * @param nn: struct NeuralNet *, ptr to the neural network
 * @param X_train: double **, input training set
 * @param y_rain: double **, output labels one-hot encoding
 * @param y_train_temp: double *, output labels used for computing accuracy
 * @param activation_fun: char *, activation function
 * @param loss: char *, loss function
 * @param opt: char *, optimizer
 * @param learning_rate: double, learning rate
 * @param num_samples_to_train: int, num samples to train per epoch
 * @param itr: int, current iteration, used for bias correction
 * 
 * @return double *: metrics found
 * */
double* model_train(struct NeuralNet* nn, double** X_train, double** y_train, double* y_train_temp, 
                    char* activation_fun, char* loss, char* opt, double learning_rate,
                    int num_samples_to_train, int itr){

    // Create an array for generating random permutation of training sample indices
    int arr[N_SAMPLES]; //all samples
    for(int i=0;i<N_SAMPLES;i++){
        arr[i] = i;
    }
    shuffle(arr, N_SAMPLES); //shuffle array

    int shuffler[num_samples_to_train]; //only the samples used in this epoch
    for(int i=0;i<num_samples_to_train;i++){
        shuffler[i] = arr[i];
    }

    // Start training the model for 1 epoch and simultaneously calculate the training error and accuracy
    int correct = 0;
    double loss_val = 0.0;

    for(int i=0;i<num_samples_to_train;i++){ //for each sample

        //shuffle(shuffler, num_samples_to_train); //it doesn't seem to have any impact since arr[i] has already been shuffled before

        int idx = -1; //predicted class init
        double max_val = (double)INT_MIN;

        // arr[i] is a random index
        for(int j=1;j<nn->n_neurons_per_layer[0]+1;j++){
            // out[0] is the input layer, load the training set
            nn->out[0][j] = X_train[arr[i]][j-1];
        }
        for(int j=1;j<nn->n_neurons_per_layer[nn->n_layers-1]+1;j++){
            // targets is the desired output, load the labels
            nn->targets[j] = y_train[arr[i]][j-1];
        }

        forward_propagation(nn, activation_fun, loss);
        back_propagation(nn, activation_fun, learning_rate, loss, opt, itr);
        loss_val += calc_loss(nn, loss); // compute and accumulate loss
            
        // compute max and class predicted
        for(int j=1;j<nn->n_neurons_per_layer[nn->n_layers-1]+1;j++){
            if(nn->out[nn->n_layers-1][j] > max_val){
                max_val =nn->out[nn->n_layers-1][j];
                idx = j-1;
            }
        }
        if(idx == (int)y_train_temp[arr[i]]){
            correct++;
        }
    }

    //avg loss
    loss_val /=(double)num_samples_to_train;
    double accuracy = (double)correct/(double)num_samples_to_train;
    static double metrics[2];
    metrics[0] = loss_val;
    metrics[1] = accuracy;
    return metrics;
}


// Function to test the model
double* model_test(struct NeuralNet* nn, double** X_test, double** y_test, double* y_test_temp, char* activation_fun, char* loss){
    int correct = 0;
    double loss_val = 0.0;
    for(int i=0;i<N_TEST_SAMPLES;i++){
        int idx = -1;
        double max_val = (double)INT_MIN;
        for(int j=1;j<nn->n_neurons_per_layer[0]+1;j++){
            nn->out[0][j] = X_test[i][j-1];
        }
        for(int j=1;j<nn->n_neurons_per_layer[nn->n_layers-1]+1;j++){
            nn->targets[j] = y_test[i][j-1];
        }
        forward_propagation(nn, activation_fun, loss);
        loss_val += calc_loss(nn, loss);
            
        for(int j=1;j<nn->n_neurons_per_layer[nn->n_layers-1]+1;j++){
            if(nn->out[nn->n_layers-1][j] > max_val){
                max_val =nn->out[nn->n_layers-1][j];
                idx = j-1;
            }
        }
        if(idx == (int)y_test_temp[i]){
            correct++;
        }
    }
    loss_val /= (double)N_TEST_SAMPLES;
    double accuracy = (double)correct/(double)N_TEST_SAMPLES;
    static double metrics[2];
    metrics[0] = loss_val;
    metrics[1] = accuracy;
    return metrics;
}


int main(){

    // Used for setting a random seed
    srand(time(NULL));
    int seed = rand();

    // Initialize neural network architecture parameters
    int n_layers = 4;
    int n_neurons_per_layer[] = {784, 64, 32, 10};

    // Create and initialize the neural network
    struct NeuralNet* nn = newNet(n_layers, n_neurons_per_layer);
    init_nn(nn);

    // Initialize the learning rate, optimizer, loss, and other hyper-parameters
    double learning_rate = 1e-4;
    double init_lr = 1e-4;
    char* activation_fun = "relu";
    char* loss = "ce";
    char* opt = "adam";
    int num_samples_to_train = 10000;
    int epochs = 5;

    // Fetch the training and test data and pre-process them
    double** X_train = malloc(N_SAMPLES*sizeof(double*));
    for(int i=0;i<N_SAMPLES;i++){
        X_train[i] = malloc(N_DIMS*sizeof(double));
    }
    double** y_train = malloc(N_SAMPLES * sizeof(double*));
    for(int i=0;i<N_SAMPLES;i++){
        y_train[i] = malloc(N_CLASSES * sizeof(double));
    }
    double* y_train_temp = malloc(N_SAMPLES*sizeof(double));
    read_csv_file(X_train, y_train_temp, y_train, "train");
    scale_data(X_train, "train");

    double** X_test = malloc(N_TEST_SAMPLES*sizeof(double*));
    for(int i=0;i<N_TEST_SAMPLES;i++){
        X_test[i] = malloc(N_DIMS*sizeof(double));
    }
    double** y_test = malloc(N_TEST_SAMPLES * sizeof(double*));
    for(int i=0;i<N_TEST_SAMPLES;i++){
        y_test[i] = malloc(N_CLASSES * sizeof(double));
    }
    double* y_test_temp = malloc(N_TEST_SAMPLES*sizeof(double));
    read_csv_file(X_test, y_test_temp, y_test, "test");
    scale_data(X_test, "test");
    normalize_data(X_train, X_test);

    // Initialize file to store metrics info for each epoch
    FILE* file = fopen("metrics_64_32.txt", "w");
    fprintf(file, "train_loss,train_acc,test_loss,test_acc\n");
    
    // Train the model for given number of epoch and test it after every epoch
    for(int itr=0;itr<epochs;itr++){
        double* train_metrics = model_train(nn, X_train, y_train, y_train_temp, activation_fun, loss, opt, learning_rate, num_samples_to_train, itr+1);
        double train_loss = train_metrics[0];
        double train_acc = train_metrics[1];
        double* test_metrics = model_test(nn, X_test, y_test, y_test_temp, activation_fun, loss);
        double test_loss = test_metrics[0];
        double test_acc = test_metrics[1];

        fprintf(file, "%lf,", train_loss);
        fprintf(file, "%lf,", train_acc);
        fprintf(file, "%lf,", test_loss);
        fprintf(file, "%lf\n", test_acc);

        printf("Epoch: %d -> ", itr+1);
        printf("Train loss: %lf, ", train_loss);
        printf("Train Accuracy: %lf, ", train_acc);
        printf("Test loss: %lf, ", test_loss);
        printf("Test Accuracy: %lf\n", test_acc);

        learning_rate = init_lr * exp(-0.1 * (itr+1));

    }

    // Close the file
    fclose(file);

    // Free the dynamically allocated memory
    free_NN(nn);

    return 0;
}
