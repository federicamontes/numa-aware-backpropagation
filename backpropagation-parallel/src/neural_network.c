#include "neural_network.h"
#include "activation.h"
#include "utils.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include "read_data.h"

// Initialize constants used in optimizers
const double epsilon = 1e-8;
const double beta = 0.9;
const double beta_1 = 0.9;
const double beta_2 = 0.999;

int num_numa_nodes = 2;




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

    fprintf(stderr, "\n[DEBUG TRAIN] Starting model_train (NN Addr: %p, Samples: %d). n_layers: %d.\n", 
            (void*)nn, num_samples_to_train, nn->n_layers);
   
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

        assert(nn != NULL);
        assert(nn->out != NULL);
        assert(nn->out[0] != NULL);
        assert(arr[i] >= 0);

        // arr[i] is a random index
        for(int j=1;j<nn->n_neurons_per_layer[0]+1;j++){
            assert(j > 0);
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
    fprintf(stderr, "[DEBUG TRAIN] Epoch finished. Final Loss: %f, Accuracy: %f.\n", loss_val, accuracy);
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
