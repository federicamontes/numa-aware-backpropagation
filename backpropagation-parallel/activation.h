#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// Sigmoid activation function
static inline double sigmoid(double x){
    double ans = (double)1/(double)(1 + exp(-x));
    return ans;
}


// Derivative of Sigmoid activation function
static inline double sigmoid_d(double x){
    return sigmoid(x)*(1-sigmoid(x));
}


// ReLU activation function
static inline double relu(double x){
    if(x < 0.0){
        return 0.0;
    }
    return x;
}


// Derivative of ReLU activation function
static inline double relu_d(double x){
    if(x < 0.0){
        return 0.0;
    }
    return 1.0;
}

// Derivative of tanh activation function
static inline double tanh_d(double x){
    double res = 1.0 - tanh(x)*tanh(x);
    return res;
}
