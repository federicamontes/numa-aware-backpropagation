#include "activation.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_d(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double relu(double x) {
    return (x < 0.0) ? 0.0 : x;
}

double relu_d(double x) {
    return (x < 0.0) ? 0.0 : 1.0;
}

double tanh_d(double x) {
    double t = tanh(x);
    return 1.0 - t * t;
}