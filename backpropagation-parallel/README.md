# Neural-Network-using-C

In this project, the backpropagation algorithm in feedforward neural networks is implemented using C. A data structure to store and manipulate neural network weights is created. Algorithms are designed to perform forward propagation, back propagation, and update weights. The user can specify the number of hidden layers, number of neurons in each hidden layer, the loss function, the optimizer, and the activation function to be used. Backpropagation is an algorithm used for updating weights of an artificial neural network using gradient descent optimization. Given a neural network and a loss function, the algorithm calculates the gradient of the error function with respect to each of the neural network’s weights. The calculation of errors takes place in a backward direction from the output layer, through the hidden layers to the input layer. Using the obtained errors, the weights are updated. For demonstration, a neural network model was trained and tested on the MNIST dataset for handwritten digit recognition.


Code taken from: https://github.com/sm823zw/Neural-Network-Backprop-using-C/tree/main

The implementation follows a Stochastic Gradient Descent (SGD) approach, processing one shuffled sample at a time (per the original code structure) to perform the Forward Propagation and Back Propagation steps, not using Mini-Batch as PyTorch or TensorFlow.

* Prerequisites

To build and run this project, you need:

    A Compiler: gcc (GNU Compiler Collection) is used in the provided Makefile.

    Math Library: The standard C math library (-lm flag is required).

    Pthread library: -pthread required for future parallelization of this code 

* Building the Project

Use the provided Makefile to compile the source file (neural_network.c)

	Clean generated files:
	make clean
	# Removes the executable (nn_original) and object files (*.o)

    Compile the executable:
	make
	# Creates the executable: nn_original

	Unzip mnist dataset → it contains training and testing set
You can then run the executable:
-	./nn_original

For now it does not take parameters as the configuration (optimizer, activation function..) is hardcode in main function

It does create an output file called metrics_63_32.txt which format is
-	train_loss,train_acc,test_loss,test_acc


# New structure of the project
.
├── CMakeLists.txt
├── include/
│   ├── activation.h        # Math: sigmoid, relu
│   ├── memory_manager.h    # System: mmap, allocation
│   ├── neural_network.h    # Core: forward propagation, backpropagation, model train and test
│   ├── read_data.h         # IO: CSV parsing
│   └── utils.h             # Helpers: randn, shuffle
├── src/
│   ├── activation.c
│   ├── memory_manager.c
│   ├── neural_network.c
│   ├── read_data.c
│   └── utils.c
├── test/
│   └── main_test.c         # High-level setup and calling trainer
├── input/
│   └── *.csv               # Dataset files
└── build/                  # Created by CMake
