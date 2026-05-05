# Neural-Network-using-C

In this project, the backpropagation algorithm in feedforward neural networks is implemented using C. A data structure to store and manipulate neural network weights is created. Algorithms are designed to perform forward propagation, back propagation, and update weights. The user can specify the number of hidden layers, number of neurons in each hidden layer, the loss function, the optimizer, and the activation function to be used. Backpropagation is an algorithm used for updating weights of an artificial neural network using gradient descent optimization. Given a neural network and a loss function, the algorithm calculates the gradient of the error function with respect to each of the neural network’s weights. The calculation of errors takes place in a backward direction from the output layer, through the hidden layers to the input layer. Using the obtained errors, the weights are updated. For demonstration, a neural network model was trained and tested on the MNIST dataset for handwritten digit recognition.


Code taken from: https://github.com/sm823zw/Neural-Network-Backprop-using-C/tree/main

The implementation follows a Stochastic Gradient Descent (SGD) approach, processing one shuffled sample at a time (per the original code structure) to perform the Forward Propagation and Back Propagation steps, using Mini-Batch as PyTorch or TensorFlow.

* Prerequisites

To build and run this project, you need:

    A Compiler: gcc (GNU Compiler Collection) is used in the provided Makefile.

    Math Library: The standard C math library (-lm flag is required).

    NUMA library: -lnuma required  
 	- Install on Ubuntu/Debian: sudo apt install libnuma-dev

* Building the Project

Use the provided Makefile to compile the project

	Clean generated files:
	make clean
	# Removes the executable and object files (*.o)

## Compile Sequential Mode
	make NUMA=0 or simply make
	# Creates the executable: nn_app_std

	Unzip mnist dataset → it contains training and testing set

## Compile NUMA-aware Mode
	make NUMA=1
	# Creates the executable: nn_app_numa

	Load Modules:
	make modules-load

	Unload Modules:
	make modules-unload

You can then run the executable:
#	./nn_app_std if NUMA=0
#   ./nn_app_numa if NUMA=1


It does create an output file called metrics_63_32.txt which format is
#	train_loss,train_acc,test_loss,test_acc


```bash
.
├── backpropagation-original/        # BASELINE: Standard sequential implementation
│   ├── neural_network.c             # Original ML logic (non-instrumented)
│   ├── activation.h                 # Standard activation functions
│   ├── read_data.h                  # MNIST CSV loader
│   └── Makefile                     # Build script for baseline execution
│
├── backpropagation-parallel/        # INSTRUMENTED: NUMA-aware parallel framework
│   ├── main.c                       # Orchestrator: handles fork(), pinning, and slabs
│   ├── neural_network.c             # Instrumented ML logic for parallel execution
│   ├── numa_api.c                   # Interface for LKM syscall communication
│   ├── wrappers.c                   # Linker-level wrappers for transparent execution
│   ├── activation.h                 # Shared math headers
│   ├── read_data.h                  # Shared data loading logic
│   └── Makefile                     # Build script for std/numa targets (nn_app_std/numa)
│
├── pte-entry-switcher/              # KERNEL: LKM support infrastructure
│   ├── PTE-entry-switcher/          # Core module for dynamic Page Table manipulation
│   │   └── pte-entry-switcher.c     # Implementation of the custom PTE-switch syscall
│   └── Linux-sys_call_table.../     # Kernel utility to locate the sys_call_table
│       ├── lib/                     # Support libraries for table discovery
│       └── usctm.c                  # System Call Table Modifier implementation
│
├── Project Report.pdf               # Comprehensive technical documentation
├── README.md                        # Project overview and architecture (This file)
└── TODO.txt                         # Development roadmap and pending tasks

```