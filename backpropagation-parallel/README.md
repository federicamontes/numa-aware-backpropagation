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

# Verify dataset files exist
ls -lh mnist_train.csv mnist_test.csv
# If zipped:
# unzip mnist_t.zip

- Load the LKM:
make modules-load (sudo already called inside)

To verify the modules have been loaded: lsmod | grep -E "the_pte_entry_switcher|the_usctm"

- Unload the LKM:
make modules-unload



Use the provided Makefile to compile the project

The Makefile supports compiling different binaries depending on the NUMA flag:
Build Flag / Target           | Generated Binary       		| Description
-------------------------------+------------------------+--------------------------------------------------------------
 make NUMA=0                   | nn_app_std             	| Standard CPU baseline (heap malloc, sequential, no wrappers)
 make NUMA=1                   | nn_app_numa_async, test_pes    | NUMA engine with 2MB mmap slabs, linker wrappers & PMD test
 make NUMA=1 test_correctness  | test_correctness       	| Standalone bit-for-bit mathematical parity test
 make clean                    | (removes *.o, binaries)	| Cleans all user-space object files and binaries

# Build the Standard Baseline (nn_app_std)
make clean
make NUMA=0

# Build the NUMA Engine (nn_app_numa_async and test_pes)
(make clean)
make NUMA=1

* Compiles with -DNUMA_API_ENABLED, links numa_api.o and wrappers.o, and applies linker hooks
(-Wl,--wrap=newNet, --wrap=model_train, --wrap=init_nn, --wrap=free_NN, --wrap=model_test, --wrap=forward_propagation).


## Parameters of executables
Application Binaries: nn_app_std and nn_app_numa_async:
./<binary> [lr] [init_lr] [act] [loss] [opt] [samples] [batch_size] [epochs] [workers (numa only)]

Positional Arg	Parameter			Example Value	Description
argv[1]		Learning Rate (lr)		0.0001		Initial step size for gradient updates
argv[2]		Decay Base LR (init_lr)		0.0001		Base LR for exponential decay
argv[3]		Activation Function (act)	relu		Hidden layer activation (relu, tanh, or sigmoid)
argv[4]		Loss Function (loss)		ce		Loss criterion (ce for Cross-Entropy, mse for Mean Squared Error)
argv[5]		Optimizer (opt)			adam		Optimization algorithm (adam or sgd)
argv[6]		Training Samples (samples)	10000		Number of training samples per epoch (up to 60,000)
argv[7]		Mini-Batch Size (batch_size)	1 or 32		Gradient accumulation window
argv[8]		Epochs (epochs)			5		Total training iterations over the dataset
argv[9]		Workers (workers)		1, 2, 4		Number of parallel worker processes (NUMA mode only)

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
