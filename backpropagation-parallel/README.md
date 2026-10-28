# NUMA-Aware Parallel Neural Network Engine in C

In this project, the backpropagation algorithm in feedforward neural networks is implemented using standard C. A data structure to store and manipulate neural network weights is created. Algorithms are designed to perform forward propagation, back propagation, and update weights using Stochastic Gradient Descent (SGD) and Adam optimizers with mini-batching. The user can specify the number of hidden layers, number of neurons in each hidden layer, the loss function, the optimizer, and the activation function to be used. For demonstration, a neural network model is trained and tested on the MNIST dataset for handwritten digit recognition.

**Novel Contribution:** This repository introduces a high-performance **NUMA-aware parallelization engine**. Using a custom Linux Kernel Module (LKM), it performs zero-copy $O(1)$ Page Middle Directory (PMD) switching at 2MB HugeTLB boundaries. This physically isolates process-private working buffers while allowing shared model parameters to be synchronized across multiple NUMA sockets. The parallelization is achieved transparently via GCC linker wrapping (`-Wl,--wrap`), requiring **zero modifications** to the standard serial machine learning logic.

Original sequential code adapted from: [https://github.com/sm823zw/Neural-Network-Backprop-using-C/tree/allocation_nn](https://github.com/sm823zw/Neural-Network-Backprop-using-C/tree/allocation_nn)

## Prerequisites

To build and run this project, you need a Linux environment with the following:

*   **Compiler:** `gcc` (GNU Compiler Collection)

*   **Math Library:** Standard C math library (`-lm` flag)

*   **NUMA Library:** `libnuma` for memory policies and hardware discovery (`-lnuma` flag)
    *   Install on Ubuntu/Debian: `sudo apt update && sudo apt install build-essential linux-headers-$(uname -r) libnuma-dev`

---

## Building the Project

### 1. Dataset Preparation
Ensure the MNIST datasets are present and uncompressed in the `backpropagation-parallel` directory:

```bash
ls -lh mnist_train.csv mnist_test.csv
# If zipped: unzip mnist_t.zip
```

### 2. Kernel Module Management (LKM)
**Compile and load the LKM into the kernel (prompts for sudo)**

```bash
make modules-load
```

**Verify the modules have been loaded successfully:**

```bash
lsmod | grep -E "the_pte_entry_switcher|the_usctm"
```

**Unload the LKM when finished:**

```bash
make modules-unload
```

### 3. Compiling the Application Binaries

Use the provided Makefile to compile the project. The build system supports compiling the standard sequential baseline alongside three distinct parallel synchronization modes.

| Make Command      | Generated Binary      | Description |
|---                |---                    |          ---|
| `make std`        | `nn_app_std`          | Standard CPU baseline (heap malloc, sequential, no wrappers). |
| `make async`      | `nn_app_numa_async`   | NUMA engine: asynchronous (Hogwild!-like) lock-free updates. |
| `make atomic`     | `nn_app_numa_atomic`  | NUMA engine: atomic FAD (Fetch-And-Add) CAS loop updates. |
| `make lock`       | `nn_app_numa_lock`    | NUMA engine: synchronous lock-based reduction (2-phase barriers). |
| `make tests`      | `test_pes`, `test_correctness` | Compiles the hardware and mathematical validation suites. |
| `make all-apps`   | All above             | Compiles all binaries and validation suites. |
| `make clean`      |       —               | Removes all user-space object files (`.o`) and binaries. |

**Example build:**

```bash
make clean
make all-apps
```

## Running
All application binaries (`nn_app_std`, `nn_app_numa_async`, `nn_app_numa_atomic`, `nn_app_numa_lock`) share the exact same CLI positional arguments. 

**Format:**

```bash
./<binary> [lr] [init_lr] [act] [loss] [opt] [samples] [batch_size] [epochs] [workers]
```

**Positional Arguments:**

| Index | Parameter | Example Value | Description |
|---|---|---|---|
| `argv[1]` | Learning Rate (`lr`) | `0.0001` | Initial step size for gradient updates. |
| `argv[2]` | Decay Base LR (`init_lr`) | `0.0001` | Base learning rate for exponential decay. |
| `argv[3]` | Activation Function (`act`) | `relu` | Hidden layer activation function (`relu`, `tanh`, or `sigmoid`). |
| `argv[4]` | Loss Function (`loss`) | `ce` | Loss criterion (`ce` for Cross-Entropy, `mse` for Mean Squared Error). |
| `argv[5]` | Optimizer (`opt`) | `adam` | Optimization algorithm (`adam` or `sgd`). |
| `argv[6]` | Training Samples | `10000` | Number of training samples per epoch (up to 60,000). |
| `argv[7]` | Mini-Batch Size | `16` | Gradient accumulation window (*ignored by `nn_app_std`*). |
| `argv[8]` | Epochs | `5` | Total number of training iterations over the dataset. |
| `argv[9]` | Workers | `2` or `4` | Number of parallel worker processes (*ignored by `nn_app_std`*). |

**Example runs:**

**Sequential Baseline**
`./nn_app_std 0.0001 0.0001 relu ce adam 10000 16 5 1`

**Parallel Version (Atomic, 4 Workers, Batch 32)**
`./nn_app_numa_atomic 0.002 0.002 relu ce adam 10000 32 5 4`

## Output:
The program will print epoch-by-epoch metrics to the console and generate an output file named metrics_training.txt with the following CSV format:

*epoch,train_loss,train_acc,test_loss,test_acc*

## Testing & Validation Suite

The project includes two *validation tests* to ensure the custom kernel memory management introduces no hardware faults or mathematical corruption.

Compile the tests using: `make tests` (or `make all-apps`).

### 1. Hardware & Memory Isolation Test (`test_pes`)
Validates that the custom `pes` syscall correctly swaps PMD entries and that sibling workers on the same NUMA node do not clobber each other's private working memory.

**Usage:** `./test_pes [num_numa_nodes] [num_workers]`

```bash
# Test 4 workers sharing 2 NUMA sockets
./test_pes 2 4
```

**Expected Output**: The test verifies that workers read the correct shared canaries for their respective sockets while maintaining perfectly isolated private writes. It should exit with >>> RESULT: SUCCESS <<<.

###2. Mathematical Parity Test (test_correctness)

Proves that the NUMA virtual memory layout and parallel gradient accumulation compute the exact same floating-point math as the standard heap-allocated sequential baseline.

**Usage:**
`./test_correctness`

**Expected Output**: The suite runs identical inputs through Forward Propagation, Backpropagation Deltas, Gradient Accumulation, and 100 steps of Adam optimization. It calculates the maximum absolute difference between the standard model and the NUMA model. It should exit with Max Difference = 0.000000000000000e+00 and >>> FINAL RESULT: 100% BIT-FOR-BIT MATHEMATICAL PARITY CONFIRMED <<<.

## 📁 Directory Structure

```text
.
├── backpropagation-original/        # BASELINE: Standard sequential implementation (reference)
│   ├── neural_network.c             # Original ML logic 
│   ├── activation.h                 # Standard activation functions
│   ├── read_data.h                  # MNIST CSV loader
│   └── Makefile                     # Build script for baseline execution
│
├── backpropagation-parallel/        # NUMA-AWARE PARALLEL FRAMEWORK
│   ├── main.c                       # Application entry point (pure C, should be transparent)
│   ├── neural_network.c             # Standard ML logic (pure C, sequential baseline)
│   ├── numa_api.c                   # Memory allocation, NUMA-aware API, and synchronization logic
│   ├── wrappers.c                   # GCC linker-level wrappers for execution transparency
│   ├── ipc_lock.h                   # 128-byte padded shared memory structures (Barriers/Mutexes)
│   ├── activation.h                 # Shared math headers
│   ├── read_data.h                  # Shared data loading logic
│   ├── test_pes.c                   # Unit test: Kernel PMD switching and memory isolation
│   ├── test_correctness.c           # Unit test: Deterministic mathematical parity validation
│   └── Makefile                     # Advanced build script for std/async/atomic/lock targets
│
├── pte-entry-switcher/              # KERNEL: LKM support infrastructure
│   ├── PTE-entry-switcher/          # Core module for dynamic Page Table manipulation
│   │   └── pte-entry-switcher.c     # Implementation of the custom PTE-switch syscall (156)
│   └── Linux-sys_call_table.../     # Kernel utility to locate the sys_call_table
│       ├── lib/                     # Support libraries for table discovery
│       └── usctm.c                  # System Call Table Modifier implementation
│
├── Project Report.pdf               # Comprehensive technical documentation
├── README.md                        # Project overview and architecture (This file)
└── TODO.txt                         # Development roadmap and pending tasks

```
