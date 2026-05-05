# NUMA-Aware Backpropagation with Dynamic PTE Switching


## Project Structure

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

- backpropagation-original/:
The "Ground Truth" codebase. This is used to ensure that any optimizations in the parallel version do not compromise the mathematical integrity of the backpropagation algorithm. This is the original code.

- backpropagation-parallel/:
It has been instrumented to interact with the Linux Kernel. This can be run both in sequential mode and in parallel mode depending on compilation flags.

- pte-entry-switcher/:
The Linux Kernel Module (LKM) that supports PTE switching.

