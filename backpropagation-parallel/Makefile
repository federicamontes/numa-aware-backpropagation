CC = gcc
CFLAGS = -Wall -Wextra -O3 -g -std=c99 -I.
LDFLAGS = -lm 

# Kernel modules directories
USCTM_DIR = ../pte-entry-switcher/Linux-sys_call_table-discoverer
PES_DIR   = ../pte-entry-switcher/PTE-entry-switcher

# Test executable name
TEST_TARGET = test_pes

# Default: NUMA disabled
NUMA ?= 0

# Base objects for the main application
OBJ = main.o neural_network.o

ifeq ($(NUMA), 1)
    # Target name for NUMA mode
    TARGET = nn_app_numa
    OBJ += numa_api.o wrappers.o
    NUMA_FLAGS = -DNUMA_API_ENABLED
    LDFLAGS += -lnuma \
               -Wl,--wrap=newNet \
               -Wl,--wrap=model_train \
               -Wl,--wrap=init_nn \
               -Wl,--wrap=free_NN \
               -Wl,--wrap=model_test \
               -Wl,--wrap=forward_propagation
    
    ALL_TARGETS = $(TARGET) $(TEST_TARGET)
    MSG = NUMA_MODE_ENABLED
else
    # Target name for Standard mode
    TARGET = nn_app_std
    ALL_TARGETS = $(TARGET)
    MSG = STANDARD_MODE_ENABLED
endif

# --- MAIN TARGETS ---

all: $(ALL_TARGETS)
	@echo "--------------------------------------------------"
	@echo "Build completed: $(MSG)"
	@echo "Binary name: $(TARGET)"
	@echo "--------------------------------------------------"

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(TARGET) $(LDFLAGS)

$(TEST_TARGET): test_numa_nn.o numa_api.o neural_network.o wrappers.o
	$(CC) $(CFLAGS) test_numa_nn.o numa_api.o neural_network.o wrappers.o -o $(TEST_TARGET) $(LDFLAGS) -lnuma

# --- COMPILATION RULES ---

neural_network.o: neural_network.c
	$(CC) $(CFLAGS) -c neural_network.c -o neural_network.o

numa_api.o: numa_api.c
	$(CC) $(CFLAGS) $(NUMA_FLAGS) -c numa_api.c -o numa_api.o

wrappers.o: wrappers.c
	$(CC) $(CFLAGS) $(NUMA_FLAGS) -c wrappers.c -o wrappers.o

main.o: main.c
	$(CC) $(CFLAGS) $(NUMA_FLAGS) -c main.c -o main.o

test_numa_nn.o: test_numa_nn.c
	$(CC) $(CFLAGS) $(NUMA_FLAGS) -c test_numa_nn.c -o test_numa_nn.o

# --- KERNEL MODULES MANAGEMENT ---

modules-load:
	@echo "--- Preparing USCTM (Syscall Table Discoverer) ---"
	cd $(USCTM_DIR) && $(MAKE) clean && $(MAKE)
	@echo "--- Loading USCTM ---"
	@if ! lsmod | grep -q "the_usctm"; then \
		cd $(USCTM_DIR) && sudo ./load.sh; \
	fi
	@echo "--- Preparing PES (PTE Entry Switcher) ---"
	cd $(PES_DIR) && $(MAKE) clean && $(MAKE)
	@echo "--- Mounting PES ---"
	cd $(PES_DIR) && $(MAKE) mount

modules-unload:
	@echo "--- Removing Kernel Modules ---"
	-sudo rmmod the_pte_entry_switcher 2>/dev/null || true
	@if [ -d $(USCTM_DIR) ]; then \
		cd $(USCTM_DIR) && sudo ./unload.sh 2>/dev/null || sudo rmmod the_usctm 2>/dev/null || true; \
	fi

clean:
	@echo "--- Cleaning up project binaries and objects ---"
	rm -f nn_app_std nn_app_numa $(TEST_TARGET) *.o
	-@if [ -d $(USCTM_DIR) ]; then cd $(USCTM_DIR) && $(MAKE) clean; fi
	-@if [ -d $(PES_DIR) ]; then cd $(PES_DIR) && $(MAKE) clean; fi

.PHONY: all modules-load modules-unload clean