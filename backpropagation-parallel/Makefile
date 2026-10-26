CC = gcc
CFLAGS = -Wall -Wextra -O3 -g -std=c99 -D_GNU_SOURCE -I.
LDFLAGS = -lm 

# Kernel modules directories
USCTM_DIR = ../pte-entry-switcher/Linux-sys_call_table-discoverer
PES_DIR   = ../pte-entry-switcher/PTE-entry-switcher

# Executable binary names
STD_TARGET         = nn_app_std
ASYNC_TARGET       = nn_app_numa_async
LOCK_TARGET        = nn_app_numa_lock
ATOMIC_TARGET      = nn_app_numa_atomic
TEST_TARGET        = test_pes
CORRECTNESS_TARGET = test_correctness

# Linker wrap flags for NUMA memory remapping
WRAP_FLAGS = -Wl,--wrap=newNet \
             -Wl,--wrap=model_train \
             -Wl,--wrap=init_nn \
             -Wl,--wrap=free_NN \
             -Wl,--wrap=model_test \
             -Wl,--wrap=forward_propagation

NUMA_LDFLAGS = $(LDFLAGS) -lnuma $(WRAP_FLAGS) -pthread

# Build configuration flags
NUMA ?= 0
# Synchronization modes: ASYNC (default), ATOMIC, or LOCK
MODE ?= ASYNC

ifeq ($(NUMA), 1)
    NUMA_FLAGS = -DNUMA_API_ENABLED

    ifeq ($(MODE), ATOMIC)
        NUMA_FLAGS += -DATOMIC_UPDATE
        TARGET = $(ATOMIC_TARGET)
        MSG = NUMA_MODE_ATOMIC_FAD
    else ifeq ($(MODE), LOCK)
        NUMA_FLAGS += -DSYNC_LOCK
        TARGET = $(LOCK_TARGET)
        MSG = NUMA_MODE_SYNC_LOCK
    else
        TARGET = $(ASYNC_TARGET)
        MSG = NUMA_MODE_ASYNC_HOGWILD
    endif
    ALL_TARGETS = $(TARGET)
else
    TARGET = $(STD_TARGET)
    MSG = STANDARD_MODE_ENABLED
    ALL_TARGETS = $(TARGET)
endif

# --- MAIN TARGETS ---

all: $(ALL_TARGETS)
	@echo "--------------------------------------------------"
	@echo "Build completed: $(MSG)"
	@echo "Binary created:  $(TARGET)"
	@echo "--------------------------------------------------"

# --- SHORTHAND TARGETS ---
std: $(STD_TARGET)
async: $(ASYNC_TARGET)
lock: $(LOCK_TARGET)
atomic: $(ATOMIC_TARGET)
all-apps: $(STD_TARGET) $(ASYNC_TARGET) $(LOCK_TARGET) $(ATOMIC_TARGET) $(TEST_TARGET) $(CORRECTNESS_TARGET)

# --- LINKING RULES ---

# 1. Standard Sequential Baseline
$(STD_TARGET): main_std.o neural_network.o
	$(CC) $(CFLAGS) main_std.o neural_network.o -o $(STD_TARGET) $(LDFLAGS)

# 2. NUMA Asynchronous (Hogwild!)
$(ASYNC_TARGET): main_numa.o neural_network.o numa_api_async.o wrappers.o
	$(CC) $(CFLAGS) main_numa.o neural_network.o numa_api_async.o wrappers.o -o $(ASYNC_TARGET) $(NUMA_LDFLAGS)

# 3. NUMA Synchronous (Lock-Based Reduction)
$(LOCK_TARGET): main_numa.o neural_network.o numa_api_lock.o wrappers_lock.o
	$(CC) $(CFLAGS) main_numa.o neural_network.o numa_api_lock.o wrappers_lock.o -o $(LOCK_TARGET) $(NUMA_LDFLAGS)

# 4. NUMA Atomic (Fetch-And-Add / FAD)
$(ATOMIC_TARGET): main_numa.o neural_network.o numa_api_atomic.o wrappers.o
	$(CC) $(CFLAGS) main_numa.o neural_network.o numa_api_atomic.o wrappers.o -o $(ATOMIC_TARGET) $(NUMA_LDFLAGS)

# 5. Test Suites (FIXED: Uses NUMA_LDFLAGS to resolve __real_* symbols)
$(TEST_TARGET): test_numa_nn.o numa_api_async.o neural_network.o wrappers.o
	$(CC) $(CFLAGS) test_numa_nn.o numa_api_async.o neural_network.o wrappers.o -o $(TEST_TARGET) $(NUMA_LDFLAGS)

$(CORRECTNESS_TARGET): test_correctness.o numa_api_async.o neural_network.o wrappers.o
	$(CC) $(CFLAGS) test_correctness.o numa_api_async.o neural_network.o wrappers.o -o $(CORRECTNESS_TARGET) $(NUMA_LDFLAGS)

# --- COMPILATION RULES ---

neural_network.o: neural_network.c
	$(CC) $(CFLAGS) -c neural_network.c -o neural_network.o

main_std.o: main.c
	$(CC) $(CFLAGS) -c main.c -o main_std.o

main_numa.o: main.c
	$(CC) $(CFLAGS) -DNUMA_API_ENABLED -c main.c -o main_numa.o

wrappers.o: wrappers.c
	$(CC) $(CFLAGS) -DNUMA_API_ENABLED -c wrappers.c -o wrappers.o

wrappers_lock.o: wrappers.c
	$(CC) $(CFLAGS) -DNUMA_API_ENABLED -DSYNC_LOCK -c wrappers.c -o wrappers_lock.o

numa_api_async.o: numa_api.c ipc_lock.h
	$(CC) $(CFLAGS) -DNUMA_API_ENABLED -DSYNC_ASYNC -c numa_api.c -o numa_api_async.o

numa_api_lock.o: numa_api.c ipc_lock.h
	$(CC) $(CFLAGS) -DNUMA_API_ENABLED -DSYNC_LOCK -c numa_api.c -o numa_api_lock.o

numa_api_atomic.o: numa_api.c ipc_lock.h
	$(CC) $(CFLAGS) -DNUMA_API_ENABLED -DATOMIC_UPDATE -c numa_api.c -o numa_api_atomic.o

test_numa_nn.o: test_numa_nn.c
	$(CC) $(CFLAGS) -DNUMA_API_ENABLED -c test_numa_nn.c -o test_numa_nn.o

test_correctness.o: test_correctness.c
	$(CC) $(CFLAGS) -DNUMA_API_ENABLED -c test_correctness.c -o test_correctness.o

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
	rm -f $(STD_TARGET) $(ASYNC_TARGET) $(LOCK_TARGET) $(ATOMIC_TARGET) $(TEST_TARGET) $(CORRECTNESS_TARGET) *.o metrics_training.txt
	-@if [ -d $(USCTM_DIR) ]; then cd $(USCTM_DIR) && $(MAKE) clean; fi
	-@if [ -d $(PES_DIR) ]; then cd $(PES_DIR) && $(MAKE) clean; fi

.PHONY: all std async lock atomic all-apps modules-load modules-unload clean