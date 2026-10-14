#ifndef ipc_lock_H
#define ipc_lock_H

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#define CACHELINE_PAD 128

// ============================================================================
// struct NodeIPCContext
// Strict 128-byte alignment prevents L1/L2 prefetcher false sharing between
// the mutex, the two phase barriers, and reduction variables.
// ============================================================================
typedef struct {
    // ------------------------------------------------------------------------
    // Cache Line 0 (Bytes 0 - 127): Process-Shared Mutex
    // ------------------------------------------------------------------------
    pthread_mutex_t lock __attribute__((aligned(CACHELINE_PAD)));

    // ------------------------------------------------------------------------
    // Cache Line 1 (Bytes 128 - 255): Barrier 1 (Gradient Accumulation)
    // ------------------------------------------------------------------------
    pthread_barrier_t barrier_accumulate __attribute__((aligned(CACHELINE_PAD)));

    // ------------------------------------------------------------------------
    // Cache Line 2 (Bytes 256 - 383): Barrier 2 (Parameter Update)
    // ------------------------------------------------------------------------
    pthread_barrier_t barrier_update __attribute__((aligned(CACHELINE_PAD)));

    // ------------------------------------------------------------------------
    // Cache Line 3 (Bytes 384 - 511): Inter-Process Batch Metrics & Shuffling
    // ------------------------------------------------------------------------
    __attribute__((aligned(CACHELINE_PAD))) double epoch_loss;
    double epoch_correct;
    int current_batch_size;
    int total_samples;
    int* shared_sample_order; // Pointer to shared shuffle permutation

    // ------------------------------------------------------------------------
    // Cache Line 4 (Bytes 512+): Shared Reduction Buffer Pointers
    // ------------------------------------------------------------------------
    __attribute__((aligned(CACHELINE_PAD))) double*** shared_dw;
    double** shared_db;
    size_t total_mmap_bytes;
} NodeIPCContext;

#endif // ipc_lock_H