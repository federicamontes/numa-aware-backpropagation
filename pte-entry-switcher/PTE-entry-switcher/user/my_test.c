#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <fcntl.h>
#include <pthread.h>
#include <assert.h>


#define PES 156 //this depends on what the kernel tells you when mounting the vtpmo module

int pes(unsigned long x, int a, int b){
	return syscall(PES,x,a,b);
}

#define PAGE_SIZE 4096
#define NUM_PAGES 4
#define PROTECT_PAGES 4
#define N 4

static unsigned char *init_address = (unsigned char *)(10LL << 39);


void print_indices(unsigned long long va){
    printf("VA = 0x%llx\n", va);
    printf(" pml4 = %llu\n", (va >> 39) & 0x1FF);
    printf(" pdp  = %llu\n", (va >> 30) & 0x1FF);
    printf(" pde  = %llu\n", (va >> 21) & 0x1FF);
    printf(" pte  = %llu\n", (va >> 12) & 0x1FF);
}

#define SCALE 8ULL //default is 4 in segment_shift but cannot allocate array with config parameter
#define NUM_PAGES_PER_SEGMENT ((512ULL * 512ULL)>>SCALE)

#define PER_LP_PREALLOCATED_MEMORY (NUM_PAGES_PER_SEGMENT * PAGE_SIZE ) /// This should be power of 2 multiplied by a page size. This is 1GB per LP.

#define NUM_PAGES_PER_MMAP	(NUM_PAGES_PER_SEGMENT >> 2)
#define MAX_MMAP			(NUM_PAGES_PER_MMAP * PAGE_SIZE) /// This is the maximum amount of memory that a single mmap() call is able to serve. TODO: this should be checked within configure.ac
#define NUM_MMAP			1//(((PER_LP_PREALLOCATED_MEMORY) / MAX_MMAP) <= 0 ? 1 : ((PER_LP_PREALLOCATED_MEMORY) / MAX_MMAP))  


#define NUM_TARGET_PAGES 	NUM_PAGES_PER_SEGMENT
#define ZONE_SIZE			(512*4096)
#define PDE_SIZE (2 * 1024 * 1024) // 2 MiB per PDE

#define BASE (2048 * 4 * 1024)
unsigned long base = BASE;
unsigned long base1 = BASE + 512 * 4 * 1024;;

int main(int argc, char** argv){
        
    int res, res2;
	char c;
	void *addr;

	char * zone_A = NULL;
	char * zone_B = NULL;

	void *the_address = init_address + PER_LP_PREALLOCATED_MEMORY * 1;
	void *the_address1 = the_address + PDE_SIZE;

	print_indices(0x00800000ULL);
    print_indices((unsigned long)the_address);

    printf("num target pages %lu -- PER_LP_PREALLOCATED_MEMORY %lu \n", NUM_PAGES_PER_SEGMENT, PER_LP_PREALLOCATED_MEMORY);
    printf("zone size %lu\n", PDE_SIZE);

    printf("base is %p - base1 is %p \n",the_address,the_address1);
    fflush(stdout);

	// Single mmap covering both PDEs
    void *lp_mem = mmap(the_address, PER_LP_PREALLOCATED_MEMORY,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS,
                        -1, 0);
    if (lp_mem == MAP_FAILED){
        perror("mmap failed");
        exit(EXIT_FAILURE);
    }

    // Assign zones inside the mapped region
    zone_A = (char*)lp_mem;          // first PDE
    //zone_B = (char*)lp_mem + PDE_SIZE; // second PDE
    zone_B = (char *)lp_mem[PDE_SIZE];

    sprintf(zone_A,"%s","Francesco");
	sprintf(zone_B,"%s","Quaglia");

	redo:
	printf("zone A has content: %s\n",zone_A);
	printf("zone B has content: %s\n",zone_B);
	fflush(stdout);

	pes(lp_mem,0,1);

	//sleep(1);

	printf("zone A has content: %s\n",zone_A);
	printf("zone B has content: %s\n",zone_B);
	fflush(stdout);

	pes(lp_mem,1,0);

	//sleep(1);

	printf("zone A has content: %s\n",zone_A);
	printf("zone B has content: %s\n",zone_B);
	fflush(stdout);
	return 0;
}