CC = gcc
CFLAGS = -Wall -Wextra -O3 -g -std=c99 -I.
LDFLAGS = -lm 

USCTM_DIR = ../pte-entry-switcher/Linux-sys_call_table-discoverer
PES_DIR   = ../pte-entry-switcher/PTE-entry-switcher

TARGET = nn_app
SRC = neural_network.c main.c

ifeq ($(NUMA), 1)
	CFLAGS += -DNUMA_API_ENABLED
	SRC += wrappers.c
	LDFLAGS += -lnuma -Wl,--wrap=model_train -Wl,--wrap=newNet -Wl,--wrap=init_nn -Wl,--wrap=free_NN
	STATUS = "MODALITA_NUMA_ATTIVA"
	K_MODULES_DEP = k-modules
else
	STATUS = "MODALITA_STANDARD"
	K_MODULES_DEP = 
endif

all: $(K_MODULES_DEP) $(TARGET)

$(TARGET): $(SRC)
	@echo "--- BUILDING APP: $(STATUS) ---"
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

k-modules:
	@echo "--- Preparazione Moduli Kernel (SOLO PER NUMA) ---"
	$(MAKE) -C $(USCTM_DIR)
	@if ! lsmod | grep -q "the_usctm"; then \
		echo "Esecuzione load.sh in $(USCTM_DIR)..."; \
		cd $(USCTM_DIR) && sudo ./load.sh; \
	fi
	$(MAKE) -C $(PES_DIR)
	echo "Esecuzione mount in $(PES_DIR)..."; \
	SCT_ADDR=$$(cat /sys/module/the_usctm/parameters/sys_call_table_address); \
	$(MAKE) -C $(PES_DIR) mount ; \
	

	
unload:
	@echo "--- Rimozione Moduli ---"
	-sudo rmmod the_pte_entry_switcher 2>/dev/null || true
	@if [ -d $(USCTM_DIR) ]; then \
		cd $(USCTM_DIR) && sudo ./unload.sh 2>/dev/null || sudo rmmod the_usctm 2>/dev/null || true; \
	fi

clean:
	rm -f $(TARGET) *.o
	-$(MAKE) -C $(USCTM_DIR) clean
	-$(MAKE) -C $(PES_DIR) clean

.PHONY: all k-modules unload clean