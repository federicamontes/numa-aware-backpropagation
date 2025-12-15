TARGET = nn_parallel


CC = gcc

CFLAGS = -Wall -Wextra -O3 -g -std=c99 -I.


LDFLAGS = -lm -pthread -lnuma



all: $(TARGET)


$(TARGET): neural_network.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)


clean:
	rm -f $(TARGET) *.o


run: $(TARGET)
	./$(TARGET) 1

.PHONY: all clean run
