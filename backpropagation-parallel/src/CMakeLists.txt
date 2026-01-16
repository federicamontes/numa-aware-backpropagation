# Define the library source files
set(SRC_FILES 
    activation.c
    memory_manager.c
    neural_network.c
    read_data.c
    utils.c
)

# Create a static library named 'nn_core' using the correct variable
add_library(nn_core STATIC ${SRC_FILES})

# Link libraries
target_link_libraries(nn_core ${LIBNUMA} ${LIBM})