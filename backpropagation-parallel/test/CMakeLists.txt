# Create the executable from test_main.c
add_executable(nn_test test_main.c)
add_executable(test_parallel test_parallel.c)

# Link the executable with our library and system dependencies
target_link_libraries(nn_test nn_core ${LIBNUMA} ${LIBM})
target_link_libraries(test_parallel nn_core ${LIBNUMA} ${LIBM})

# Optimization flags for the test runner
target_compile_options(nn_test PRIVATE -O3 -march=native)
target_compile_options(test_parallel PRIVATE -O3 -march=native)
