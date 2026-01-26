cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkHogwild C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find system libraries
find_library(LIBNUMA numa REQUIRED)
find_library(LIBM m REQUIRED)

add_definitions(-DINPUT_PATH="${CMAKE_SOURCE_DIR}/input/")
add_compile_options(-g)

# Make headers available to all subdirectories
include_directories(include)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(test)
