cmake_minimum_required(VERSION 2.8)
project(tiny_lenet)
set(CMAKE_BINARY_DIR ${CMAKE_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

set(TINY_DNN_PATH ${TINY_DNN_PATH})

include_directories("${PROJECT_SOURCE_DIR}" "$ENV{TINY_DNN_PATH}")

add_executable(train ${PROJECT_SOURCE_DIR}/train.cpp)
target_compile_definitions(train PRIVATE CNN_USE_AVX CNN_USE_SSE)
target_compile_options(train PRIVATE -mavx2 -msse3 -O2)
target_link_libraries(train stdc++ m pthread)

add_executable(classify ${PROJECT_SOURCE_DIR}/classify.cpp)
target_compile_definitions(classify PRIVATE CNN_USE_AVX CNN_USE_SSE)
target_compile_options(classify PRIVATE -mavx2 -msse3 -O2)
target_link_libraries(classify stdc++ m pthread)

add_executable(get_kernels ${PROJECT_SOURCE_DIR}/get_kernels.cpp)
target_link_libraries(get_kernels stdc++ m pthread)

link_directories("$ENV{TINY_DNN_PATH}")
