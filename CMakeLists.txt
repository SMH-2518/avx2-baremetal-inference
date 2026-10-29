cmake_minimum_required(VERSION 3.15)
project(BareMetalInference VERSION 1.0.0 LANGUAGES CXX)

# Enforce strict C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Core optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wpedantic")

# Safely check for hardware-specific AVX2 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_NATIVE)

if(COMPILER_SUPPORTS_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
else()
    message(WARNING "AVX2 flag not supported by this compiler.")
endif()

if(COMPILER_SUPPORTS_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Define the executable
add_executable(engine engine.cpp)