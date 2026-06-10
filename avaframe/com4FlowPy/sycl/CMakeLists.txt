cmake_minimum_required(VERSION 3.20)
project(com4FlowPy_sycl LANGUAGES CXX)

# Optional cache variable for specifying a compatible GCC toolchain version
set(GCC_INSTALL_DIR "" CACHE PATH "Optional path to GCC installation directory (e.g., /usr/lib/gcc/x86_64-linux-gnu/15)")
if(GCC_INSTALL_DIR)
    add_compile_options("--gcc-install-dir=${GCC_INSTALL_DIR}")
    add_link_options("--gcc-install-dir=${GCC_INSTALL_DIR}")
endif()

# Let optimization flags be configured by the build type (e.g., CMAKE_BUILD_TYPE=Release)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(AdaptiveCpp REQUIRED)

# Create pybind11 library module
pybind11_add_module(sycl_core MODULE
    sycl_core.cpp
)

# System-independent include search for IDE support
find_path(ACPP_IDE_INCLUDE_DIR
    NAMES sycl/sycl.hpp
    HINTS ENV ACPP_DIR ENV AdaptiveCpp_DIR
    PATH_SUFFIXES include include/AdaptiveCpp
)
message(STATUS "ACPP_IDE_INCLUDE_DIR: ${ACPP_IDE_INCLUDE_DIR}")
if(ACPP_IDE_INCLUDE_DIR)
    target_compile_options(sycl_core PRIVATE "-isystem" "${ACPP_IDE_INCLUDE_DIR}")
endif()

# Link pybind11 module with AdaptiveCpp
target_link_libraries(sycl_core PRIVATE pybind11::module)
add_sycl_to_target(TARGET sycl_core SOURCES sycl_core.cpp)

# Output build properties for user info
message(STATUS "Build target: sycl_core")
message(STATUS "AdaptiveCpp Compiler & Flags applied automatically by CMake.")