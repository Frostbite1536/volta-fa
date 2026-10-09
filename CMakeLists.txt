cmake_minimum_required(VERSION 3.24)
project(volta_fa LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA is optional: without a CUDA compiler (dev laptop) only the CPU reference
# and tests are built. Kernels still get static sm_70 reports via bin/sm70-report.py.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  # Must be set before enable_language(CUDA), or CMake picks the compiler default.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70)
  endif()
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
    message(FATAL_ERROR "volta-fa targets sm_70 only (INV-ARCH-001), got ${CMAKE_CUDA_ARCHITECTURES}")
  endif()
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    message(FATAL_ERROR "CUDA ${CMAKE_CUDA_COMPILER_VERSION} dropped sm_70; use CUDA 12.x (INV-ARCH-003)")
  endif()
  set(CMAKE_CUDA_STANDARD 17)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
                      $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
  set(VFA_HAVE_CUDA ON)
else()
  message(STATUS "volta-fa: no CUDA compiler found, building CPU reference and tests only")
  set(VFA_HAVE_CUDA OFF)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()

enable_testing()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/reference/CMakeLists.txt)
  add_subdirectory(src/reference)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
  add_subdirectory(tests)
endif()
# CUDA targets (src/attention, src/p2p, bench) are added under if(VFA_HAVE_CUDA) as they land.
