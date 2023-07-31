# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)

# specify no particular build architectures
set(CMAKE_CUDA_ARCHITECTURES OFF)
# initialize CUDA_ARCHITECTURES from CMAKE_CUDA_ARCHITECTURES whine CMAKE_CUDA_COMPILER_ID is NVIDIA.
cmake_policy(SET CMP0104 NEW)

project(assignments LANGUAGES C CXX)

# make sure to set the CUDA_HOST_COMPILER to a g++ version not larger than 10.
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 11.0)
    message("SETTING COMPILER TO g++-10")
    set(CMAKE_CUDA_HOST_COMPILER g++-10)
  else()
    message("USINGE CXX_COMPILER")
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  endif()
endif()

# enable CUDA language.
enable_language(CUDA)

if(NOT DEFINED CUDAToolkit_FOUND)
  find_package(CUDAToolkit)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/a1")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/a1)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/a2")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/a2)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/a3")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/a3)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gpu-grader")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu-grader)
endif()
