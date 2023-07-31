# CMakeLists.txt
# Minimum required version of CMake is 3.18 since it started supporting CUDA better since than.
cmake_minimum_required(VERSION 3.18)

# specify no particular build architectures
set(CMAKE_CUDA_ARCHITECTURES OFF)
# initialize CUDA_ARCHITECTURES from CMAKE_CUDA_ARCHITECTURES whine CMAKE_CUDA_COMPILER_ID is NVIDIA.
cmake_policy(SET CMP0104 NEW)

# not listing CUDA as a language here but rather enabling it a step later. Current CUDA versions can't deal with too new gcc-versions.
project(assignment2 LANGUAGES C CXX)

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
  message("Again searched")
endif()

# setting some compiler flags.
if(APPLE)
  set( CMAKE_CXX_FLAGS "-W -Wall -Wno-unused-variable  -Wno-unused-parameter")
endif(APPLE)

if(UNIX AND NOT APPLE)
  set( CMAKE_CXX_FLAGS "-W -Wall -Wno-unused-variable  -Wno-unused-parameter -lpthread -pthread")
endif(UNIX AND NOT APPLE)

# switch for choosing which src directory to use (skeleton = OFF, solution = ON)
option(USE_SOLUTION "defines if the solution source should be compiled or the skeleton code should be used" OFF)

# select source directory
if(${USE_SOLUTION})
  set(A2_DIR ${CMAKE_CURRENT_SOURCE_DIR}/solution)
else()
  set(A2_DIR ${CMAKE_CURRENT_SOURCE_DIR}/skeleton)
endif()

# set variables in parent directory, that a2 was found and what to include in linking against it
set(A2_FOUND True PARENT_SCOPE)
set(A2_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/inc)

# add the library containing the exercises
add_library(a2lib SHARED ${A2_DIR}/a2.cu)
target_include_directories(a2lib PUBLIC ${A2_INCLUDE_DIR})
set_target_properties(a2lib PROPERTIES
                            CUDA_SEPARABLE_COMPILATION ON
                            CUDA_STANDARD 14
                            CXX_STANDARD 14
)

# add the executable from the main.cpp
add_executable(a2 ${A2_DIR}/main.cpp)
# enable include of cuda runtime variables
target_include_directories(a2 PUBLIC
                              ${A2_INCLUDE_DIR}
                              ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
set_target_properties(a2 PROPERTIES
                         CUDA_STANDARD 14
                         CXX_STANDARD 14
)
#link libraries
target_link_libraries(a2 PUBLIC a2lib CUDA::cudart)

#set debugflags for cuda code if CMAKE_BUILD_TYPE is set to Debug
IF (CMAKE_BUILD_TYPE MATCHES "Debug")
    target_compile_options(a2lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            -G -src-in-ptx
            >)
    target_compile_options(a2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            -G -src-in-ptx
            >)
ENDIF()
