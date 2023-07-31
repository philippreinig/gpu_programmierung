# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(assignment1 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)

if(APPLE)
  set( CMAKE_CXX_FLAGS "-W -Wall -Wno-unused-variable  -Wno-unused-parameter")
endif(APPLE)

if(UNIX AND NOT APPLE)
  set( CMAKE_CXX_FLAGS "-W -Wall -Wno-unused-variable  -Wno-unused-parameter -lpthread -pthread")
endif(UNIX AND NOT APPLE)

option(USE_SOLUTION "defines if the solution source should be compiled or the skeleton code should be used" OFF)

if(${USE_SOLUTION})
  set(A1_DIR ${CMAKE_CURRENT_SOURCE_DIR}/solution)
else()
  set(A1_DIR ${CMAKE_CURRENT_SOURCE_DIR}/skeleton)
endif()

set(A1_FOUND True PARENT_SCOPE)
set(A1_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/inc)

add_library(a1lib SHARED ${A1_DIR}/a1.cpp)
target_include_directories(a1lib PUBLIC ${A1_INCLUDE_DIR})
set_property(TARGET a1lib PROPERTY CXX_STANDARD 14)


add_executable(a1 ${A1_DIR}/main.cpp)
target_include_directories(a1 PUBLIC ${A1_INCLUDE_DIR})
set_property(TARGET a1 PROPERTY CXX_STANDARD 14)
target_link_libraries(a1 PUBLIC a1lib)
