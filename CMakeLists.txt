cmake_minimum_required(VERSION 3.22...3.30)
project(jacobi)

add_executable(jacobi)
target_sources(jacobi PRIVATE host.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(jacobi PUBLIC TT::Metalium)
