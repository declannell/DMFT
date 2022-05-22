cmake_minimum_required(VERSION 3.23)

project(dmft)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(dmft main.cpp transport.cpp parameters.cpp interacting_gf.cpp dmft.cpp leads_self_energy.cpp)

target_compile_options(dmft PRIVATE ${MPI_CXX_COMPILE_FLAGS} -O3 -Wall -g)

target_link_libraries(dmft ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
