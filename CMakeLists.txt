cmake_minimum_required(VERSION 3.16)

project(dmft)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(dmft src/main.cpp src/utilis.cpp src/transport.cpp src/parameters.cpp src/interacting_gf.cpp src/dmft.cpp src/leads_self_energy.cpp src/aim.cpp src/analytic_gf.cpp src/sigma_2.cpp src/pseudo_gf.cpp src/nca.cpp)

target_include_directories(dmft PRIVATE ../include) #this will need to be changed to the relative path

target_compile_options(dmft PRIVATE ${MPI_CXX_COMPILE_FLAGS} -O3 -Wall -g)

target_link_libraries(dmft ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
