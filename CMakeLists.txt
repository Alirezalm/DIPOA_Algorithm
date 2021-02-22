cmake_minimum_required(VERSION 3.17)
project(DIPOA_Algorithm)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS " -O3")
set(CMAKE_CXX_FLAGS " -DIL_STD -I/home/alireza/gurobi903/linux64/include -I/home/alireza/cplex/cplex/include -I/home/alireza/cplex/concert/include
 -L/home/alireza/cplex/cplex/lib/x86-64_linux/static_pic -L/home/alireza/cplex/concert/lib/x86-64_linux/static_pic -lilocplex -lconcert -lcplex -lm -lpthread ")

add_executable(DIPOA main.cpp includes/external_libs.h includes/unc_solver.h
        src/unc_solver.cpp utils/results.cpp includes/results.h
        test_functions/log_reg_random.h includes/DCCP.h src/DCCP.cpp
        includes/rhadmm.h src/rhadmm.cpp includes/cut_generation.h src/cut_generation.cpp includes/master_milp.h src/master_milp.cpp)