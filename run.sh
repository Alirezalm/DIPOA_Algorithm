#!/bin/bash

echo "script to compile and run DIPOA algorithm"
echo "written by Alireza"


echo "compiling by mpic++ ..."

P=$(mpic++ -O3  ./includes/*.h ./src/*.cpp ./utils/*.cpp test_functions/*.h ./main.cpp -o dipoa_mpi)

echo "compilation finished"

echo "$P"

mpirun ./dipoa_mpi

