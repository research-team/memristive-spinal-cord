#!/bin/bash

read -p "[1] Job name      : " job

echo "- - - - - - - - - - - - - - -"

#source /usr/mpi/gcc/openmpi-1.10.2-lsf/bin/mpivars.sh

# log files
err="$HOME/GRAS/log/$1/%J($job).err"
out="$HOME/GRAS/log/$1/%J($job).out"

# set by default 1 thread
export OMP_NUM_THREADS=1

# OpenMPI executable
mpirun="/usr/mpi/gcc/openmpi-1.10.2-lsf/bin/mpirun"

# machine file
machinefile="bmk-x2-a1-ch1-10"

# .cu file path
cu_file="$HOME/GRAS/two_muscle_simulaton.cpp"

# build file path
build="$HOME/GRAS/build"

# build the .cu file
if nvcc -std=c++11 -o build two_muscle_simulation.cu
then
        echo "successfully built"
        echo "- - - - - - - - - - - - - - -"
        # run the main command
        bsub -J ${job} -e ${err} -o ${out} -m "${machinefile}" -n 1 ${mpirun} -np 1 ${build} 21 40 100 2 0 0 0
else
        echo "ERROR in build"
fi