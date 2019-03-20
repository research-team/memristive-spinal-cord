#!/bin/bash

nvcc -o output cuda_sim.cu
./output
rm ./output

python3 ../plot_results.py ./
rm ./spikes.dat ./curr.dat ./volt.dat