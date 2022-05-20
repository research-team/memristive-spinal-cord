#!/bin/bash -l
#SBATCH --job-name=Neuron_calculation    
#SBATCH --output=Neuron_calculation.slurmout
#SBATCH --error=Neuron_calculation.slurmerr
#SBATCH --nodes=1
#SBATCH --ntasks=48

srun --mpi=pmix ./neuron.sif nrniv -mpi -python cpg_rat.py