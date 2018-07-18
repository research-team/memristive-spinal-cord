#!/bin/bash
# -b, --spikebuf ARG       Spike buffer size. (100000)
# -c, --threading          Parallel threads. The default is serial threads.
# -d, --datpath ARG        Path containing CoreNeuron data files. (.)
# -dt, --dt ARG            Fixed time step. The default value is set by defaults.dat or is 0.025.
# -e, --tstop ARG          Stop time (ms). (100)
# -f, --filesdat ARG       Name for the distribution file. (files.dat)
# -g, --prcellgid ARG      Output prcellstate information for the gid NUMBER.
# -gpu, --gpu              Enable use of GPUs. The default implies cpu only run.
# -i, --dt_io ARG          Dt of I/O. (0.1)
# -k, --forwardskip ARG    Forwardskip to TIME
# -l, --celsius ARG        Temperature in degC. The default value is set in defaults.dat or else is 34.0.
# -mpi                     Enable MPI. In order to initialize MPI environment this argument must be specified.
# -o, --outpath ARG        Path to place output data files. (.)
# -p, --pattern ARG        Apply patternstim using the specified spike file.
# -R, --cell-permute ARG   Cell permutation, 0 No; 1 optimise node adjacency; 2 optimize parent adjacency. (1)
# -r, --report ARG         Enable voltage report (0 for disable, 1 for soma, 2 for full compartment).
# -s, --tstart ARG         Start time (ms). (0)
# -v, --voltage ARG        Initial voltage used for nrn_finitialize(1, v_init). If 1000, then nrn_finitialize(0,...). (-65.)
# -W, --nwarp ARG          Number of warps to balance. (0)
# -w, --dt_report ARG      Dt for soma reports (using ReportingLib). (0.1)
# -x, --extracon ARG       Number of extra random connections in each thread to other duplicate models (int).
# -z, --multiple ARG       Model duplication factor. Model size is normal size * (int).
# --binqueue               Use bin queue.
# --mindelay ARG           Maximum integration interval (likely reduced by minimum NetCon delay). (10)
# --ms-phases ARG          Number of multisend phases, 1 or 2. (2)
# --ms-subintervals ARG    Number of multisend subintervals, 1 or 2. (2)
# --multisend              Use Multisend spike exchange instead of Allgather.
# --read-config ARG        Read configuration file filename.
# --show                   Print args.
# --spkcompress ARG        Spike compression. Up to ARG are exchanged during MPI_Allgather. (0)
# --write-config ARG       Write configuration file filename.

# Test file: CoreNeuron/source_CoreNeuron/tests/integration/ring

WD="/gpfs/GLOBAL_JOB_REPO_KPFU/openlab/"

# Check if no arguments
if [ $# -eq 0 ]
  then
    echo "Set the main simulation file!"
    exit 1
fi

read -p "[1] Job name      : " job
read -p "[2] Stop time (ms): " stop_time
echo "- - - - - - - - - - - - - - -"

# Create work directory
today=$(date "+%Y-%m-%d_%H:%M")
result_dir="${WD}CoreNeuron/results/${today}_${job}"
mkdir ${result_dir}

${WD}CoreNeuron/build/bin/coreneuron_exec -d ${WD}$1 -e ${stop_time} --gpu -o ${result_dir}

echo "- - - - - - - - - - - - - - -"
echo
