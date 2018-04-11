Firstly, it is required to compile .mod files

under UNIX/Linux: *nrnivmodl* in project directory

Then run the program:

*mpiexec -np 8 nrniv -mpi parallelarc.hoc*  (8 is the number of processes)
