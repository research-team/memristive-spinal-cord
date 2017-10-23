MUSCLE SPINDLE FEEDBACK CIRCUIT MODEL by 

Marco Capogrosso and Emanuele Formento

Published in:
Mechanisms Underlying the Neuromodulation of Spinal Circuits for Correcting Gait and Balance Deficits after Spinal Cord Injury.
Moraud EM, Capogrosso M, Formento E, Wenger N, DiGiovanna J, Courtine G, Micera S.
Neuron. 2016 Feb 17;89(4):814-28. doi: 10.1016/j.neuron.2016.01.009. Epub 2016 Feb 4.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

To run this program it is required to have Neuron with python installed. 
For faster simulations it is also suggested to install Neuron with the parallel option.

Once Neuron is properly installed compile the mod files in ./hoc_files/mod_files
	by executing in the terminal:
		cd ./hoc_files/
		nrnivmodl ./mod_files

Then you can run the program
	by executing in the terminal:
		cd ./hoc_files
		python ../python_files/main.py # to run the program with a single process
		mpiexec -np 8 python ../python_files/main.py # tu run the program in with mpi (8 is the number of processes)
		mpiexec -np 8 nrniv -mpi -python ../python_files/main.py # as before but running the program from nrniv

