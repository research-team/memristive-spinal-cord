# MUSCLE SPINDLE FEEDBACK CIRCUIT MODEL by

# Marco Capogrosso and Emanuele Formento

# Published in:
# Mechanisms Underlying the Neuromodulation of Spinal Circuits for Correcting Gait and Balance Deficits after Spinal Cord Injury.
# Moraud EM, Capogrosso M, Formento E, Wenger N, DiGiovanna J, Courtine G, Micera S.
# Neuron. 2016 Feb 17;89(4):814-28. doi: 10.1016/j.neuron.2016.01.009. Epub 2016 Feb 4.


import sys
import logging

sys.path.append('../python_files')
from NeuralNetwork import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
sizeComm = comm.Get_size()
rank = comm.Get_rank()


def main():

    logger = logging.getLogger('neuron')
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler('simulation.log')

    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    simulationType = None
    if rank == 0:
        simulationType = input(
            "\nWhich Experiment would you like to perform? \nEnter 1 to compute recruitment curves \nEnter 2 to perform a dynamic stepping simulation\n ")
    comm.Barrier()
    simulationType = comm.bcast(simulationType, root=0)
    if simulationType != 1 and simulationType != 2:
        if rank == 0: print "Invalid input!"
        sys.exit(-1)

    # Setting the simulation parameters
    if simulationType == 1:
        network = None
        if rank == 0:
            ans = raw_input("Do you want to simulate the Extensor or Flexors recruitment curve (e/f)?\n")
            if ans == "e":
                network = "extensor"
                print "Extensor network set"
            elif ans == "f":
                network = "flexor"
                print "Flexor network set"
            else:
                print "Invalid input, Extensor network set"
                network = "extensor"
            print "Starting the recrutiment curve simulation..."
        comm.Barrier()
        network = comm.bcast(network, root=0)

    elif simulationType == 2:
        amplitude = None
        frequency = None
        if rank == 0:
            ans = raw_input(
                "Do you want to modify the predefined parameters of stimulation (40Hz EES and optimal amplitude) (y/n)?\n")
            if ans == "y":
                frequency = input("Please insert the frequency of stimulation (0-200):\t")
                if frequency >= 0 and frequency <= 200:
                    print "Frequency of stimulation set to: " + str(frequency) + "Hz\n"
                else:
                    print "Invalid frequency value - EES frequency set to 40 Hz\n"
                    frequency = 40
                amplitude = input(
                    "Please insert the amplitude of stimulation\nInsert -1 to chose the 'optimal' amplitude of stimulation (amplitude that leads to the largest recruitment of afferent fibers without recruiting efferent fibers)\nOr insert a current from 0 to 600 uA:\n\t")
                if amplitude == -1:
                    amplitude = "optimal"
                    print "Amplitude of stimulation set to: " + str(amplitude) + "\n"
                elif amplitude >= 0 and amplitude <= 600:
                    print "Amplitude of stimulation set to: " + str(amplitude) + " uA\n"
                else:
                    print "Invalid amplitude value - amplitude set to optimal\n"
                    amplitude = "optimal"
            else:
                amplitude = "optimal"
                frequency = 40
            print "Starting dynamic simulation..."
        comm.Barrier()
        amplitude = comm.bcast(amplitude, root=0)
        frequency = comm.bcast(frequency, root=0)

    # Starting simulations
    logging.info('Simulation started')
    sim = NeuralNetwork()
    if simulationType == 1:
        sim.computeRecruitCurve(network)
    elif simulationType == 2:
        sim.runSimulation(frequency, "", amplitude)

    del sim
    logging.info('Done!')


if __name__ == '__main__':
    main()
