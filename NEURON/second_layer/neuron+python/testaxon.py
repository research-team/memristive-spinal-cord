from neuron import h, gui
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyplot
import math
#neuron.load_mechanisms("./mod")
from bioaffrat import bioaffrat
from interneuron import interneuron
from motoneuron import motoneuron
from muscle import muscle

import random

def set_recording_vectors(compartment):
    ''' recording voltage
    Parameters
    ----------
    compartment: NEURON section
        compartment for recording
    Returns
    -------
    v_vec: h.Vector()
        recorded voltage
    t_vec: h.Vector()
        recorded time
    '''
    v_vec = h.Vector()   # Membrane potential vector at compartment
    v_vec2 = h.Vector()
    v_vec1 = h.Vector()
    t_vec = h.Vector()        # Time stamp vector
    v_vec.record(compartment(0.5)._ref_vext[0])
    v_vec1.record(compartment(0.5)._ref_ik)
    v_vec2.record(compartment(0.5)._ref_ina)

    # v_vec2.record(compartment(0.5)._ref_h_fastchannels)
    # v_vec3.record(compartment(0.5)._ref_n_fastchannels)
    t_vec.record(h._ref_t)
    return v_vec, v_vec1, v_vec2, t_vec

def simulate(cell, tstop=200, vinit=-70):
    ''' simulation control
    Parameters
    ----------
    cell: NEURON cell
        cell for simulation
    tstop: int (ms)
        simulation time
    vinit: int (mV)
        initialized voltage
    '''
    h.tstop = tstop
    h.v_init = vinit
    h.run()

def show_output(v_vec, v_vec1, v_vec2, t_vec):
    ''' show graphs
    Parameters
    ----------
    v_vec: h.Vector()
        recorded voltage
    t_vec: h.Vector()
        recorded time
    '''
    # t_vec = list(t_vec)[100:]
    # v_vec = list(v_vec)[100:]
    # dend_plot =
    pyplot.plot(t_vec, v_vec, label = 'ina')
    # pyplot.plot(t_vec, v_vec2, label = 'ica')
    # pyplot.plot(t_vec, v_vec1, label = 'ik')

    pyplot.legend()
    # f = open('./res.txt', 'w')
    # for v in list(v_vec):
    #     f.write(str(v)+"\n")
    pyplot.xlabel('time (ms)')
    pyplot.ylabel('mV')

if __name__ == '__main__':
    muscle = muscle()


    cell = motoneuron(60)
    aff = bioaffrat()
    # branch_vec, t_vec = set_recording_vectors(cell.axonL.node[len(cell.axonL.node)-1])
    stim = h.NetStim()
    stim.number = 10
    stim.start = 30
    stim.interval = 4
    ncstim = h.NetCon(stim, cell.synlistex[0])
    ncstim.delay = 1
    ncstim.weight[0] = 0.75
    stim3 = h.NetStim()
    stim3.number = 3
    stim3.start = 80
    stim3.interval = 5
    ncstim3 = h.NetCon(stim3, cell.synlistex[2])
    ncstim3.delay = 1
    ncstim3.weight[0] = 0.75
    # # dummy = h.Section(name='dummy')
    stim2 = h.IaGenerator(0.5)
    stim2.number = 1000000
    h.setpointer(muscle.muscle_unit(0.5)._ref_F_fHill, 'fhill', stim2)
    v_vec, v_vec1, v_vec2, t_vec = set_recording_vectors(muscle.soma)
    # soma_vec2, t_vec = set_recording_vectors(cell.soma)
    ncstim2 = h.NetCon(stim2, aff.synlistex[1])
    ncstim2.delay = 1
    ncstim2.weight[0] = 0.05

    nc1 = h.NetCon(aff.node[len(aff.node)-1](0.5)._ref_v, cell.synlistex[5], sec=aff.node[len(aff.node)-1])
    nc1.delay = 1
    nc1.weight[0] = 0.5
    nclist = []
    for i in range(50):
        nc2 = h.NetCon(stim, muscle.synlistex[i])
        nc2.delay = random.gauss(1, 0.05)
        nc2.weight[0] = random.gauss(0.5, 0.03)
        nclist.append(nc2)

    for i in range(50,80):
        nc2 = h.NetCon(stim3, muscle.synlistex[i])
        nc2.delay = random.gauss(1, 0.05)
        nc2.weight[0] = random.gauss(0.5, 0.05)
        nclist.append(nc2)

    # stim = h.IClamp(cell.axonL.node[len(cell.axonL.node)-1](0.5))
    # stim.delay = 150
    # stim.dur = 5
    # stim.amp = 0.1
    # print("Number of model - ",cell.numofmodel)
    for sec in h.allsec():
        h.psection(sec=sec)
    simulate(cell)
    # show_output(branch_vec, t_vec)
    show_output(v_vec, v_vec1, v_vec2, t_vec)
    # show_output(soma_vec2, t_vec)
    pyplot.show()
