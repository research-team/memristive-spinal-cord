from neuron import h, gui
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyplot
import math
#neuron.load_mechanisms("./mod")
from bioaff import bioaff
from interneuron import interneuron
from motoneuron import motoneuron

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
    t_vec = h.Vector()        # Time stamp vector
    v_vec.record(compartment(0.5)._ref_v)
    t_vec.record(h._ref_t)
    return v_vec, t_vec

def simulate(cell, tstop=600, vinit=-70):
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

def show_output(v_vec, t_vec):
    ''' show graphs
    Parameters
    ----------
    v_vec: h.Vector()
        recorded voltage
    t_vec: h.Vector()
        recorded time
    '''
    t_vec = list(t_vec)[100:]
    v_vec = list(v_vec)[100:]
    dend_plot = pyplot.plot(t_vec, v_vec)
    # f = open('./res.txt', 'w')
    # for v in list(v_vec):
    #     f.write(str(v)+"\n")
    pyplot.xlabel('time (ms)')
    pyplot.ylabel('mV')

if __name__ == '__main__':
    cell = motoneuron(40)
    # branch_vec, t_vec = set_recording_vectors(cell.axonL.node[len(cell.axonL.node)-1])
    soma_vec, t_vec = set_recording_vectors(cell.muscle.muscle_unit)
    stim = h.NetStim()
    stim.number = 10
    stim.start = 10
    ncstim = h.NetCon(stim, cell.synlistex[0])
    ncstim.delay = 1
    ncstim.weight[0] = 0.5
    stim2 = h.NetStim()
    stim2.number = 10
    stim2.start = 300
    ncstim2 = h.NetCon(stim2, cell.synlistex[1])
    ncstim2.delay = 1
    ncstim2.weight[0] = 0.5
    # stim = h.IClamp(cell.axonL.node[len(cell.axonL.node)-1](0.5))
    # stim.delay = 150
    # stim.dur = 5
    # stim.amp = 0.1
    # print("Number of model - ",cell.numofmodel)
    for sec in h.allsec():
        h.psection(sec=sec)
    simulate(cell)
    # show_output(branch_vec, t_vec)
    show_output(soma_vec, t_vec)
    pyplot.show()
