from neuron import h, gui
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyplot
import math
#neuron.load_mechanisms("./mod")
from bioaff import bioaff

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

def simulate(cell, tstop=300, vinit=-70):
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
    dend_plot = pyplot.plot(t_vec, v_vec)
    # f = open('./res.txt', 'w')
    # for v in list(v_vec):
    #     f.write(str(v)+"\n")
    pyplot.xlabel('time (ms)')
    pyplot.ylabel('mV')

if __name__ == '__main__':
    cell = bioaff(100)
    branch_vec, t_vec = set_recording_vectors(cell.node[len(cell.node)-1])
    soma_vec, t_vec = set_recording_vectors(cell.soma)
    stim = h.IClamp(cell.soma(1))
    stim.delay = 150
    stim.dur = 5
    stim.amp = 0.5
    # print("Number of model - ",cell.numofmodel)
    for sec in h.allsec():
        h.psection(sec=sec)
    simulate(cell)
    show_output(branch_vec, t_vec)
    show_output(soma_vec, t_vec)
    pyplot.show()