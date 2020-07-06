from neuron import h
import random
h.load_file('stdlib.hoc') #for h.lambda_f

import random

class muscle(object):
  '''
  muscle class with parameters:
    ...
  '''
  def __init__(self):
    self.topol()
    self.subsets()
    self.geom()
    self.geom_nseg()
    self.biophys()
    self.synlistex = []
    self.synlistinh = []
    self.synapses()

    def __del__(self):
    #print 'delete ', self
      pass

  def topol(self):
    '''
    Creates section
    '''
    self.muscle_unit = h.Section(name='muscle_unit', cell=self)
    self.soma = h.Section(name='soma', cell=self)
    self.muscle_unit.connect(self.soma(1))

  def subsets(self):
    '''
    NEURON staff
    adds sections in NEURON SectionList
    '''
    self.all = h.SectionList()
    for sec in h.allsec():
      self.all.append(sec=sec)

  def geom(self):
    '''
    Adds length and diameter to sections
    '''
    self.muscle_unit.L = 3000 # microns
    self.muscle_unit.diam = 40 # microns
    self.soma.L = 3000 # microns
    self.soma.diam = 40 # microns

  def geom_nseg(self):
    '''
    Calculates numder of segments in section
    '''
    for sec in self.all:
      sec.nseg = 3#int((sec.L/(0.1*h.lambda_f(100)) + .9)/2.)*2 + 1

  def biophys(self):
    '''
    Adds channels and their parameters
    '''
    self.muscle_unit.cm = 3.6  # cm uf/cm2
    self.muscle_unit.insert('Ca_conc')
    self.muscle_unit.insert('pas')
    self.muscle_unit.g_pas = 0.004
    self.muscle_unit.e_pas = -70
    self.muscle_unit.Ra = 1.1

    self.soma.cm = 3.6 # cm uf/cm2
    self.soma.Ra = 1.1
    self.soma.insert('Ca_conc')
    self.soma.insert('fastchannels')
    self.soma.insert('kir')
    self.soma.insert('na14a')
    self.soma.insert('cal')
    self.soma.insert('K_No')
    self.soma.insert('cac1')
    # self.soma.insert('pas')
    # self.soma.g_pas = 0.0002
    self.soma.gmax_cac1 = 0.005
    self.soma.gbar_na14a = 0.75
    self.soma.gkbar_kir = 0.01
    self.soma.gnabar_fastchannels=0.55
    self.soma.gkbar_fastchannels=0.01
    self.soma.gl_fastchannels=0.01
    self.soma.el_fastchannels=-70
    # self.soma.gnabar_hh = 0.35
    # self.soma.gkbar_hh = 0.02
    # self.soma.gl_hh = 0.002
    self.soma.gkmax_K_No = 0.01
    self.soma.gcalbar_cal = 0.1

    self.soma.ena = 55
    self.soma.ek = -80

    # self.soma.gcaN_motoneuron = 0.0#001
    # self.soma.gnabar_motoneuron = 0.2
    # self.soma.gcaL_motoneuron = 0.0003
    # self.soma.gl_motoneuron = 0.005
    # self.soma.gkrect_motoneuron = 0.05
    # self.soma.gcak_motoneuron =  0.01

    self.soma.insert('extracellular')  #adds extracellular mechanism for recording extracellular potential

    rec = h.xm(self.muscle_unit(0.5))

    self.muscle_unit.insert('CaSP')
    self.muscle_unit.insert('fHill')

  def connect2target(self, target):
    '''
    NEURON staff
    Adds presynapses
    Parameters
    ----------
    target: NEURON cell
        target neuron
    Returns
    -------
    nc: NEURON NetCon
        connection between neurons
    '''
    nc = h.NetCon(self.muscle_unit(0.5)._ref_v, target, sec = self.muscle_unit)
    nc.threshold = 10
    return nc

  def synapses(self):
    '''
    Adds synapses
    '''
    for i in range(100):
        s = h.ExpSyn(self.soma(0.5)) # Inhibitory
        s.tau = 0.3
        # s.tau2 = 0.5
        s.e = 0
        self.synlistex.append(s)
        # s = h.ExpSyn(self.muscle_unit(0.5)) # Inhibitory
        # s.tau = 0.2
        # # s.tau2 = 0.5
        # s.e = 30
        # self.synlistinh.append(s)


  def is_art(self):
    return 0
