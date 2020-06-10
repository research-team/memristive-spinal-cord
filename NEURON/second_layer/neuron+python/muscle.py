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
    self.biophys()
    self.synlistex = []
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
    self.muscle_unit.L = 30 # microns
    self.muscle_unit.diam = 30 # microns
    self.soma.L = 30 # microns
    self.soma.diam = 30 # microns

  def geom_nseg(self):
    '''
    Calculates numder of segments in section
    '''
    for sec in self.all:
      sec.nseg = int((sec.L/(0.1*h.lambda_f(100)) + .9)/2.)*2 + 1

  def biophys(self):
    '''
    Adds channels and their parameters
    '''
    self.muscle_unit.cm = 20 # cm uf/cm2
    self.muscle_unit.insert('pas')
    self.muscle_unit.g_pas = 0.002

    self.soma.cm = random.uniform(3, 4) # cm uf/cm2
    self.soma.Ra = random.uniform(30, 60)
    self.soma.insert('Ca_conc')
    self.soma.insert('na14a')
    self.soma.insert('motoneuron')
    self.soma.gbar_na14a = 0.3
    self.soma.gnabar_motoneuron = 0.3
    self.soma.gcaL_motoneuron = 0.002
    self.soma.gl_motoneuron = 0.002
    self.soma.gkrect_motoneuron = 0.1
    self.soma.gcak_motoneuron =  0.1
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
    nc = h.NetCon(self.soma(0)._ref_v, target, sec = self.soma)
    nc.threshold = 10
    return nc

  def synapses(self):
    '''
    Adds synapses
    '''
    for i in range(200):
      s = h.ExpSyn(self.soma(0.5)) # Excitatory
      s.tau = 0.1
      s.e = 0
      self.synlistex.append(s)


  def is_art(self):
    return 0
