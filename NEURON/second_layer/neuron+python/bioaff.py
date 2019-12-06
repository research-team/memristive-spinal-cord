from neuron import h
h.load_file('stdlib.hoc') #for h.lambda_f

import random
from axon import axon

class bioaff(object):
  '''
  Afferent with bio-axon class with parameters:
    soma: NEURON Section (creates by topol())
    dend: NEURON Section (creates by topol())
    axon parameters from: https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=3810&file=/MRGaxon/MRGaxon.hoc#tabs-2
    synlistinh: list (creates by synapses())
      list of inhibitory synapses
    synlistex: list (creates by synapses())
      list of excitatory synapses
    synlistees: list (creates by synapses())
      list of excitatory synapses for connection with generators 
  '''
  def __init__(self):
    #create axon
    self.axonL = axon(random.randint(50, 100))
    self.axonR = axon(random.randint(5, 15))
    self.topol()
    self.subsets()
    self.geom()
    self.biophys()
    self.synlistees = []
    self.synlistex = []
    self.synlistinh = []
    self.synapses()

  def __del__(self):
    #print 'delete ', self
    pass

  def topol(self):
    '''
    Creates sections soma, dend, axon and connects them 
    '''
    self.soma = h.Section(name='soma', cell=self)
    self.axonR.node[0].connect(self.soma(1))
    self.axonL.node[0].connect(self.soma(1))

    #self.basic_shape()  

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
    self.soma.L = self.soma.diam = random.uniform(15, 35) # microns
    h.define_shape()

  def biophys(self):
    '''
    Adds channels and their parameters 
    '''
    self.soma.insert('hh')
    self.soma.gnabar_hh = 0.3
    self.soma.gkbar_hh = 0.04
    self.soma.gl_hh = 0.00017
    self.soma.el_hh = -70
    self.soma.Ra = 200
    self.soma.cm = 2
    self.soma.insert('extracellular')

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
    nc = h.NetCon(self.soma(0.5)._ref_v, target, sec = self.soma)
    nc.threshold = 10
    return nc

  def synapses(self):
    #for sec in self.axonL.node:
    for i in range(3):
      for j in range(50): 
        s = h.ExpSyn(self.axonL.node[len(self.axonL.node)-1-i](0.5)) # Excitatory
        s.tau = 0.1
        s.e = 50
        self.synlistex.append(s)  
        s = h.ExpSyn(self.axonR.node[i+1](0.5)) # Excitatory
        s.tau = 0.1
        s.e = 50
        self.synlistees.append(s)
    for i in range(200): 
      s = h.Exp2Syn(self.soma(0.5)) # Inhibitory
      s.tau1 = 1.5
      s.tau2 = 2
      s.e = -80
      self.synlistinh.append(s)
      
  def is_art(self):
    return 0
