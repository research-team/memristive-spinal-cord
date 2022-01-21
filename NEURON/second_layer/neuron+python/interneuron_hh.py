from neuron import h, gui
import random

import random

class interneuron_hh(object):
  '''
  Interneuron class with parameters:
    delay: bool
      Does it have 5ht receptors?
      -Yes: True
      -No: False
    soma: NEURON Section (creates by topol())
    dend: NEURON Section (creates by topol())
    axon: NEURON Section (creates by topol())
    synlistinh: list (creates by synapses())
      list of inhibitory synapses
    synlistex: list (creates by synapses())
      list of excitatory synapses
    synlistees: list (creates by synapses())
      list of excitatory synapses for connection with generators
    x, y, z: int
      3D coordinates (isn't used)
    diffs: list
      list of diffusion mechanisms (NEURON staff)
    recs: list
      list of receptors mechanisms (NEURON staff)
  '''
  def __init__(self, N):
    self.diffs = []
    self.recs = []
    self.nsyn = N
    self.topol()
    self.subsets()
    self.geom()
    # self.geom_nseg()
    self.biophys()
    self.synlistinh = []
    self.synlistex = []
    self.synlistees = []
    self.synapses()
    self.x = self.y = self.z = 0.

    def __del__(self):
    #print 'delete ', self
      pass

  def topol(self):
    '''
    Creates sections soma, dend, axon and connects them
    if it's delay creates section dend[]: array
    '''
    self.soma = h.Section(name='soma', cell=self)

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
    self.soma.L = self.soma.diam = 10 # microns

  def geom_nseg(self):
    '''
    Calculates numder of segments in section
    '''
    for sec in self.all:
      sec.nseg = int((sec.L/(0.1*h.lambda_f(100)) + .9)/2.)*2 + 1

  def biophys(self):
    '''
    Adds channels and their parameters
    if delay is true, adds 5ht receptors
    '''
    self.soma.cm = 1 # cm uf/cm2 - membrane capacitance
    self.soma.Ra = 100 # Ra ohm cm - membrane resistance
    self.soma.insert('hh')

  def position(self, x, y, z):
    '''
    NEURON staff
    Adds 3D position
    '''
    soma.push()
    for i in range(h.n3d()):
      h.pt3dchange(i, x-self.x+h.x3d(i), y-self.y+h.y3d(i), z-self.z+h.z3d(i), h.diam3d(i))
    self.x = x; self.y = y; self.z = z
    h.pop_section()

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
    nc = h.NetCon(self.soma(1)._ref_v, target, sec = self.soma)
    nc.threshold = -10
    return nc

  def synapses(self):
    '''
    Adds synapses
    '''
    for i in range(self.nsyn + 1):
      s = h.ExpSyn(self.soma(0.8)) # Excitatory
      s.tau = 0.5
      s.e = 15
      self.synlistex.append(s)
      s = h.Exp2Syn(self.soma(0.5)) # Inhibitory
      s.tau1 = 0.5
      s.tau2 = 3.5
      s.e = -80
      self.synlistinh.append(s)

  def is_art(self):
    return 0
