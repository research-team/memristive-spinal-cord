from neuron import h
import random
h.load_file('stdlib.hoc') #for h.lambda_f

class interneuron(object):
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
  def __init__(self, delay):
    self.delay = delay
    self.diffs = [ś
    self.recs = []
    self.topol()
    self.subsets()
    self.geom()
    self.geom_nseg()
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
    self.axon = h.Section(name='axon', cell= self)
    if self.delay:
      self.dend = [h.Section(name='dend[%d]' % i) for i in range(15)]
      self.dend[0].connect(self.soma(1))
      for i in range(1, len(self.dend)):
        self.dend[i].connect(self.dend[i-1])
    else:
      self.dend = h.Section(name='dend', cell= self)
      self.dend.connect(self.soma(1))
    self.axon.connect(self.soma(1))

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
    self.axon.L = 150 # microns
    self.axon.diam = 1 # microns
    if self.delay:
      for sec in self.dend:
        sec.L = 200 # microns
        sec.diam = 1 # microns
    else:
      self.dend.L = 200 # microns
      self.dend.diam = 1 # microns

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
    for sec in self.all:
      sec.Ra = 100 # Ra ohm cm - membrane resistance
      sec.cm = 1 # cm uf/cm2 - membrane capacitance
    self.soma.insert('hh')
    self.soma.gnabar_hh = 0.3
    self.soma.gkbar_hh = 0.04
    self.soma.gl_hh = 0.00017
    self.soma.el_hh = -70 
    self.soma.insert('extracellular') #adds extracellular mechanism for recording extracellular potential

    if self.delay:
      distance = random.uniform(10, 100)
      for sec in self.dend:
        sec.insert('hh')
        diff = h.slow_5HT(sec(0.5))
        rec = h.r5ht3a(sec(0.5))
        rec.gmax = 2
        diff.h = random.gauss(distance, distance/5)
        diff.tx1 = 1+(diff.h/1250)*1000
        h.setpointer(diff._ref_serotonin, 'serotonin', rec)
        self.diffs.append(diff)
        self.recs.append(rec)  
    else:
      self.dend.insert('pas')
      self.dend.g_pas = 0.001
      self.dend.e_pas = -65

    self.axon.insert('hh')

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
    nc = h.NetCon(self.axon(1)._ref_v, target, sec = self.axon)
    nc.threshold = 10
    return nc

  def synapses(self):
    '''
    Adds synapses 
    '''
    if self.delay:
      for sec in self.dend:
        for i in range(50): 
          s = h.ExpSyn(sec(0.5)) # Excitatory
          s.tau = 0.1
          s.e = 50
          self.synlistex.append(s)
          s = h.Exp2Syn(sec(0.5)) # Inhibitory
          s.tau1 = 1.5
          s.tau2 = 2
          s.e = -80
          self.synlistinh.append(s)  
          s = h.ExpSyn(sec(0.8)) # Excitatory
          s.tau = 0.1
          s.e = 50
          self.synlistees.append(s)  
    else:
      for i in range(200): 
        s = h.ExpSyn(self.dend(0.5)) # Excitatory
        s.tau = 0.1
        s.e = 50
        self.synlistex.append(s)
        s = h.Exp2Syn(self.dend(0.5)) # Inhibitory
        s.tau1 = 1.5
        s.tau2 = 2
        s.e = -80
        self.synlistinh.append(s)  
        s = h.ExpSyn(self.dend(0.8)) # Excitatory
        s.tau = 0.1
        s.e = 50
        self.synlistees.append(s)  

  def is_art(self):
    return 0