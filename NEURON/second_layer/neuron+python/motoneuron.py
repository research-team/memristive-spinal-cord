from neuron import h
h.load_file('stdlib.hoc') #for h.lambda_f

class motoneuron(object):
  '''
  Motoneuron class with parameters:
    soma: NEURON Section (creates by topol())
    dend: NEURON Section (creates by topol())
    axon: NEURON Section (creates by topol())
    synlistinh: list (creates by synapses())
      list of inhibitory synapses
    synlistex: list (creates by synapses())
      list of excitatory synapses
    x, y, z: int
      3D coordinates 
  '''
  def __init__(self):
    #print 'construct ', self
    self.topol()
    self.subsets()
    self.geom()
    self.biophys()
    self.geom_nseg()
    self.synlistinh = []
    self.synlistex = []
    self.synapses()
    self.x = self.y = self.z = 0.

  def __del__(self):
    #print 'delete ', self
    pass

  def topol(self):
    '''
    Creates sections soma, dend, axon and connects them 
    '''
    self.soma = h.Section(name='soma', cell=self)
    self.dend = h.Section(name='dend', cell= self)
    self.axon = h.Section(name='axon', cell= self)
    self.dend.connect(self.soma(1))
    self.axon.connect(self.soma(1))
    #self.basic_shape()  

  def basic_shape(self):
    '''
    Adds 3D coordinates
    '''
    self.soma.push()
    h.pt3dclear()
    h.pt3dadd(0, 0, 0, 1)
    h.pt3dadd(15, 0, 0, 1)
    h.pop_section()
    self.dend.push()
    h.pt3dclear()
    h.pt3dadd(15, 0, 0, 1)
    h.pt3dadd(215, 0, 0, 1)
    h.pop_section()
    self.axon.push()
    h.pt3dclear()
    h.pt3dadd(0, 0, 0, 1)
    h.pt3dadd(-150, 0, 0, 1)
    h.pop_section()

  def subsets(self):
    '''
    NEURON staff
    adds sections in NEURON SectionList
    '''
    self.all = h.SectionList()
    self.all.append(sec=self.soma)
    self.all.append(sec=self.dend)
    self.all.append(sec=self.axon)

  def geom(self):
    '''
    Adds length and diameter to sections
    '''
    self.soma.L = self.soma.diam = 10 # microns
    self.dend.L = 200 # microns
    self.dend.diam = 1 # microns
    self.axon.L = 150 # microns
    self.axon.diam = 1 # microns

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
    for sec in self.all:
      sec.Ra = 100 # Ra ohm cm - membrane resistance
      sec.cm = 1 # cm uf/cm2 - membrane capacitance
    self.soma.insert('motoneuron_5ht')
    self.soma.insert('extracellular') #adds extracellular mechanism for recording extracellular potential

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
    nc = h.NetCon(self.soma(1)._ref_v, target, sec = self.soma)
    nc.threshold = 10
    return nc

  def synapses(self):
    '''
    Adds synapses 
    '''
    for i in range(200): 
      s = h.ExpSyn(self.soma(0.8)) # Excitatory
      s.tau = 0.1
      s.e = 50
      self.synlistex.append(s)
      s = h.Exp2Syn(self.soma(0.5)) # Inhibitory
      s.tau1 = 1.5
      s.tau2 = 2
      s.e = -80
      self.synlistinh.append(s)    

  def is_art(self):
    return 0
