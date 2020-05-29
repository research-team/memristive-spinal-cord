from neuron import h
h.load_file('stdlib.hoc') #for h.lambda_f

import random
# from axon import axon
from muscle import muscle

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
  def __init__(self, diam):
    #print 'construct ', self
    #create axon
    self.diam = diam
    PI = 3.14
    #topological parameters
    self.number = 10
    self.axonnodes = self.number + 1
    self.paranodes1 = 2*self.number
    self.paranodes2 = 2*self.number
    self.axoninter = 6*self.number
    #morphological parameters
    self.fiberD = 10.0
    self.paralength1 = 3
    self.nodelength = 1.0
    space_p1 = 0.002
    space_p2 = 0.004
    space_i = 0.004
    self.g = 0.690
    self.axonD = 6.9
    self.nodeD = 3.3
    self.paraD1 = 3.3
    self.paraD2 = 6.9
    deltax = 1150
    self.paralength2 = 46
    self.nl = 120
    self.interlength=(deltax-self.nodelength-(2*self.paralength1)-(2*self.paralength2))/6
    #electrical parameters
    self.rhoa=0.7e6 #Ohm-um
    self.mycm=0.1 #uF/cm2/lamella membrane
    self.mygm=0.001 #S/cm2/lamella membrane
    self.Rpn0=(self.rhoa*.01)/(PI*((((self.nodeD/2)+space_p1)**2)-((self.nodeD/2)**2)))
    self.Rpn1=(self.rhoa*.01)/(PI*((((self.paraD1/2)+space_p1)**2)-((self.paraD1/2)**2)))
    self.Rpn2=(self.rhoa*.01)/(PI*((((self.paraD2/2)+space_p2)**2)-((self.paraD2/2)**2)))
    self.Rpx=(self.rhoa*.01)/(PI*((((self.axonD/2)+space_i)**2)-((self.axonD/2)**2)))
    self.topol()
    self.subsets()
    self.geom()
    self.biophys()
    #self.geom_nseg()
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
    self.node = [h.Section(name='node[%d]' % i, cell=self) for i in range(self.axonnodes)]
    self.MYSA = [h.Section(name='MYSA[%d]' % i, cell=self) for i in range(self.paranodes1)]
    self.FLUT = [h.Section(name='FLUT[%d]' % i, cell=self) for i in range(self.paranodes2)]
    self.STIN = [h.Section(name='STIN[%d]' % i, cell=self) for i in range(self.axoninter)]

    for i in range(self.number):
      self.MYSA[2*i].connect(self.node[i](1))
      self.FLUT[2*i].connect(self.MYSA[2*i](1))
      self.STIN[6*i].connect(self.FLUT[2*i](1))
      self.STIN[6*i+1].connect(self.STIN[6*i](1))
      self.STIN[6*i+2].connect(self.STIN[6*i+1](1))
      self.STIN[6*i+3].connect(self.STIN[6*i+2](1))
      self.STIN[6*i+4].connect(self.STIN[6*i+3](1))
      self.STIN[6*i+5].connect(self.STIN[6*i+4](1))
      self.FLUT[2*i+1].connect(self.STIN[6*i+5](1))
      self.MYSA[2*i+1].connect(self.FLUT[2*i+1](1))
      self.node[i+1].connect(self.MYSA[2*i+1](1))

    self.node[0].connect(self.soma(1))
    self.dend.connect(self.soma(0))
    # self.muscle.muscle_unit.connect(self.axon.node[self.axon.axonnodes-1](1))
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
    for sec in h.allsec():
      self.all.append(sec=sec)

  def geom(self):
    '''
    Adds length and diameter to sections
    '''
    self.soma.L = self.soma.diam = self.diam # microns
    self.dend.L = 200 # microns
    self.dend.diam = 1 # microns

    for sec in self.node:
      sec.L = self.nodelength   # microns
      sec.diam = self.nodeD  # microns
      sec.nseg = 1

    for sec in self.MYSA:
      sec.L = self.paralength1   # microns
      sec.diam = self.fiberD  # microns
      sec.nseg = 1

    for sec in self.FLUT:
      sec.L = self.paralength2   # microns
      sec.diam = self.fiberD  # microns
      sec.nseg = 1

    for sec in self.STIN:
      sec.L = self.interlength   # microns
      sec.diam = self.fiberD  # microns
      sec.nseg = 1

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
    self.soma.insert('motoneuron')
    self.soma.insert('extracellular') #adds extracellular mechanism for recording extracellular potential
    self.soma.Ra = 200 # Ra ohm cm - membrane resistance
    self.soma.cm = random.gauss(2, 0.1) # cm uf/cm2 - membrane capacitance
    if self.diam > 50:
      self.soma.gnabar_motoneuron = 0.4
      self.soma.gcaL_motoneuron = 0.003
      self.soma.gl_motoneuron = 0.005
      self.soma.gkrect_motoneuron = 0.1
    else:
      self.soma.gnabar_motoneuron = 0.055
      self.soma.gcaL_motoneuron = 0.0005
      self.soma.gkrect_motoneuron = 0.01
    '''
    self.soma.insert('pas')
    self.soma.g_pas = 0.002
    self.soma.e_pas = -80
    '''
    self.dend.insert('pas')
    self.dend.g_pas = 0.0002
    self.dend.e_pas = -80
    self.dend.Ra = 100 # Ra ohm cm - membrane resistance
    self.dend.cm = 1 # cm uf/cm2 - membrane capacitance

    '''
    self.axon.insert('hh')
    self.axon.gnabar_hh = 0.5
    self.axon.gkbar_hh = 0.1
    self.axon.gl_hh = 0.01
    self.axon.el_hh = -70
    self.axon.Ra = 70 # Ra ohm cm - membrane resistance
    self.axon.cm = 2 # cm uf/cm2 - membrane capacitance
    '''

    for sec in self.node:
      sec.Ra = self.rhoa/10000
      sec.cm = 2
      sec.insert('axnode')
      sec.insert('extracellular')
      sec.xraxial[1] = self.Rpn0
      sec.xg[1] = 1e10
      sec.xc[1] = 0

    for sec in self.MYSA:
      sec.Ra = self.rhoa*(1/((self.paraD1/self.fiberD)**2))/10000
      sec.cm = 2*self.paraD1/self.fiberD
      sec.insert('pas')
      sec.g_pas = 0.001*self.paraD1/self.fiberD
      sec.e_pas = -80
      sec.insert('extracellular')
      sec.xraxial[1] = self.Rpn1
      sec.xg[1] = self.mygm/(self.nl*2)
      sec.xc[1] = self.mycm/(self.nl*2)

    for sec in self.FLUT:
      sec.Ra = self.rhoa*(1/((self.paraD2/self.fiberD)**2))/10000
      sec.cm = 2*self.paraD2/self.fiberD
      sec.insert('pas')
      sec.g_pas = 0.0001*self.paraD2/self.fiberD
      sec.e_pas = -80
      sec.insert('extracellular')
      sec.xraxial[1] = self.Rpn2
      sec.xg[1] = self.mygm/(self.nl*2)
      sec.xc[1] = self.mycm/(self.nl*2)

    for sec in self.STIN:
      sec.Ra = self.rhoa*(1/((self.axonD/self.fiberD)**2))/10000
      sec.cm = 2*self.axonD/self.fiberD
      sec.insert('pas')
      sec.g_pas = 0.0001*self.axonD/self.fiberD
      sec.e_pas = -80
      sec.insert('extracellular')
      sec.xraxial[1] = self.Rpx
      sec.xg[1] = self.mygm/(self.nl*2)
      sec.xc[1] = self.mycm/(self.nl*2)

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
    nc = h.NetCon(self.node[len(self.node)-1](0.5)._ref_v, target, sec=self.node[len(self.node)-1])
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
      s.tau1 = 1
      s.tau2 = 1.5
      s.e = -80
      self.synlistinh.append(s)

  def is_art(self):
    return 0
