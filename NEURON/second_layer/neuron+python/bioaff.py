from neuron import h
h.load_file('stdlib.hoc') #for h.lambda_f

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
  def __init__(self, number):
    PI = 3.14
    #topological parameters
    self.number = number  
    self.axonnodes = number + 1        
    self.paranodes1 = 2*number
    self.paranodes2 = 2*number
    self.axoninter = 6*number     
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
    self.dend = h.Section(name='dend', cell= self)
    self.node = [h.Section(name='node[%d]' % i, cell=self) for i in range(self.axonnodes)]
    self.MYSA = [h.Section(name='MYSA[%d]' % i, cell=self) for i in range(self.paranodes1)]
    self.FLUT = [h.Section(name='FLUT[%d]' % i, cell=self) for i in range(self.paranodes2)]
    self.STIN = [h.Section(name='STIN[%d]' % i, cell=self) for i in range(self.axoninter)]

    self.dend.connect(self.soma(1))
    self.node[0].connect(self.soma(1))

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
    self.soma.L = self.soma.diam = 10 # microns
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

    h.define_shape()

  def biophys(self):
    '''
    Adds channels and their parameters 
    '''
    self.soma.insert('hh')
    self.soma.gnabar_hh = 0.3
    self.soma.gkbar_hh = 0.04
    self.soma.gl_hh = 0.00017
    self.soma.el_hh = -60
    self.soma.Ra = 100
    self.soma.cm = 1

    self.dend.insert('pas')
    self.dend.g_pas = 0.001
    self.dend.e_pas = -65
    self.dend.Ra = 100
    self.dend.cm = 1

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
    nc = h.NetCon(self.node[self.axonnodes-1](1)._ref_v, target, sec = self.node[self.axonnodes-1])
    nc.threshold = 10
    return nc

  def synapses(self):
    for i in range(100): 
      s = h.ExpSyn(self.dend(0.5)) # Excitatory
      s.tau = 0.1
      s.e = 50
      self.synlistex.append(s)
      s = h.ExpSyn(self.dend(0.5)) # Excitatory
      s.tau = 0.1
      s.e = 50
      self.synlistees.append(s)
      s = h.Exp2Syn(self.dend(0.5)) # Inhibitory
      s.tau1 = 1.5
      s.tau2 = 2
      s.e = -80
      self.synlistinh.append(s)  

     

  def is_art(self):
    return 0
