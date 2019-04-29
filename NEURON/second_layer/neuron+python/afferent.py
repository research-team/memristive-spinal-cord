from neuron import h
h.load_file('stdlib.hoc') #for h.lambda_f

class afferent(object):
  def __init__(self):
    #print 'construct ', self
    self.topol()
    self.subsets()
    self.geom()
    self.biophys()
    #self.geom_nseg()
    self.synlistees = []
    self.synlistex = []
    self.synlistinh = []
    self.synapses()
    self.x = self.y = self.z = 0.

  def __del__(self):
    #print 'delete ', self
    pass

  def topol(self):
    self.soma = h.Section(name='soma', cell=self)
    self.dend = h.Section(name='dend', cell= self)
    self.axon = h.Section(name='axon', cell= self)
    self.dend.connect(self.soma(1))
    self.axon.connect(self.soma(1))
    #self.basic_shape()  

  def basic_shape(self):
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
    self.all = h.SectionList()
    self.all.append(sec=self.soma)
    self.all.append(sec=self.dend)
    self.all.append(sec=self.axon)

  def geom(self):
    self.soma.L = self.soma.diam = 10
    self.dend.L = 200
    self.dend.diam = 1
    self.axon.L = 150
    self.axon.diam = 1

  def geom_nseg(self):
    for sec in self.all:
      sec.nseg = int((sec.L/(0.1*h.lambda_f(100)) + .9)/2.)*2 + 1

  def biophys(self):
    for sec in self.all:
      sec.Ra = 100
      sec.cm = 1
    self.soma.insert('hh')
    self.soma.gnabar_hh = 0.3
    self.soma.gkbar_hh = 0.04
    self.soma.gl_hh = 0.00017
    self.soma.el_hh = -60
    

    self.dend.insert('pas')
    self.dend.g_pas = 0.001
    self.dend.e_pas = -65

    self.axon.insert('hh')

  def position(self, x, y, z):
    soma.push()
    for i in range(h.n3d()):
      h.pt3dchange(i, x-self.x+h.x3d(i), y-self.y+h.y3d(i), z-self.z+h.z3d(i), h.diam3d(i))
    self.x = x; self.y = y; self.z = z
    h.pop_section()

  def connect2target(self, target):
    nc = h.NetCon(self.axon(1)._ref_v, target, sec = self.axon)
    nc.threshold = 10
    return nc

  def synapses(self):
    for i in range(200): 
      s = h.ExpSyn(self.dend(0.5)) # E0
      s.tau = 0.1
      s.e = 50
      self.synlistex.append(s)
      s = h.ExpSyn(self.dend(0.5)) # E1
      s.tau = 0.1
      s.e = 50
      self.synlistees.append(s)
      s = h.Exp2Syn(self.dend(0.5)) # I1
      s.tau1 = 1.5
      s.tau2 = 2
      s.e = -80
      self.synlistinh.append(s)  

  def is_art(self):
    return 0
