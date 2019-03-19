from neuron import h
import random
h.load_file('stdlib.hoc') #for h.lambda_f

class interneuron(object):
  def __init__(self, delay):
    #print 'construct ', self
    self.delay = delay
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
    self.all = h.SectionList()
    for sec in h.allsec():
      self.all.append(sec=sec)    

  def geom(self):
    self.soma.L = self.soma.diam = 10
    self.axon.L = 150
    self.axon.diam = 1
    if self.delay:
      for sec in self.dend:
        sec.L = 200
        sec.diam = 1
    else:
      self.dend.L = 200
      self.dend.diam = 1

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
    self.soma.insert('extracellular')

    if self.delay:
      distance = random.uniform(10, 100)
      for sec in self.dend:
        sec.insert('hh')
        diff = h.diff_slow(sec(0.5))
        rec = h.r5ht3a(sec(0.5))
        rec.gmax = 2
        diff.h = random.gauss(distance, distance/5)
        diff.tx1 = 1+(diff.h/1250)*1000
        h.setpointer(diff._ref_subs, 'serotonin', rec)
    else:
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
    if self.delay:
      for sec in self.dend:
        for i in range(10): 
          s = h.ExpSyn(sec(0.5)) # E0
          s.tau = 0.1
          s.e = 50
          self.synlistex.append(s)
          s = h.Exp2Syn(sec(0.5)) # I1
          s.tau1 = 1.5
          s.tau2 = 2
          s.e = -80
          self.synlistinh.append(s)  
          s = h.ExpSyn(sec(0.8)) # I1
          s.tau = 0.1
          s.e = 50
          self.synlistees.append(s)  
    else:
      for i in range(100): 
        s = h.ExpSyn(self.dend(0.5)) # E0
        s.tau = 0.1
        s.e = 50
        self.synlistex.append(s)
        s = h.Exp2Syn(self.dend(0.5)) # I1
        s.tau1 = 1.5
        s.tau2 = 2
        s.e = -80
        self.synlistinh.append(s)  
        s = h.ExpSyn(self.dend(0.8)) # I1
        s.tau = 0.1
        s.e = 50
        self.synlistees.append(s)  

  def is_art(self):
    return 0