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

    def __del__(self):
    #print 'delete ', self
      pass

  def topol(self):
    '''
    Creates section
    '''
    self.muscle = h.Section(name='muscle', cell=self)

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
    self.muscle.L = 10 # microns
    self.muscle.diam = 10 # microns

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
    self.muscle.cm = 20 # cm uf/cm2
    self.muscle.insert('pas')
    self.muscle.g_pas = 0.002

    rec = h.xm(self.muscle(0.5))

    self.muscle.insert('CaSP')
    self.muscle.insert('fHill')

  def is_art(self):
    return 0
