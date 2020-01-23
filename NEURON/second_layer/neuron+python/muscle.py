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
    self.muscle_unit = h.Section(name='muscle_unit', cell=self)

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
    self.muscle_unit.L = 10 # microns
    self.muscle_unit.diam = 10 # microns

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

    rec = h.xm(self.muscle_unit(0.5))

    self.muscle_unit.insert('CaSP')
    self.muscle_unit.insert('fHill')

  def is_art(self):
    return 0
