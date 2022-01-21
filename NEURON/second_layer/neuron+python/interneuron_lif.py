from neuron import h, gui
import random

class interneuron_lif(object):
  def __init__(self):
    self.cell = h.IntFire1()
    self.cell.m = 0.
    self.x = self.y = self.z = 0.

    def __del__(self):
    #print 'delete ', self
      pass

  def is_art(self):
    return 0
