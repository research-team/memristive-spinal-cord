import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
from neuron import h
h.load_file('nrngui.hoc')

#paralleling NEURON staff
pc = h.ParallelContext()
rank = int(pc.id()) 
nhost = int(pc.nhost())

#param
speed = 25 # duration of layer 25 = 21 cm/s; 50 = 15 cm/s; 125 = 6 cm/s
ees_i = 25 # interval between self.ees stimulus 
versions = 25
step_number = 2 # number of steps

from interneuron import interneuron
from motoneuron import motoneuron
from bioaff import bioaff

import random

'''
network creation
see topology https://github.cself.OM/research-team/memristive-spinal-cord/blob/master/doc/dself.Iagram/cpg_generatoself.R_FE_paper.png
and all will be clear
'''
class CPG:
  def __init__(self, speed, ees_int, inh_p, step_number, N = 20):

    self.interneurons = []
    self.motoneurons = []
    self.afferents = []
    self.stims = []
    self.ncell = N
    self.groups = []
    self.motogroups = []
    self.affgroups = []
   
    nMN = 200
    nAff = 120
    nInt = 196

    self.OM1_0E = self.addpool(self.ncell, False, "OM1_0E")
    self.OM1_0F = self.addpool(self.ncell, False, "OM1_0F")
    self.OM2_0 = self.addpool(self.ncell, False, "OM2_0")
    self.OM3_0 = self.addpool(self.ncell, False, "OM3_0")
    self.OM4_0 = self.addpool(self.ncell, False, "OM4_0")    
    self.OM5_0  = self.addpool(self.ncell, False, "OM5_0")

    self.OM1_1 = self.addpool(self.ncell, False, "OM1_1")
    self.OM1_2E = self.addpool(self.ncell, False, "OM1_2E")
    self.OM1_2F = self.addpool(self.ncell, False, "OM1_2F")
    self.OM1_3 = self.addpool(self.ncell, False, "OM1_3")

    self.OM2_1 = self.addpool(self.ncell, False, "OM2_1")
    self.OM2_2E = self.addpool(self.ncell, False, "OM2_2E")
    self.OM2_2F = self.addpool(self.ncell, False, "OM2_2F")
    self.OM2_3 = self.addpool(self.ncell, False, "OM2_3")
   
    self.OM3_1 = self.addpool(self.ncell, False, "OM3_1")
    self.OM3_2E = self.addpool(self.ncell, False, "OM3_2E")
    self.OM3_2F = self.addpool(self.ncell, False, "OM3_2F")
    self.OM3_3 = self.addpool(self.ncell, False, "OM3_3")
    
    self.OM4_1 = self.addpool(self.ncell, False, "OM4_1")
    self.OM4_2E = self.addpool(self.ncell, False, "OM4_2E")
    self.OM4_2F = self.addpool(self.ncell, False, "OM4_2F")
    self.OM4_3 = self.addpool(self.ncell, False, "OM4_3")

    self.OM5_1 = self.addpool(self.ncell, False, "OM5_1")
    self.OM5_2 = self.addpool(self.ncell, False, "OM5_2")
    self.OM5_3 = self.addpool(self.ncell, False, "OM5_3")
   
    self.CV1 = self.addafferents(self.ncell, "CV1")
    self.CV2 = self.addafferents(self.ncell, "CV2")
    self.CV3 = self.addafferents(self.ncell, "CV3")
    self.CV4 = self.addafferents(self.ncell, "CV4")
    self.CV5 = self.addafferents(self.ncell, "CV5")

    self.CV1_1 = self.addafferents(self.ncell, "CV1_1")
    self.CV2_1 = self.addafferents(self.ncell, "CV2_1")
    self.CV3_1 = self.addafferents(self.ncell, "CV3_1")
    self.CV4a_1 = self.addafferents(self.ncell, "CV4_1")
    self.CV4b_1 = self.addafferents(self.ncell, "CV4_1")
    self.CV5_1 = self.addafferents(self.ncell, "CV5_1")

    self.sens_aff = self.addafferents(nAff, "sens_aff")
    self.Ia_aff_E = self.addafferents(nAff, "Ia_aff_E")
    self.Ia_aff_F = self.addafferents(nAff, "Ia_aff_F")
    
    self.mns_E = self.addmotoneurons(nMN, "mns_E")
    self.mns_F = self.addmotoneurons(nMN, "mns_F")

    #interneuronal pool
    self.IP1_E = self.addpool(self.ncell, False, "IP1_E")
    self.IP1_F = self.addpool(self.ncell, False, "IP1_F")
    self.IP2_E = self.addpool(self.ncell, False, "IP2_E")
    self.IP2_F = self.addpool(self.ncell, False, "IP2_F")
    self.IP3_E = self.addpool(self.ncell, False, "IP3_E")
    self.IP3_F = self.addpool(self.ncell, False, "IP3_F")
    self.IP4_E = self.addpool(self.ncell, False, "IP4_E")
    self.IP4_F = self.addpool(self.ncell, False, "IP4_F")
    self.IP5_E = self.addpool(self.ncell, False, "IP5_E")
    self.IP5_F = self.addpool(self.ncell, False, "IP5_F")
    
    self.IP_E = [self.IP1_E, self.IP2_E, self.IP3_E, self.IP4_E, self.IP5_E]
    self.IP_F = [self.IP1_F, self.IP2_F, self.IP3_F, self.IP4_F, self.IP5_F]
    self.IP_E = sum(self.IP_E, [])
    self.IP_F = sum(self.IP_F, [])
    
    #reflex arc
    self.Ia_E = self.addpool(nInt, False, "Ia_E")
    self.iIP_E = self.addpool(nInt, False, "iIP_E")
    self.R_E = self.addpool(nInt, False, "R_E")

    self.Ia_F = self.addpool(nInt, False, "Ia_F")
    self.iIP_F = self.addpool(nInt, False, "iIP_F")
    self.R_F = self.addpool(nInt, False, "R_F")
    self.Iagener_E = []
    self.Iagener_F = []
    
    # ees
    self.ees = self.addgener(1, ees_int, 10000, False)
    
    #skin inputs
    c_int = 5
    self.C1 = []
    self.C2 = []
    self.C3 = []
    self.C4a = []
    self.C4b = []
    self.C5 = []
    self.C_1 = []
    self.C_0 = []

    for i in range(step_number):
        self.C1.append(self.addgener(speed*0 + i*(speed*6 + 125), c_int, speed/c_int))
    for i in range(step_number):
        self.C2.append(self.addgener(speed*1 + i*(speed*6 + 125), c_int, speed/c_int))
    for i in range(step_number):
        self.C3.append(self.addgener(speed*2 + i*(speed*6 + 125), c_int, speed/c_int))
    for i in range(step_number):
        self.C4a.append(self.addgener(speed*3 + i*(speed*6 + 125), c_int, speed/c_int))
    for i in range(step_number):
        self.C4b.append(self.addgener(speed*4 + i*(speed*6 + 125), c_int, speed/c_int))
    for i in range(step_number):
        self.C5.append(self.addgener(speed*5 + i*(speed*6 + 125), c_int, speed/c_int))
    for i in range(step_number):
        self.Iagener_E.append(self.addIagener((1 + i*(speed*6 + 125)), self.ncell, speed))
    for i in range(step_number):
        self.Iagener_F.append(self.addIagener((speed*6 + i*(speed*6 + 125)), self.ncell, 25))
    for i in range(step_number):
        self.C_0.append(self.addgener(speed*6 + i*(speed*6 + 125), c_int, (125/c_int-1), False))
    '''
    for i in range(step_number):
        self.C_1.append(self.addgener(speed*0 + i*(speed*6 + 125), c_int, 6*speed/c_int))
    '''

    self.C_1 = [self.CV1_1, self.CV2_1, self.CV3_1, self.CV4a_1, self.CV4b_1, self.CV5_1]
    self.C_1 = sum(self.C_1, [])
    
    self.Iagener_E = sum(self.Iagener_E, [])
    self.Iagener_F = sum(self.Iagener_F, [])

  def updateconnections(self):
    self.exnclist = []
    self.inhnclist = []
    self.eesnclist = []
    self.stimnclist = []

    #generators
    self.createmotif(self.OM1_0E, self.OM1_1, self.OM1_2E, self.OM1_3) 
    self.createmotif(self.OM2_0, self.OM2_1, self.OM2_2E, self.OM2_3)
    self.createmotif(self.OM3_0, self.OM3_1, self.OM3_2E, self.OM3_3)
    self.createmotif(self.OM4_0, self.OM4_1, self.OM4_2E, self.OM4_3)
    self.createmotif(self.OM5_0, self.OM5_1, self.OM5_2, self.OM5_3)

    #extra flexor connections
    self.exconnectcells(self.OM2_2F, self.OM3_2F, 0.08, 2, 50)
    self.exconnectcells(self.OM3_2F, self.OM4_2F, 0.08, 2, 50)
    self.exconnectcells(self.OM4_2F, self.OM5_0, 0.08, 2, 50)

    self.exconnectcells(self.OM1_1, self.OM1_2F, 0.05, 2, 27)
    self.exconnectcells(self.OM1_2F, self.OM1_1, 0.05, 3, 27)
    self.exconnectcells(self.OM2_1, self.OM2_2F, 0.05, 2, 27)
    self.exconnectcells(self.OM2_2F, self.OM2_1, 0.05, 3, 27)
    self.exconnectcells(self.OM3_1, self.OM3_2F, 0.05, 2, 27)
    self.exconnectcells(self.OM3_2F, self.OM3_1, 0.05, 3, 27)
    self.exconnectcells(self.OM4_1, self.OM4_2F, 0.05, 2, 27)
    self.exconnectcells(self.OM4_2F, self.OM4_1, 0.05, 3, 27)
    
    self.exconnectcells(self.CV1, self.OM1_0F, 0.002, 2, 50)
    self.exconnectcells(self.OM1_0F, self.OM1_1, 0.05, 2, 27)
    self.exconnectcells(self.OM1_2F, self.OM2_2F , 0.05, 3, 50)

    #between delays vself.Ia excitatory pools
    #extensor
    self.exconnectcells(self.CV1, self.CV2, 0.5, 1, 27)
    self.exconnectcells(self.CV2, self.CV3, 0.5, 1, 27)
    self.exconnectcells(self.CV3, self.CV4, 0.5, 1, 27)
    self.exconnectcells(self.CV4, self.CV5, 0.5, 1, 27)

    self.exconnectcells(self.CV1, self.OM1_0E, 0.00037, 1, 27)
    self.exconnectcells(self.CV2, self.OM2_0, 0.00037, 1, 27)
    self.exconnectcells(self.CV3, self.OM3_0, 0.00037, 1, 27)
    self.exconnectcells(self.CV4, self.OM4_0, 0.00037, 1, 27)
    self.exconnectcells(self.CV5, self.OM5_0, 0.00039, 1, 27)

    #inhibitory projections
    #extensor
    self.exconnectcells(self.C3, self.OM1_3, 0.9, 1, 80)

    self.exconnectcells(self.C4a, self.OM1_3, 0.9, 1, 80)
    self.exconnectcells(self.C4a, self.OM2_3, 0.9, 1, 80)

    self.exconnectcells(self.C4b, self.OM1_3, 0.9, 1, 80)
    self.exconnectcells(self.C4b, self.OM2_3, 0.9, 1, 80)

    self.exconnectcells(self.C5, self.OM1_3, 0.9, 1, 80)
    self.exconnectcells(self.C5, self.OM2_3, 0.9, 1, 80)
    self.exconnectcells(self.C5, self.OM3_3, 0.9, 1, 80)
    #self.exconnectcells(self.CV5_1, self.OM4_3, 0.8, 1, 50)
    
    #ees
    self.genconnect(self.ees, self.Ia_aff_F, 1, 0, random.randint(10, 50))
    self.genconnect(self.ees, self.Ia_aff_E, 1, 0, random.randint(10, 50))
    self.genconnect(self.ees, self.CV1, 0.5, 1, random.randint(10, 50))

    self.exconnectcells(self.Ia_aff_E, self.mns_E, 0.8, 2, random.randint(10, 50))
    self.exconnectcells(self.Ia_aff_F, self.mns_F, 0.8, 2, random.randint(10, 50))

    #IP
    #Extensor
    self.exconnectcells(self.OM1_2E, self.IP1_E, 0.5, 1, 50)
    self.exconnectcells(self.OM2_2E, self.IP2_E, 0.5, 2, 50)
    self.exconnectcells(self.OM3_2E, self.IP3_E, 0.5, 3, 50)
    self.exconnectcells(self.OM4_2E, self.IP4_E, 0.5, 3, 50)
    self.exconnectcells(self.OM5_2, self.IP5_E, 0.5, 3, 50)

    self.exconnectcells(self.IP3_E, self.mns_E, 0.8, 3, random.randint(10, 50))
    self.exconnectcells(self.IP2_E, self.mns_E[:int(len(self.mns_E)/2)], 0.8, 2, random.randint(10, 50))
    self.exconnectcells(self.IP5_E, self.mns_E[:int(len(self.mns_E)/2)], 0.8, 3, random.randint(10, 50))
    self.exconnectcells(self.IP4_E, self.mns_E, 0.8, 3, random.randint(10, 50))
    self.exconnectcells(self.IP1_E, self.mns_E[:int(len(self.mns_E)/3)], 0.4, 1, random.randint(10, 50))

    self.inhconnectcells(self.IP1_E, self.Ia_aff_E, 0.025, 2, 80)
    self.inhconnectcells(self.IP2_E, self.Ia_aff_E, 0.025, 2, 80)
    self.inhconnectcells(self.IP3_E, self.Ia_aff_E, 0.07, 2, 80)
    self.inhconnectcells(self.IP4_E, self.Ia_aff_E, 0.08, 2, 80)

    #Flexor
    self.exconnectcells(self.OM1_2E, self.IP1_F, 0.5, 1, 50)
    self.exconnectcells(self.OM2_2F, self.IP2_F, 0.5, 2, 50)
    self.exconnectcells(self.OM3_2F, self.IP3_F, 0.5, 1, 50)
    self.exconnectcells(self.OM4_2F, self.IP4_F, 0.5, 1, 50)
    self.exconnectcells(self.OM5_2, self.IP5_F, 0.5, 2, 50)

    self.exconnectcells(self.IP3_F, self.mns_F, 0.5, 2, random.randint(10, 50))
    self.exconnectcells(self.IP2_F, self.mns_F[:int(3*len(self.mns_E)/5)], 0.5, 2, random.randint(10, 50))
    self.exconnectcells(self.IP5_F, self.mns_F[:int(3*len(self.mns_E)/5)], 0.5, 2, random.randint(10, 50))
    self.exconnectcells(self.IP4_F, self.mns_F[int(len(self.mns_E)/5):], 0.5, 2, random.randint(10, 50))
    self.exconnectcells(self.IP1_F, self.mns_F[:int(3*len(self.mns_E)/5)], 0.5, 2, random.randint(10, 50))

    self.inhconnectcells(self.IP_F, self.Ia_aff_F, 0.0001, 2, 80)
  
    #skin inputs
    self.exconnectcells(self.C1, self.CV1_1, 0.8, 1, 50)
    self.exconnectcells(self.C2, self.CV2_1, 0.8, 1, 50)
    self.exconnectcells(self.C3, self.CV3_1, 0.8, 2, 50)
    self.exconnectcells(self.C4a, self.CV4a_1, 0.8, 2, 50)
    self.exconnectcells(self.C4b, self.CV4b_1, 0.8, 2, 50)
    self.exconnectcells(self.C5, self.CV5_1, 0.8, 2, 50)

    #C1
    self.exconnectcells(self.CV1_1, self.OM1_0E, 0.0003, 1, 30)

    #C2
    self.exconnectcells(self.CV2_1, self.OM1_0E, 0.0003, 1, 30)
    self.exconnectcells(self.CV2_1, self.OM2_0, 0.0004, 1, 27)

    #C3
    self.exconnectcells(self.CV3_1, self.OM2_0, 0.0004, 1, 27)
    self.exconnectcells(self.CV3_1, self.OM3_0 , 0.0004, 1, 27)

    #C4
    self.exconnectcells(self.CV4a_1, self.OM3_0 , 0.00035, 3, 27)
    self.exconnectcells(self.CV4a_1, self.OM4_0, 0.00038, 3, 27)
    self.exconnectcells(self.CV4b_1, self.OM3_0 , 0.00032, 3, 27)
    self.exconnectcells(self.CV4b_1, self.OM4_0, 0.0003, 3, 27)

    #C5
    self.exconnectcells(self.CV5_1, self.OM5_0 , 0.0003, 3, 27)
    self.exconnectcells(self.CV5_1, self.OM4_0, 0.0001, 3, 22)
    
    #C=1 Extensor
    self.exconnectcells(self.IP_E, self.iIP_E, 0.5, 1, 50)

    self.exconnectcells(self.CV1_1, self.iIP_E, 0.5, 1, 50)
    self.exconnectcells(self.CV2_1, self.iIP_E, 0.5, 1, 50)
    self.exconnectcells(self.CV3_1, self.iIP_E, 0.5, 1, 50)
    self.exconnectcells(self.CV4a_1, self.iIP_E, 0.5, 1, 50)
    self.exconnectcells(self.CV4b_1, self.iIP_E, 0.5, 1, 50)
    self.exconnectcells(self.CV5_1, self.iIP_E, 0.5, 1, 50)
    
    self.inhconnectcells(self.iIP_E, self.OM1_0F, 0.99, 1, 100)
    self.inhconnectcells(self.iIP_E, self.OM1_2F, 0.99, 1, 100)
    self.inhconnectcells(self.iIP_E, self.OM2_2F, 0.99, 1, 100)
    self.inhconnectcells(self.iIP_E, self.OM4_2F, 0.99, 1, 100)
    self.inhconnectcells(self.iIP_E, self.OM3_2F, 0.99, 1, 100)

    self.inhconnectcells(self.iIP_E, self.IP_F, 0.99, 1, 100)
    self.inhconnectcells(self.iIP_E, self.Ia_aff_F, 0.99, 1, 100)
    self.inhconnectcells(self.iIP_E, self.mns_F, 0.99, 1, 100)

    #self.exconnectcells(self.C_1, self.iIP_E, 0.5, 1, 50)
    #self.inhconnectcells(self.iIP_E, self.IP_F, 0.99, 1, 100)
    '''
    self.inhconnectcells(self.IP_E, self.Ia_aff_F, 0.99, 1, 100)
    self.inhconnectcells(self.IP_E, self.mns_F, 0.99, 1, 100)
    self.inhconnectcells(self.iIP_E, self.OM1_0F, 0.8, 1, 80)
    self.inhconnectcells(self.iIP_E, self.OM2_2F, 0.8, 1, 80)
    self.inhconnectcells(self.iIP_E, self.OM4_2F, 0.8, 1, 80)
    self.inhconnectcells(self.iIP_E, self.OM3_2F, 0.8, 1, 80)
    self.inhconnectcells(self.iIP_E, self.IP_F, 0.9, 1, 100)
    self.inhconnectcells(self.iIP_E, self.Ia_aff_F, 0.9, 1, 80)
    self.inhconnectcells(self.IP_E, self.mns_F, 0.9, 1, 100)
    self.inhconnectcells(self.iIP_E, self.mns_F, 0.9, 1, 100)
    ''' 
    #C=0 Flexor
    #self.exconnectcells(self.IP_F, self.iIP_F, 0.9, 1, 50)
    self.inhconnectcells(self.iIP_F, self.IP_E, 0.99, 1, 80)
    self.inhconnectcells(self.C_0, self.Ia_aff_E, 0.9, 1, 80)
    self.inhconnectcells(self.C_0, self.IP_E, 0.99, 1, 80)
      
    #reflex arc
    # self.exconnectcells(self.Iagener_E, self.Ia_aff_E[:int(len(self.Ia_aff_E)/4)], 0.08, 1, 20)
    #self.exconnectcells(self.Ia_aff_E, self.Ia_E, 0.5, 1, 30)
    self.exconnectcells(self.iIP_E, self.Ia_E, 0.5, 1, 50)
    self.exconnectcells(self.Ia_aff_E[:int(len(self.Ia_aff_E)/6)], self.Ia_E, 0.8, 1, 30)
    self.exconnectcells(self.mns_E, self.R_E, 0.00025, 1, 30)
    self.inhconnectcells(self.Ia_E, self.mns_F, 0.08, 1, 45)
    self.inhconnectcells(self.R_E, self.mns_E, 0.005, 1, 45)
    self.inhconnectcells(self.R_E, self.Ia_E, 0.001, 1, 40)

    # self.exconnectcells(self.Iagener_F, self.Ia_aff_F[:int(len(self.Ia_aff_F)/6)], 0.02, 1, 20)
    #self.exconnectcells(self.Ia_aff_F, self.Ia_F, 0.5, 1, 30)
    self.exconnectcells(self.iIP_F, self.Ia_F, 0.5, 1, 50)
    self.exconnectcells(self.Ia_aff_F[:int(len(self.Ia_aff_F)/6)], self.Ia_F, 0.8, 1, 30)
    self.exconnectcells(self.mns_F, self.R_F, 0.0004, 1, 30)
    self.inhconnectcells(self.Ia_F, self.mns_E, 0.04, 1, 45)
    self.inhconnectcells(self.R_F, self.mns_F, 0.005, 1, 45)
    self.inhconnectcells(self.R_F, self.Ia_F, 0.001, 1, 20)

    self.inhconnectcells(self.R_E, self.R_F, 0.04, 1, 30)
    self.inhconnectcells(self.Ia_E, self.Ia_F, 0.08, 1, 30)
    self.inhconnectcells(self.R_F, self.R_E, 0.04, 1, 30)
    self.inhconnectcells(self.Ia_F, self.Ia_E, 0.08, 1, 50)
    self.inhconnectcells(self.iIP_E, self.iIP_F, 0.04, 1, 30)
    self.inhconnectcells(self.iIP_F, self.iIP_E, 0.04, 1, 30)
    
  def addpool(self, num, delaytype, name="test"):
    '''
    Creates interneuronal pool and returns gids of pool
    Parameters
    ----------
    num: int
        neurons number in pool
    delaytype: bool
        Does it have 5ht receptors?
        -Yes: True 
        -No: False
    Returns
    -------
    gids: list
        the list of neurons gids
    '''
    gids = []
    gid = 0
    for i in range(rank, num, nhost):
      cell = interneuron(delaytype)
      self.interneurons.append(cell)
      while(pc.gid_exists(gid)!=0):
        gid+=1
      gids.append(gid)
      pc.set_gid2node(gid, rank)
      nc = cell.connect2target(None)
      pc.cell(gid, nc)
    
    # ToDo remove me (Alex code) - NO
    self.groups.append((gids, name))

    return gids

  def addmotoneurons(self, num, name):
    '''
    Creates pool of motoneurons and returns gids of pool
    Parameters
    ----------
    num: int
        neurons number in pool
    Returns
    -------
    gids: list
        list of neurons gids
    '''
    gids = []
    gid = 0
    for i in range(rank, num, nhost):
      cell = motoneuron()
      self.motoneurons.append(cell)
      while(pc.gid_exists(gid)!= 0):
        gid+=1
      gids.append(gid)
      pc.set_gid2node(gid, rank)
      nc = cell.connect2target(None)
      pc.cell(gid, nc)

    self.motogroups.append((gids, name))

    return gids

  def addafferents(self, num, name):
    '''
    Creates pool of afferents and returns gids of pool
    Parameters
    ----------
    num: int
        neurons number in pool
    Returns
    -------
    gids: list
        list of neurons gids
    '''
    gids = []
    gid = 0
    for i in range(rank, num, nhost):
      cell = bioaff(random.randint(1, 3))
      self.afferents.append(cell)
      while(pc.gid_exists(gid)!=0):
        gid+=1
      gids.append(gid)
      pc.set_gid2node(gid, rank)
      nc = cell.connect2target(None)
      pc.cell(gid, nc)

    self.affgroups.append((gids, name))
    return gids

  def addgener(self, start, interval, nums, r=True):
    '''
    Creates generator and returns generator gid
    Parameters
    ----------
    start: int
        generator start up
    interval: int
        interval between signals in generator 
    nums: int
        signals number
    Returns
    -------
    gid: int
        generator gid 
    '''
    gid = 0
    stim = h.NetStim()
    stim.number = nums
    if r:
      stim.start = random.uniform(start, start + 4)
      stim.noise = 0.1
    else:
      stim.start = random.uniform(start, start + 1)
    stim.interval = interval
    #skinstim.noise = 0.1
    self.stims.append(stim)
    while(pc.gid_exists(gid)!=0):
        gid+=1
    pc.set_gid2node(gid, rank)
    ncstim = h.NetCon(stim, None)
    pc.cell(gid, ncstim)
    return gid
  
  def addIagener(self, start, num, speed):
    '''
    Creates self.Ia generators and returns generator gids
    Parameters
    ----------
    start: int
        generator start up
    num: int
        number in pool
    Returns
    -------
    gids: list
        generators gids
    '''
    gids = []
    gid = 0
    for i in range(rank, num, nhost):
      stim = h.SpikeGenerator(0.5)
      stim.start = start
      stim.number = 1000000
      stim.speed = speed*6 
      stim.k = 3/stim.speed
      self.stims.append(stim)
      while(pc.gid_exists(gid)!=0):
        gid+=1
      gids.append(gid)
      pc.set_gid2node(gid, rank)
      ncstim = h.NetCon(stim, None)
      pc.cell(gid, ncstim)
    
    return gids

  def exconnectcells(self, pre, post, weight, delay, nsyn):
    ''' Connects with excitatory synapses 
      Parameters
      ----------
      pre: list
          list of presynase neurons gids
      post: list
          list of postsynapse neurons gids
      weight: float
          weight of synapse
          used with Gaussself.Ian distribution
      delay: int
          synaptic delay
          used with Gaussself.Ian distribution
      nsyn: int
          numder of synapses
    '''
    random_range = random.randint(5, 15)
    for i in post:
      if pc.gid_exists(i):
        for j in range(nsyn):
          srcgid = random.randint(pre[0], pre[-1])
          target = pc.gid2cell(i)
          syn = target.synlistex[j]
          nc = pc.gid_connect(srcgid, syn)
          self.exnclist.append(nc)
          nc.delay = random.gauss(delay, delay/random_range)
          nc.weight[0] = random.gauss(weight, weight/random_range)

  def inhconnectcells(self, pre, post, weight, delay, nsyn):
    ''' Connects with inhibitory synapses 
      Parameters
      ----------
      pre: list
          list of presynase neurons gids
      post: list
          list of postsynapse neurons gids
      weight: float
          weight of synapse
          used with Gaussself.Ian distribution
      delay: int
          synaptic delay
          used with Gaussself.Ian distribution
      nsyn: int
          numder of synapses
    '''
    random_range = random.randint(5, 15)
    for i in post:
      if pc.gid_exists(i):
        for j in range(nsyn):
          srcgid = random.randint(pre[0], pre[-1]) 
          target = pc.gid2cell(i)
          syn = target.synlistinh[j]
          nc = pc.gid_connect(srcgid, syn)
          self.inhnclist.append(nc)
          nc.delay = random.gauss(delay, delay/random_range)
          nc.weight[0] = random.gauss(weight, weight/random_range)

  def genconnect(self, gen_gid, afferents_gids, weight, delay, nsyn):
    ''' Connects with generator 
      Parameters
      ----------
      afferents_gids: list
          list of presynase neurons gids
      gen_gid: int
          generator gid
      weight: float
          weight of synapse
          used with Gaussself.Ian distribution
      delay: int
          synaptic delay
          used with Gaussself.Ian distribution
      nsyn: int
          numder of synapses
    '''
    random_range = random.randint(5, 15)
    for i in afferents_gids:
      if pc.gid_exists(i):
        for j in range(nsyn): 
          target = pc.gid2cell(i)
          syn = target.synlistees[j]
          nc = pc.gid_connect(gen_gid, syn)
          self.stimnclist.append(nc)
          nc.delay = random.gauss(delay, delay/random_range)
          nc.weight[0] = random.gauss(weight, weight/random_range)

  def inhgenconnect(self, gen_gid, afferents_gids, weight, delay, nsyn):
    ''' Connects with generator vself.Ia inhibitory synapses 
      Parameters
      ----------
      gen_gid: int
          generator gid
      afferents_gids: list
          list of presynase neurons gids
      weight: float
          weight of synapse
          used with Gaussself.Ian distribution
      delay: int
          synaptic delay
          used with Gaussself.Ian distribution
      nsyn: int
          numder of synapses
    '''
    for i in afferents_gids:
      if pc.gid_exists(i):
        for j in range(nsyn): 
          target = pc.gid2cell(i)
          syn = target.synlistinh[j]
          nc = pc.gid_connect(gen_gid, syn)
          self.stimnclist.append(nc)
          nc.delay = random.gauss(delay, delay/10)
          nc.weight[0] = random.gauss(weight, weight/10)

  def createmotif(self, OM0, OM1, OM2, OM3):
    ''' Connects motif module
      see https://github.cself.OM/research-team/memristive-spinal-cord/blob/master/doc/dself.Iagram/cpg_generatoself.R_FE_paper.png
      Parameters
      ----------
      self.OM0: list
          list of self.OM0 pool gids
      self.OM1: list
          list of self.OM1 pool gids
      self.OM2: list
          list of self.OM2 pool gids
      self.OM3: list
          list of self.OM3 pool gids
    '''
    self.exconnectcells(OM0, OM1, 0.05, 1, 27)
    self.exconnectcells(OM1, OM2, 0.05, 2, 27)
    self.exconnectcells(OM2, OM1, 0.05, 2, 27)
    self.exconnectcells(OM2, OM3, 0.00001, 1, 27)
    #self.exconnectcells(g1, g3, 0.001, 1, 27)
    #self.inhconnectcells(g3, g1, 0.08, 1, 27)
    self.inhconnectcells(OM3, OM2, 0.9, 1, 99)

def spike_record(pool, version):
  ''' Records spikes from gids 
    Parameters
    ----------
    pool: list
      list of neurons gids
    version: int
        test number
    Returns
    -------
    v_vec: list of h.Vector()
        recorded voltage
  '''
  v_vec = []

  for i in pool:
    cell = pc.gid2cell(i)
    vec = h.Vector()
    vec.record(cell.soma(0.5)._ref_vext[0])
    v_vec.append(vec)
  return v_vec

def avgarr(z):
  ''' Summarizes extracellular voltage in pool
    Parameters
    ----------
    z: list
      list of neurons voltage
    Returns
    -------
    summa: list 
        list of summarized voltage
  '''
  summa = 0
  for item in z:
    summa += np.array(item)
  return summa
 
def spikeout(pool, name, version, v_vec):
  ''' Reports simulation results 
    Parameters
    ----------
    pool: list
      list of neurons gids
    name: string
      pool name
    version: int
        test number
    v_vec: list of h.Vector()
        recorded voltage
  '''
  global rank 
  pc.barrier()
  for i in range(nhost):
    if i == rank:
      outavg = []
      for j in range(len(pool)):
        outavg.append(list(v_vec[j]))
      outavg = avgarr(outavg)
      path=str('./res/'+ name + 'r%dv%d'%(rank, version))
      f = open(path, 'w')
      for v in outavg:
          f.write(str(v)+"\n")
    pc.barrier()

def prun(speed, step_number):
  ''' simulation control 
  Parameters
  ----------
  speed: int
    duration of each layer 
  '''
  tstop = (6*speed + 125)*step_number
  pc.set_maxstep(10)
  h.stdinit()
  pc.psolve(tstop)

def finish():
  ''' proper exit '''
  pc.runworker()
  pc.done()
  h.quit()

if __name__ == '__main__':
    '''
    cpg_ex: cpg
        topology of central pattern generation + reflex arc 
    '''
    k_nrns = 0
    k_name = 1
    cpg_ex = CPG(speed, ees_i, 100, step_number)

    for i in range(versions): 
      cpg_ex.updateconnections()         
      motorecorders = []
      for group in cpg_ex.motogroups:
        motorecorders.append(spike_record(group[k_nrns], i))
      # affrecorders = []
      # for group in cpg_ex.affgroups:
      #   affrecorders.append(spike_record(group[k_nrns], i))   
      # recorders = []
      # for group in cpg_ex.groups:
      #   recorders.append(spike_record(group[k_nrns], i))

      print("- "*10, "\nstart")
      prun(speed, step_number)
      print("- "*10, "\nend")

      for group, recorder in zip(cpg_ex.motogroups, motorecorders):
        spikeout(group[k_nrns], group[k_name], i, recorder)
      # for group, recorder in zip(cpg_ex.affgroups, affrecorders):
      #   spikeout(group[k_nrns], group[k_name], i, recorder)
      # for group, recorder in zip(cpg_ex.groups, recorders):
      #   spikeout(group[k_nrns], group[k_name], i, recorder)
           
    #if (nhost > 1):
    finish()