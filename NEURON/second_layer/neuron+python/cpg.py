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
EES_i = 25 # interval between EES stimulus 
versions = 1
step_number = 1 # number of steps

from interneuron import interneuron
from motoneuron import motoneuron
from bioaff import bioaff

import random

'''
network creation
see topology https://github.com/research-team/memristive-spinal-cord/blob/master/doc/diagram/cpg_generator_FE_paper.png
and all will be clear
'''
class CPG:
  def __init__(self, speed, EES_int, inh_p, step_number, N = 20):

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

    OM1_0E = self.addpool(self.ncell, False, "OM1_0E")
    OM1_0F = self.addpool(self.ncell, False, "OM1_0F")
      
    OM2_0E = self.addpool(self.ncell, False, "OM2_0E")
    OM2_0F = self.addpool(self.ncell, False, "OM2_0F")
    
    OM3_0 = self.addpool(self.ncell, False, "OM3_0")

    OM4_0E = self.addpool(self.ncell, False, "OM4_0E")
    OM4_0F = self.addpool(self.ncell, False, "OM4_0F")
    
    OM5_0  = self.addpool(self.ncell, False, "OM5_0")

    OM1_1 = self.addpool(self.ncell, False, "OM1_1")
    OM1_2 = self.addpool(self.ncell, False, "OM1_2")
    OM1_3 = self.addpool(self.ncell, False, "OM1_3")

    OM2_1 = self.addpool(self.ncell, False, "OM2_1")
    OM2_2 = self.addpool(self.ncell, False, "OM2_2")
    OM2_3 = self.addpool(self.ncell, False, "OM2_3")
   
    OM3_1 = self.addpool(self.ncell, False, "OM3_1")
    OM3_2E = self.addpool(self.ncell, False, "OM3_2E")
    OM3_2F = self.addpool(self.ncell, False, "OM3_2F")
    OM3_3 = self.addpool(self.ncell, False, "OM3_3")
    
    OM4_1 = self.addpool(self.ncell, False, "OM4_1")
    OM4_2 = self.addpool(self.ncell, False, "OM4_2")
    OM4_3 = self.addpool(self.ncell, False, "OM4_3")

    OM5_1 = self.addpool(self.ncell, False, "OM5_1")
    OM5_2 = self.addpool(self.ncell, False, "OM5_2")
    OM5_3 = self.addpool(self.ncell, False, "OM5_3")
   
    CV1 = self.addafferents(self.ncell, "CV1")
    CV2 = self.addafferents(self.ncell, "CV2")
    CV3 = self.addafferents(self.ncell, "CV3")
    CV4 = self.addafferents(self.ncell, "CV4")
    CV5 = self.addafferents(self.ncell, "CV5")

    CV1_1 = self.addafferents(self.ncell, "CV1_1")
    CV2_1 = self.addafferents(self.ncell, "CV2_1")
    CV3_1 = self.addafferents(self.ncell, "CV3_1")
    CV4_1 = self.addafferents(self.ncell, "CV4_1")
    CV5_1 = self.addafferents(self.ncell, "CV5_1")

    sens_aff = self.addafferents(nAff, "sens_aff")
    Ia_aff_E = self.addafferents(nAff, "Ia_aff_E")
    Ia_aff_F = self.addafferents(nAff, "Ia_aff_F")

    mns_E = self.addmotoneurons(nMN, "mns_E")
    mns_F = self.addmotoneurons(nMN, "mns_F")

    #interneuronal pool
    IP1_E = self.addpool(self.ncell, False, "IP1_E")
    IP1_F = self.addpool(self.ncell, False, "IP1_F")
    IP2_E = self.addpool(self.ncell, False, "IP2_E")
    IP2_F = self.addpool(self.ncell, False, "IP2_F")
    IP3_E = self.addpool(self.ncell, False, "IP3_E")
    IP3_F = self.addpool(self.ncell, False, "IP3_F")
    IP4_E = self.addpool(self.ncell, False, "IP4_E")
    IP4_F = self.addpool(self.ncell, False, "IP4_F")
    IP5_E = self.addpool(self.ncell, False, "IP5_E")
    IP5_F = self.addpool(self.ncell, False, "IP5_F")

    # EES
    ees = self.addgener(1, EES_int, 10000)
    
    #skin inputs
    c_int = 5
    C1 = []
    C2 = []
    C3 = []
    C4 = []
    C5 = []
    C_1 = []
    C_0 = []
    Iagener_E = []

    for i in range(step_number):
        C1.append(self.addgener(speed*0 + i*(speed*6 + 125), c_int, speed/c_int))
    for i in range(step_number):
        C2.append(self.addgener(speed*1 + i*(speed*6 + 125), c_int, speed/c_int))
    for i in range(step_number):
        C3.append(self.addgener(speed*2 + i*(speed*6 + 125), c_int, speed/c_int))
    for i in range(step_number):
        C4.append(self.addgener(speed*3 + i*(speed*6 + 125), c_int, 2*speed/c_int))
    for i in range(step_number):
        C5.append(self.addgener(speed*5 + i*(speed*6 + 125), c_int, speed/c_int))
    for i in range(step_number):
        Iagener_E.append(self.addIagener((1 + i*(speed*6 + 125)), self.ncell, speed))
    '''
    for i in range(step_number):
        C_1.append(self.addgener(speed*0 + i*(speed*6 + 125), c_int, 6*speed/c_int))
    '''
    for i in range(step_number):
        C_0.append(self.addgener(speed*6 + i*(speed*6 + 125), c_int, 125/c_int))

    #reflex arc
    Ia_E = self.addpool(nInt, False, "Ia_E")
    R_E = self.addpool(nInt, False, "R_E")

    Ia_F = self.addpool(nInt, False, "Ia_F")
    R_F = self.addpool(nInt, False, "R_F")
   
    C_1.append(CV1_1)
    C_1.append(CV2_1)
    C_1.append(CV3_1)
    C_1.append(CV4_1)
    C_1.append(CV5_1)
    C_1.append(IP1_E)
    C_1.append(IP2_E)
    C_1.append(IP3_E)
    C_1.append(IP4_E)
    C_1.append(IP5_E)
    #C_1.append(Ia_E)

    C_1 = np.ravel(C_1)
    Iagener_E = np.ravel(Iagener_E)

    #generators
    createmotif(OM1_0E, OM1_1, OM1_2, OM1_3) 
    createmotif(OM2_0E, OM2_1, OM2_2, OM2_3)
    createmotif(OM3_0, OM3_1, OM3_2E, OM3_3)
    exconnectcells(OM3_1, OM3_2F, 0.05, 3, 27)
    exconnectcells(OM3_2F, OM3_1, 0.05, 4, 27)
    createmotif(OM4_0E, OM4_1, OM4_2, OM4_3)
    createmotif(OM5_0, OM5_1, OM5_2, OM5_3)

    #extra flexor connections
    exconnectcells(OM2_0F, OM3_0, 0.005, 1, 27)
    exconnectcells(OM3_2F, OM4_2, 0.005, 1, 27)
    exconnectcells(OM4_0F, OM5_0, 0.0035, 1, 27)

    exconnectcells(CV1, OM1_0F, 0.05, 1, 27)
    exconnectcells(CV2, OM2_0F, 0.005, 1, 27)
    exconnectcells(CV4, OM4_0F, 0.005, 1, 27)

    exconnectcells(OM1_0F, OM1_1, 0.05, 2, 27)
    exconnectcells(OM1_0F, OM2_2 , 0.05, 1, 27)
    exconnectcells(OM2_0F, OM2_1, 0.05, 2, 27)
    exconnectcells(OM4_0F, OM4_1, 0.05, 2, 27)

    #between delays via excitatory pools
    #extensor
    exconnectcells(CV1, CV2, 0.5, 1, 27)
    exconnectcells(CV2, CV3, 0.5, 1, 27)
    exconnectcells(CV3, CV4, 0.5, 1, 27)
    exconnectcells(CV4, CV5, 0.5, 1, 27)

    exconnectcells(CV1, OM1_0E, 0.0037, 1, 27)
    exconnectcells(CV2, OM2_0E, 0.00037, 1, 27)
    exconnectcells(CV3, OM3_0, 0.00037, 1, 27)
    exconnectcells(CV4, OM4_0E, 0.00037, 1, 27)
    exconnectcells(CV5, OM5_0, 0.00037, 1, 27)

    #inhibitory projections
    #extensor
    exconnectcells(CV3_1, OM1_3, 0.8, 1, 50)

    exconnectcells(CV4_1, OM1_3, 0.8, 1, 50)
    exconnectcells(CV4_1, OM2_3, 0.8, 1, 50)

    exconnectcells(CV5_1, OM1_3, 0.8, 1, 50)
    exconnectcells(CV5_1, OM2_3, 0.8, 1, 50)
    exconnectcells(CV5_1, OM3_3, 0.8, 1, 50)
    #exconnectcells(C5, OM4_3, 0.8, 1, 50)
    
    #EES
    genconnect(ees, Ia_aff_F, 1, 0, 50)
    genconnect(ees, Ia_aff_E, 1, 0, 50)
    genconnect(ees, CV1, 0.5, 0, 50)

    exconnectcells(Ia_aff_E, mns_E, 0.8, 1, 50)
    exconnectcells(Ia_aff_F, mns_F, 0.5, 1, 50)

    #IP
    #Extensor
    exconnectcells(OM1_2, IP1_E, 0.5, 1, 50)
    exconnectcells(OM2_2, IP2_E, 0.5, 2, 50)
    exconnectcells(OM3_2E, IP3_E, 0.5, 2, 50)
    exconnectcells(OM4_2, IP4_E, 0.5, 2, 50)
    exconnectcells(OM5_2, IP5_E, 0.5, 2, 50)

    exconnectcells(IP3_E, mns_E, 0.8, 2, 80)
    exconnectcells(IP2_E, mns_E[:int(3*len(mns_E)/5)], 0.8, 2, 80)
    exconnectcells(IP5_E, mns_E[:int(3*len(mns_E)/5)], 0.8, 2, 80)
    exconnectcells(IP4_E, mns_E[int(len(mns_E)/5):], 0.8, 2, 60)
    exconnectcells(IP1_E, mns_E[:int(len(mns_E)/5)], 0.8, 2, 60)

    inhconnectcells(IP1_E, Ia_aff_E, 0.002, 2, 40)
    #inhconnectcells(IP2_E, Ia_aff_E, 0.0001, 2, 40)
    inhconnectcells(IP3_E, Ia_aff_E, 0.0015, 2, 30)
    inhconnectcells(IP4_E, Ia_aff_E, 0.02, 2, 40)
    inhconnectcells(IP5_E, Ia_aff_E, 0.01, 2, 40)

    #Flexor
    exconnectcells(OM1_2, IP1_F, 0.5, 2, 50)
    exconnectcells(OM2_2, IP2_F, 0.5, 1, 50)
    exconnectcells(OM3_2F, IP3_F, 0.5, 2, 50)
    exconnectcells(OM4_2, IP4_F, 0.5, 2, 50)
    exconnectcells(OM5_2, IP5_F, 0.5, 2, 50)
    
    exconnectcells(IP1_F, mns_F[:2*int(len(mns_F)/5)], 0.8, 2, 60)
    exconnectcells(IP2_F, mns_F[int(len(mns_F)/5):int(3*len(mns_F)/5)], 0.8, 2, 80)
    exconnectcells(IP3_F, mns_F, 0.8, 2, 80)
    exconnectcells(IP4_F, mns_F[int(2*len(mns_F)/5):], 0.8, 2, 80)
    exconnectcells(IP5_F, mns_F[int(3*len(mns_F)/5):], 0.8, 2, 80)

    #skin inputs
    exconnectcells(C1, CV1_1, 0.8, 1, 50)
    exconnectcells(C2, CV2_1, 0.8, 1, 50)
    exconnectcells(C3, CV3_1, 0.8, 1, 50)
    exconnectcells(C4, CV4_1, 0.8, 1, 50)
    exconnectcells(C5, CV5_1, 0.8, 1, 50)

    #C1
    exconnectcells(CV1_1, OM1_0E, 0.0005, 1, 50)

    #C2
    exconnectcells(CV2_1, OM1_0E, 0.0005, 1, 30)
    exconnectcells(CV2_1, OM2_0E, 0.00045, 1, 27)

    #C3
    exconnectcells(CV3_1, OM2_0E, 0.00045, 1, 27)
    exconnectcells(CV3_1, OM3_0 , 0.00035, 1, 27)

    #C4
    exconnectcells(CV4_1, OM3_0 , 0.00037, 1, 27)
    exconnectcells(CV4_1, OM4_0E, 0.00041, 1, 27)

    #C5
    exconnectcells(CV5_1, OM5_0 , 0.00036, 1, 30)
    exconnectcells(CV5_1, OM4_0E, 0.00036, 1, 30)
    
    #C=1 Extensor
    inhconnectcells(C_1, OM1_0F, 0.8, 1, 80)
    inhconnectcells(C_1, OM2_0F, 0.8, 1, 80)
    inhconnectcells(C_1, OM4_0F, 0.8, 1, 80)
    inhconnectcells(C_1, OM3_2F, 0.8, 1, 80)

    inhconnectcells(C_1, IP1_F, 0.8, 1, 60)
    inhconnectcells(C_1, IP2_F, 0.8, 1, 60)
    inhconnectcells(C_1, IP3_F, 0.8, 1, 60)
    inhconnectcells(C_1, IP4_F, 0.8, 1, 60)
    inhconnectcells(C_1, IP5_F, 0.8, 1, 60)

    inhconnectcells(C_1, Ia_aff_F, 0.9, 1, 80)

    #C=0 Flexor
    inhconnectcells(C_0, IP1_E, 0.8, 1, 60)
    inhconnectcells(C_0, IP2_E, 0.8, 1, 60)
    inhconnectcells(C_0, IP3_E, 0.8, 1, 60)
    inhconnectcells(C_0, IP4_E, 0.8, 1, 60)
    inhconnectcells(C_0, IP5_E, 0.8, 1, 60)

    inhconnectcells(C_0, Ia_aff_E, 0.8, 1, 80)

    #reflex arc
    exconnectcells(Iagener_E, Ia_aff_E[:int(len(Ia_aff_E)/6)], 0.8, 1, 20)
    exconnectcells(Ia_aff_E, Ia_E, 0.5, 1, 30)
    exconnectcells(Ia_aff_E[:int(len(Ia_aff_E)/6)], Ia_E, 0.8, 1, 30)
    exconnectcells(mns_E, R_E, 0.00025, 1, 30)
    inhconnectcells(Ia_E, mns_F, 0.04, 1, 45)
    inhconnectcells(R_E, mns_E, 0.05, 1, 45)
    inhconnectcells(R_E, Ia_E, 0.01, 1, 40)

    exconnectcells(Ia_aff_F, Ia_F, 0.01, 1, 30)
    exconnectcells(mns_F, R_F, 0.0002, 1, 30)
    inhconnectcells(Ia_F, mns_E, 0.04, 1, 45)
    inhconnectcells(R_F, mns_F, 0.01, 1, 45)
    inhconnectcells(R_F, Ia_F, 0.001, 1, 20)

    inhconnectcells(R_E, R_F, 0.04, 1, 30)
    inhconnectcells(Ia_E, Ia_F, 0.04, 1, 30)
    inhconnectcells(R_F, R_E, 0.04, 1, 30)
    inhconnectcells(Ia_F, Ia_E, 0.04, 1, 30)

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
      cell = bioaff(random.randint(2, 10))
      self.afferents.append(cell)
      while(pc.gid_exists(gid)!=0):
        gid+=1
      gids.append(gid)
      pc.set_gid2node(gid, rank)
      nc = cell.connect2target(None)
      pc.cell(gid, nc)

    self.affgroups.append((gids, name))
    return gids

  def addgener(self, start, interval, nums):
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
    stim.start = random.uniform(start, start + 3)
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
    Creates Ia generators and returns generator gids
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

exnclist = []
inhnclist = []
eesnclist = []
stimnclist = []

def exconnectcells(pre, post, weight, delay, nsyn):
  ''' Connects with excitatory synapses 
    Parameters
    ----------
    pre: list
        list of presynase neurons gids
    post: list
        list of postsynapse neurons gids
    weight: float
        weight of synapse
        used with Gaussian distribution
    delay: int
        synaptic delay
        used with Gaussian distribution
    nsyn: int
        numder of synapses
  '''
  global exnclist
  for i in post:
    if pc.gid_exists(i):
      for j in range(nsyn):
        srcgid = random.randint(pre[0], pre[-1])
        target = pc.gid2cell(i)
        syn = target.synlistex[j]
        nc = pc.gid_connect(srcgid, syn)
        exnclist.append(nc)
        nc.delay = random.gauss(delay, delay/8)
        nc.weight[0] = random.gauss(weight, weight/10)

def inhconnectcells(pre, post, weight, delay, nsyn):
  ''' Connects with inhibitory synapses 
    Parameters
    ----------
    pre: list
        list of presynase neurons gids
    post: list
        list of postsynapse neurons gids
    weight: float
        weight of synapse
        used with Gaussian distribution
    delay: int
        synaptic delay
        used with Gaussian distribution
    nsyn: int
        numder of synapses
  '''
  global inhnclist
  for i in post:
    if pc.gid_exists(i):
      for j in range(nsyn):
        srcgid = random.randint(pre[0], pre[-1]) 
        target = pc.gid2cell(i)
        syn = target.synlistinh[j]
        nc = pc.gid_connect(srcgid, syn)
        inhnclist.append(nc)
        nc.delay = random.gauss(delay, 0.01)
        nc.weight[0] = random.gauss(weight, weight/10)

def genconnect(gen_gid, afferents_gids, weight, delay, nsyn):
  ''' Connects with generator 
    Parameters
    ----------
    afferents_gids: list
        list of presynase neurons gids
    gen_gid: int
        generator gid
    weight: float
        weight of synapse
        used with Gaussian distribution
    delay: int
        synaptic delay
        used with Gaussian distribution
    nsyn: int
        numder of synapses
  '''
  global stimnclist
  for i in afferents_gids:
    if pc.gid_exists(i):
      for j in range(nsyn): 
        target = pc.gid2cell(i)
        syn = target.synlistees[j]
        nc = pc.gid_connect(gen_gid, syn)
        stimnclist.append(nc)
        nc.delay = random.gauss(delay, delay/8)
        nc.weight[0] = random.gauss(weight, weight/10)

def inhgenconnect(gen_gid, afferents_gids, weight, delay, nsyn):
  ''' Connects with generator via inhibitory synapses 
    Parameters
    ----------
    gen_gid: int
        generator gid
    afferents_gids: list
        list of presynase neurons gids
    weight: float
        weight of synapse
        used with Gaussian distribution
    delay: int
        synaptic delay
        used with Gaussian distribution
    nsyn: int
        numder of synapses
  '''
  global stimnclist
  for i in afferents_gids:
    if pc.gid_exists(i):
      for j in range(nsyn): 
        target = pc.gid2cell(i)
        syn = target.synlistinh[j]
        nc = pc.gid_connect(gen_gid, syn)
        stimnclist.append(nc)
        nc.delay = random.gauss(delay, delay/10)
        nc.weight[0] = random.gauss(weight, weight/10)

def createmotif(om0, om1, om2, om3):
  ''' Connects motif module
    see https://github.com/research-team/memristive-spinal-cord/blob/master/doc/diagram/cpg_generator_FE_paper.png
    Parameters
    ----------
    om0: list
        list of OM0 pool gids
    om1: list
        list of OM1 pool gids
    om2: list
        list of OM2 pool gids
    om3: list
        list of OM3 pool gids
  '''
  exconnectcells(om0, om1, 0.05, 1, 27)
  exconnectcells(om1, om2, 0.05, 3, 27)
  exconnectcells(om2, om1, 0.05, 3, 27)
  exconnectcells(om2, om3, 0.001, 1, 27)
  #exconnectcells(g1, g3, 0.001, 1, 27)
  #inhconnectcells(g3, g1, 0.08, 1, 27)
  inhconnectcells(om3, om2, 0.8, 1, 27)

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
  tstop = 150 #(6*speed + 125)*step_number
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

    for i in range(versions):
      cpg_ex = CPG(speed, EES_i, 100, step_number)
            
      motorecorders = []
      for group in cpg_ex.motogroups:
        motorecorders.append(spike_record(group[k_nrns], i))
      
      affrecorders = []
      for group in cpg_ex.affgroups:
        affrecorders.append(spike_record(group[k_nrns], i))
      
      recorders = []
      for group in cpg_ex.groups:
        recorders.append(spike_record(group[k_nrns], i))
       
      print("- "*10, "\nstart")
      prun(speed, step_number)
      print("- "*10, "\nend")
      
      for group, recorder in zip(cpg_ex.affgroups, affrecorders):
        spikeout(group[k_nrns], group[k_name], i, recorder)

      for group, recorder in zip(cpg_ex.groups, recorders):
        spikeout(group[k_nrns], group[k_name], i, recorder)
      
      for group, recorder in zip(cpg_ex.motogroups, motorecorders):
        spikeout(group[k_nrns], group[k_name], i, recorder)
      
    #if (nhost > 1):
    finish()