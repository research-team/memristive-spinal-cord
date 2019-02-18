import logging
logging.basicConfig(level=logging.DEBUG)
from neuron import h
h.load_file('nrngui.hoc')
pc = h.ParallelContext()
rank = int(pc.id())
nhost = int(pc.nhost())

from interneuron import interneuron
from motoneuron import motoneuron
from afferent import afferent
import random


# network creation

class cpg:
  def __init__(self, speed, EES_int, inh_p, version = 0, N = 20):

    self.interneurons = []
    self.motoneurons = []
    self.afferents = []
    self.stims = []
    self.ncell = N

    nMN = 168
    nAff = 120
    nInt = 196

    D1_1E = self.addpool(self.ncell)
    D1_2E = self.addpool(self.ncell)
    D1_3E = self.addpool(self.ncell)
    D1_4E = self.addpool(self.ncell)
    D1_1F = self.addpool(self.ncell)
    D1_2F = self.addpool(self.ncell)
    D1_3F = self.addpool(self.ncell)
    D1_4F = self.addpool(self.ncell)
  
    D2_1E = self.addpool(self.ncell)
    D2_2E = self.addpool(self.ncell)
    D2_3E = self.addpool(self.ncell)
    D2_4E = self.addpool(self.ncell)
    D2_1F = self.addpool(self.ncell)
    D2_2F = self.addpool(self.ncell)
    D2_3F = self.addpool(self.ncell)
    D2_4F = self.addpool(self.ncell)

    D3_1 = self.addpool(self.ncell)
    D3_2 = self.addpool(self.ncell)
    D3_3 = self.addpool(self.ncell)
    D3_4 = self.addpool(self.ncell)

    D4_1E = self.addpool(self.ncell)
    D4_2E = self.addpool(self.ncell)
    D4_3E = self.addpool(self.ncell)
    D4_4E = self.addpool(self.ncell)
    D4_1F = self.addpool(self.ncell)
    D4_2F = self.addpool(self.ncell)
    D4_3F = self.addpool(self.ncell)
    D4_4F = self.addpool(self.ncell)

    D5_1 = self.addpool(self.ncell)
    D5_2 = self.addpool(self.ncell)
    D5_3 = self.addpool(self.ncell)
    D5_4 = self.addpool(self.ncell)

    G1_1 = self.addpool(self.ncell)
    G1_2 = self.addpool(self.ncell)
    G1_3 = self.addpool(self.ncell)

    G2_1E = self.addpool(self.ncell)
    G2_2E = self.addpool(self.ncell)
    G2_3E = self.addpool(self.ncell)
    G2_1F = self.addpool(self.ncell)
    G2_2F = self.addpool(self.ncell)
    G2_3F = self.addpool(self.ncell)

    G3_1E = self.addpool(self.ncell)
    G3_2E = self.addpool(self.ncell)
    G3_3E = self.addpool(self.ncell)
    G3_1F = self.addpool(self.ncell)
    G3_2F = self.addpool(self.ncell)
    G3_3F = self.addpool(self.ncell)

    G4_1 = self.addpool(self.ncell)
    G4_2 = self.addpool(self.ncell)
    G4_3 = self.addpool(self.ncell)

    G5_1E = self.addpool(self.ncell)
    G5_2E = self.addpool(self.ncell)
    G5_3E = self.addpool(self.ncell)
    G5_1F = self.addpool(self.ncell)
    G5_2F = self.addpool(self.ncell)
    G5_3F = self.addpool(self.ncell)

    E1_E = self.addpool(self.ncell)
    E2_E = self.addpool(self.ncell)
    E3_E = self.addpool(self.ncell)
    E4_E = self.addpool(self.ncell)

    E1_F = self.addpool(self.ncell)
    E2_F = self.addpool(self.ncell)
    E3_F = self.addpool(self.ncell)
    E4_F = self.addpool(self.ncell)

    I3_E = self.addpool(self.ncell)
    I4_E = self.addpool(self.ncell)
    I5_E = self.addpool(self.ncell)

    I5_F = self.addpool(self.ncell)
        
    sens_aff = self.addafferents(nAff)
    Ia_aff_E = self.addafferents(nAff)
    Ia_aff_F = self.addafferents(nAff)

    mns_E = self.addmotoneurons(nMN)
    mns_F = self.addmotoneurons(nMN)

    IP_E = self.addpool(nInt)
    IP_F = self.addpool(nInt)

    ees = self.addgener(1, EES_int, 10000)

    c_int = 5

    C1 = self.addgener(speed*0, c_int, speed/c_int)
    C2 = self.addgener(speed*0, c_int, 2*speed/c_int)
    C3 = self.addgener(speed*1, c_int, 2*speed/c_int)
    C4 = self.addgener(speed*3, c_int, 2*speed/c_int)
    C5 = self.addgener(speed*4, c_int, 2*speed/c_int)

    C_1 = self.addgener(speed*0, c_int, 6*speed/c_int)
    C_0 = self.addgener(speed*6, c_int, 125/c_int)
  
    #delays
    connectdelay_extensor(D1_1E, D1_2E, D1_3E, D1_4E)
    connectdelay_flexor(D1_1F, D1_2F, D1_3F, D1_4F)

    connectdelay_extensor(D2_1E, D2_2E, D2_3E, D2_4E)
    connectdelay_flexor(D2_1F, D2_2F, D2_3F, D2_4F)

    connectdelay_extensor(D3_1, D3_2, D3_3, D3_4)

    connectdelay_extensor(D4_1E, D4_2E, D4_3E, D4_4E)
    connectdelay_flexor(D4_1F, D4_2F, D4_3F, D4_4F)

    connectdelay_extensor(D5_1, D5_2, D5_3, D5_4)

    #generators
    connectgenerator(G1_1, G1_2, G1_3)
    
    connectgenerator(G2_1E, G2_2E, G2_3E)
    connectgenerator(G2_1F, G2_2F, G2_3F)

    connectgenerator(G3_1E, G3_2E, G3_3E)
    connectgenerator(G3_1F, G3_2F, G3_3F)

    connectgenerator(G4_1, G4_2, G4_3)

    connectgenerator(G5_1E, G5_2E, G5_3E)
    connectgenerator(G5_1F, G5_2F, G5_3F)

    #between delays (FLEXOR)
    exconnectcells(D2_3F, D3_1, 0.00015, 1, 27)
    exconnectcells(D2_3F, D3_4, 0.00025, 1, 27)
    exconnectcells(D4_3F, D5_1, 0.0002, 1, 27)
    exconnectcells(D4_3F, D5_4, 0.00025, 1, 27)

    #between delays via excitatory pools
    #extensor
    exconnectcells(D1_3E, E1_E, 0.5, 2, 27)
    exconnectcells(E1_E, E2_E, 0.5, 2, 27)
    exconnectcells(E2_E, E3_E, 0.5, 2, 27)
    exconnectcells(E3_E, E4_E, 0.5, 2, 27)

    connectexpools_extensor(D2_1E, D2_4E, E1_E)
    connectexpools_extensor(D3_1, D3_4, E2_E)
    connectexpools_extensor(D4_1E, D4_4E, E3_E)
    connectexpools_extensor(D5_1, D5_4, E4_E)

    #flexor
    exconnectcells(D1_3F, E1_F, 0.5, 2, 27)
    exconnectcells(E1_F, E2_F, 0.5, 2, 27)
    exconnectcells(E2_F, E3_F, 0.5, 2, 27)
    exconnectcells(E3_F, E4_F, 0.5, 2, 27)

    connectexpools_flexor(D2_1F, D2_4F, E1_F)
    connectexpools_flexor(D3_1, D3_4, E2_F)
    connectexpools_flexor(D4_1F, D4_4F, E3_F)
    connectexpools_flexor(D5_1, D5_4, E4_F)

    #delay -> generator
    #extensor
    exconnectcells(D1_3E, G1_1, 0.05, 2, 27)
    exconnectcells(D2_3E, G2_1E, 0.05, 2, 27)
    exconnectcells(D3_3, G3_1E, 0.05, 2, 27)
    exconnectcells(D4_3E, G4_1, 0.05, 2, 27)
    exconnectcells(D5_3, G5_1E, 0.05, 2, 27)

    #flexor
    exconnectcells(D1_3F, G1_1, 0.005, 2, 27)
    exconnectcells(D1_3F, G2_1F, 0.005, 2, 27)
    exconnectcells(D3_3, G3_1F, 0.005, 2, 27)
    exconnectcells(D5_3, G5_1F, 0.005, 2, 27)

    #generator -> delay(FLEXOR)
    exconnectcells(G2_1F, D2_1F, 0.0002, 1, 27)
    exconnectcells(G2_1F, D2_4F, 0.0001, 1, 27)
    exconnectcells(G4_1, D4_1F, 0.0003, 1, 27)
    exconnectcells(G4_1, D4_4F, 0.00025, 1, 27)

    #between generators (FLEXOR)
    exconnectcells(G3_1F, G4_1, 0.05, 2, 27)
    exconnectcells(G3_2F, G4_1, 0.05, 2, 27)

    #inhibitory projections
    #extensor
    inhconnectcells(I3_E, G1_2, 0.8, 1, 27)
    inhconnectcells(I3_E, G1_1, 0.8, 1, 27)

    inhconnectcells(I4_E, G2_1E, 0.8, 1, 27)
    inhconnectcells(I4_E, G2_2E, 0.8, 1, 27)

    inhconnectcells(I5_E, G1_1, 0.8, 1, 27)
    inhconnectcells(I5_E, G1_2, 0.8, 1, 27)
    inhconnectcells(I5_E, G2_1E, 0.8, 1, 27)
    inhconnectcells(I5_E, G2_2E, 0.8, 1, 27)
    inhconnectcells(I5_E, G3_1E, 0.8, 1, 27)
    inhconnectcells(I5_E, G3_2E, 0.8, 1, 27)
    inhconnectcells(I5_E, G4_1, 0.8, 1, 27)
    inhconnectcells(I5_E, G4_2, 0.8, 1, 27)

    #flexor
    inhconnectcells(I5_F, G1_1, 0.8, 1, 27)
    inhconnectcells(I5_F, G1_2, 0.8, 1, 27)
    inhconnectcells(I5_F, G2_1F, 0.8, 1, 27)
    inhconnectcells(I5_F, G2_2F, 0.8, 1, 27)
    inhconnectcells(I5_F, G3_1F, 0.8, 1, 27)
    inhconnectcells(I5_F, G3_2F, 0.8, 1, 27)
    inhconnectcells(I5_F, G4_1, 0.8, 1, 27)
    inhconnectcells(I5_F, G4_2, 0.8, 1, 27)
    
    #EES
    genconnect(sens_aff, ees, 1, 0, 50)
    genconnect(Ia_aff_E, ees, 1, 0, 50)
    genconnect(Ia_aff_F, ees, 1, 0, 50)

    exconnectcells(sens_aff, D1_1E, 0.5, 1, 50)
    exconnectcells(sens_aff, D1_4E, 0.5, 1, 50)

    exconnectcells(sens_aff, D1_1F, 0.5, 1, 50)
    exconnectcells(sens_aff, D1_4F, 0.5, 1, 50)

    exconnectcells(Ia_aff_E, mns_E, 0.5, 1, 50)
    exconnectcells(Ia_aff_F, mns_F, 0.5, 1, 50)

    #skin inputs
    #C1
    genconnect(D1_1E, C1, 0.0001, 1, 50)
    genconnect(D1_4E, C1, 0.0001, 1, 50)

    #C2
    genconnect(D1_1E, C2, 0.00035, 1, 27)
    genconnect(D1_4E, C2, 0.00035, 1, 27)
    genconnect(D2_1E, C2, 0.00035, 1, 27)
    genconnect(D2_4E, C2, 0.00035, 1, 27)

    #C3
    genconnect(D2_1E, C3, 0.0004, 1, 27)
    genconnect(D2_4E, C3, 0.0004, 1, 27)
    genconnect(D3_1, C3, 0.0004, 1, 27)
    genconnect(D3_2, C3, 0.0004, 1, 27)
    genconnect(I3_E, C3, 1, 1, 50)

    #C4
    genconnect(D3_1, C4, 0.0002, 1, 60)
    genconnect(D3_4, C4, 0.0002, 1, 60)
    genconnect(D4_1E, C4, 0.0002, 1, 60)
    genconnect(D4_4E, C4, 0.0002, 1, 60)
    genconnect(I4_E, C4, 1, 1, 50)

    #C5
    genconnect(D5_1, C5, 0.00028, 1, 30)
    genconnect(D5_4, C5, 0.00028, 1, 30)
    genconnect(D4_1E, C5, 0.00028, 1, 30)
    genconnect(D4_4E, C5, 0.00028, 1, 30)
    genconnect(I4_E, C5, 1, 1, 50)

    #C=1 Extensor
    inhconnectcells(C_1, D1_1F, 0.8, 1, 50)
    inhconnectcells(C_1, D1_2F, 0.8, 1, 50)
    inhconnectcells(C_1, D1_3F, 0.8, 1, 50)
    inhconnectcells(C_1, D1_4F, 0.8, 1, 50)

    inhconnectcells(C_1, D2_1F, 0.8, 1, 50)
    inhconnectcells(C_1, D2_2F, 0.8, 1, 50)
    inhconnectcells(C_1, D2_3F, 0.8, 1, 50)
    inhconnectcells(C_1, D2_4F, 0.8, 1, 50)

    inhconnectcells(C_1, D4_1F, 0.8, 1, 50)
    inhconnectcells(C_1, D4_2F, 0.8, 1, 50)
    inhconnectcells(C_1, D4_3F, 0.8, 1, 50)
    inhconnectcells(C_1, D4_4F, 0.8, 1, 50)

    inhconnectcells(C_1, G2_1F, 0.8, 1, 50)
    inhconnectcells(C_1, G2_2F, 0.8, 1, 50)
    inhconnectcells(C_1, G2_3F, 0.8, 1, 50)

    inhconnectcells(C_1, G3_1F, 0.8, 1, 50)
    inhconnectcells(C_1, G3_2F, 0.8, 1, 50)
    inhconnectcells(C_1, G3_3F, 0.8, 1, 50)

    inhconnectcells(C_1, G5_1F, 0.8, 1, 50)
    inhconnectcells(C_1, G5_2F, 0.8, 1, 50)
    inhconnectcells(C_1, G5_3F, 0.8, 1, 50)

    inhconnectcells(C_1, IP_F, 0.8, 1, 50)

    #C=0 Flexor
    inhconnectcells(C_0, D1_1E, 0.8, 1, 50)
    inhconnectcells(C_0, D1_2E, 0.8, 1, 50)
    inhconnectcells(C_0, D1_3E, 0.8, 1, 50)
    inhconnectcells(C_0, D1_4E, 0.8, 1, 50)

    inhconnectcells(C_0, D2_1E, 0.8, 1, 50)
    inhconnectcells(C_0, D2_2E, 0.8, 1, 50)
    inhconnectcells(C_0, D2_3E, 0.8, 1, 50)
    inhconnectcells(C_0, D2_4E, 0.8, 1, 50)

    inhconnectcells(C_0, D4_1E, 0.8, 1, 50)
    inhconnectcells(C_0, D4_2E, 0.8, 1, 50)
    inhconnectcells(C_0, D4_3E, 0.8, 1, 50)
    inhconnectcells(C_0, D4_4E, 0.8, 1, 50)

    inhconnectcells(C_0, G2_1E, 0.8, 1, 50)
    inhconnectcells(C_0, G2_2E, 0.8, 1, 50)
    inhconnectcells(C_0, G2_3E, 0.8, 1, 50)

    inhconnectcells(C_0, G3_1E, 0.8, 1, 50)
    inhconnectcells(C_0, G3_2E, 0.8, 1, 50)
    inhconnectcells(C_0, G3_3E, 0.8, 1, 50)

    inhconnectcells(C_0, G5_1E, 0.8, 1, 50)
    inhconnectcells(C_0, G5_2E, 0.8, 1, 50)
    inhconnectcells(C_0, G5_3E, 0.8, 1, 50)

    inhconnectcells(C_1, IP_F, 0.8, 1, 50)

  def addpool(self, num):
    gids = []
    gid = 0
    for i in range(rank, num, nhost):
      cell = interneuron()
      self.interneurons.append(cell)
      while(pc.gid_exists(gid)!=0):
        gid+=1
      gids.append(gid)
      pc.set_gid2node(gid, rank)
      nc = cell.connect2target(None)
      pc.cell(gid, nc)
    return gids

  def addmotoneurons(self, num):
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
    return gids

  def addafferents(self, num):
    gids = []
    gid = 0
    for i in range(rank, num, nhost):
      cell = afferent()
      self.afferents.append(cell)
      while(pc.gid_exists(gid)!=0):
        gid+=1
      gids.append(gid)
      pc.set_gid2node(gid, rank)
      nc = cell.connect2target(None)
      pc.cell(gid, nc)
    return gids

  def addgener(self, start, interval, nums):
    gid = 0
    stim = h.NetStim()
    stim.number = nums
    stim.start = start
    stim.interval = interval
    #skinstim.noise = 0.1
    self.stims.append(stim)
    while(pc.gid_exists(gid)!=0):
        gid+=1
    pc.set_gid2node(gid, rank)
    ncstim = h.NetCon(stim, None)
    pc.cell(gid, ncstim)
    return gid


exnclist = []
inhnclist = []
eesnclist = []
stimnclist = []

def exconnectcells(pre, post, weight, delay, nsyn):
  ''' connection with excitatory synapse '''
  global exnclist
  # not efficient but demonstrates use of pc.gid_exists
  for i in post:
    if pc.gid_exists(i):
      for j in range(nsyn):
        srcgid = random.randint(pre[0], pre[-1])
        target = pc.gid2cell(i)
        syn = target.synlistex[j]
        nc = pc.gid_connect(srcgid, syn)
        exnclist.append(nc)
        nc.delay = random.gauss(delay, delay/15)
        nc.weight[0] = random.gauss(weight, weight/10)

def inhconnectcells(pre, post, weight, delay, nsyn):
  ''' connection with inhibitory synapse '''
  global inhnclist
  # not efficient but demonstrates use of pc.gid_exists
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

def genconnect(afferents_gids, gen_gid, weight, delay, nsyn):
  ''' stimulate afferents with NetStim '''
  global stimnclist
  for i in afferents_gids:
    if pc.gid_exists(i):
      for j in range(nsyn):        
        target = pc.gid2cell(i)
        syn = target.synlistees[j]
        nc = pc.gid_connect(gen_gid, syn)
        stimnclist.append(nc)
        nc.delay = random.gauss(delay, delay/10)
        nc.weight[0] = 1 #random.gauss(weight, weight/10)


def connectdelay_extensor(d1, d2, d3, d4):
  exconnectcells(d2, d1, 0.05, 3, 27)
  exconnectcells(d1, d2, 0.05, 3, 27)
  exconnectcells(d1, d3, 0.01, 1, 27)
  exconnectcells(d2, d3, 0.01, 1, 27)
  inhconnectcells(d4, d3, 0.001, 1, 27)
  inhconnectcells(d3, d2, 0.08, 1, 27)
  inhconnectcells(d3, d1, 0.08, 1, 27)

def connectdelay_flexor(d1, d2, d3, d4):
  exconnectcells(d2, d1, 0.05, 3, 27)
  exconnectcells(d1, d2, 0.05, 3, 27)
  exconnectcells(d2, d3, 0.01, 1, 27)
  exconnectcells(d1, d3, 0.01, 1, 27)
  inhconnectcells(d4, d3, 0.18, 1, 27)
  inhconnectcells(d3, d2, 0.1, 1, 27)
  inhconnectcells(d3, d1, 0.08, 1, 27)

def connectgenerator(g1, g2, g3):
  exconnectcells(g1, g2, 0.05, 3, 27)
  exconnectcells(g2, g1, 0.05, 3, 27)
  exconnectcells(g2, g3, 0.005, 1, 27)
  exconnectcells(g1, g3, 0.005, 1, 27)
  inhconnectcells(g3, g1, 0.08, 1, 27)
  inhconnectcells(g3, g2, 0.08, 1, 27)

def connectexpools_extensor(d1, d4, ep):
  exconnectcells(ep, d1, 0.00037, 1, 27)
  exconnectcells(ep, d4, 0.00037, 1, 27)

def connectexpools_flexor(d1, d4, ep):
  exconnectcells(ep, d1, 0.0002, 1, 27)
  exconnectcells(ep, d4, 0.0002, 1, 27)
  

cpg = cpg(25, 25, 100)
