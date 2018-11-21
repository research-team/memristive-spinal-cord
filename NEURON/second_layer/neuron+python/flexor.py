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

interneurons = []
motoneurons = []
afferents = []
exnclist = []
inhnclist = []
eesnclist = []
stimnclist = []
skinstims = []
soma_v_vec = []
moto_v_vec = []
ncIalist = []

speed = 25
version = 0
ncell = 20       
nMN = 168      
nAff = 200
nIP = 480
ncells = ncell*39+nIP+nMN+2*nAff

def addnetwork():
  ''' create cells and connections '''
  addinterneurons(0, ncell*39+nIP)
  addmotoneurons(ncell*39+nIP, ncell*39+nIP+nMN)
  addafferents(ncell*39+nIP+nMN, ncell*39+nIP+nMN+2*nAff+5*ncell)
  addees()
  connectcells()

def addinterneurons(start, end):
  global interneurons, rank, nhost
  for i in range(rank+start, end, nhost):
    cell = interneuron()
    interneurons.append(cell)
    pc.set_gid2node(i, rank)
    nc = cell.connect2target(None)
    pc.cell(i, nc)

def addmotoneurons(start, end):
  global motoneurons, rank, nhost
  for i in range(rank+start, end, nhost):
    cell = motoneuron()
    motoneurons.append(cell)
    pc.set_gid2node(i, rank)
    nc = cell.connect2target(None)
    pc.cell(i, nc)

def addafferents(start, end):
  global afferents, rank, nhost
  for i in range(rank+start, end, nhost):
    cell = afferent()
    afferents.append(cell)
    pc.set_gid2node(i, rank)
    nc = cell.connect2target(None)
    pc.cell(i, nc)


def connectcells():
  ''' connection between cells '''
  # delay
  exconnectcells(0, ncell, 0.05, 0.1, ncell*39+nIP+nMN, ncell*39+nIP+nMN+nAff, 27)
  exconnectcells(ncell*10, ncell*11, 0.05, 0.1, ncell*39+nIP+nMN, ncell*39+nIP+nMN+nAff, 27)
  
  for i in range(5):
    exconnectcells(ncell*i, ncell*(i+1), 0.05, 3, ncell*(i+5), ncell*(i+6), 27)
    exconnectcells(ncell*(i+5), ncell*(i+6), 0.05, 3, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+15), ncell*(i+16), 0.01, 1, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+15), ncell*(i+16), 0.01, 1, ncell*(i+5), ncell*(i+6), 27)
    inhconnectcells(ncell*(i+15), ncell*(i+16), 0.18, 1, ncell*(i+10), ncell*(i+11), 27)
    inhconnectcells(ncell*(i+5), ncell*(i+6), 0.08, 1, ncell*(i+15), ncell*(i+16), 27)
    inhconnectcells(ncell*i, ncell*(i+1), 0.08, 1, ncell*(i+15), ncell*(i+16), 27)

    inhconnectcells(ncell*(15), ncell*(16), 0.05, 1, ncell*(10), ncell*(11), 27)


  #inhconnectcells(ncell*19, ncell*20, 0.1, 1, ncell*14, ncell*15, 32)
  
  # between delays
  exconnectcells(ncell*35, ncell*36, 0.05, 1, ncell*15, ncell*16, 27)
  exconnectcells(ncell*2, ncell*3, 0.00015, 1, ncell*16, ncell*17, 27)
  exconnectcells(ncell*12, ncell*13, 0.00025, 1, ncell*16, ncell*17, 27)
  exconnectcells(ncell*4, ncell*5, 0.0002, 1, ncell*18, ncell*19, 27)
  exconnectcells(ncell*14, ncell*15, 0.00025, 1, ncell*18, ncell*19, 27)

  for i in range(35, 38):
    exconnectcells(ncell*(i+1), ncell*(i+2), 0.05, 2, ncell*i, ncell*(i+1), 27)

  for i in range(1, 5): 
    exconnectcells(ncell*i, ncell*(i+1), 0.0002, 1, ncell*(i+34), ncell*(i+35), 27)
    exconnectcells(ncell*(i+10), ncell*(i+11), 0.0002, 1, ncell*(i+34), ncell*(i+35), 27)
  
  # generators
  for i in range(20, 25):
    exconnectcells(ncell*i, ncell*(i+1), 0.05, 3, ncell*(i+5), ncell*(i+6), 27)
    exconnectcells(ncell*(i+5), ncell*(i+6), 0.05, 3, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+10), ncell*(i+11), 0.05, 1, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+10), ncell*(i+11), 0.05, 1, ncell*(i+5), ncell*(i+6), 27)
    inhconnectcells(ncell*(i+5), ncell*(i+6), 0.04, 1, ncell*(i+10), ncell*(i+11), 27)
    inhconnectcells(ncell*i, ncell*(i+1), 0.04, 1, ncell*(i+10), ncell*(i+11), 27)

  exconnectcells(ncell*23, ncell*24, 0.05, 1, ncell*22, ncell*23, 27)

  # delay -> generator
  exconnectcells(ncell*20, ncell*21, 0.005, 1, ncell*15, ncell*16, 27)
  exconnectcells(ncell*21, ncell*22, 0.005, 1, ncell*15, ncell*16, 27)
  exconnectcells(ncell*22, ncell*23, 0.005, 1, ncell*17, ncell*18, 27)
  exconnectcells(ncell*24, ncell*25, 0.005, 1, ncell*19, ncell*20, 27)

  # generator -> delay
  exconnectcells(ncell, ncell*2, 0.0002, 1, ncell*21, ncell*22, 27)
  exconnectcells(ncell*11, ncell*12, 0.0001, 1, ncell*21, ncell*22, 27)
  exconnectcells(ncell*3, ncell*4, 0.0003, 1, ncell*23, ncell*24, 27)
  exconnectcells(ncell*13, ncell*14, 0.00025, 1, ncell*23, ncell*24, 27)

  # inhibitory projections
  for i in range(20, 23):
    #inhconnectcells(ncell*i, ncell*(i+1), 0.8, 1, ncell*4, ncell*5, 27)
    #inhconnectcells(ncell*(i+5), ncell*(i+6), 0.8, 1, ncell*4, ncell*5, 27)
    inhconnectcells(ncell*i, ncell*(i+1), 0.8, 1, ncell*29, ncell*30, 27)
    inhconnectcells(ncell*(i+5), ncell*(i+6), 0.8, 1, ncell*29, ncell*30, 27)
    inhconnectcells(ncell*i, ncell*(i+1), 0.8, 1, ncell*24, ncell*25, 27)
    inhconnectcells(ncell*(i+5), ncell*(i+6), 0.8, 1, ncell*24, ncell*25, 27)

  inhconnectcells(ncell*20, ncell*21, 0.8, 1, ncell*2, ncell*3, 50)
  inhconnectcells(ncell*25, ncell*26, 0.8, 1, ncell*2, ncell*3, 50)
  
  # ip connections
  for i in range(0, 5):
    exconnectcells(ncell*39+int(nIP/10)*i, ncell*39+int(nIP/10)*(i+1), 0.1, 1, ncell*(i+20), ncell*(i+21), 28)
    exconnectcells(ncell*39+int(nIP/10)*(i+5), ncell*39+int(nIP/10)*(i+6), 0.1, 1, ncell*(i+25), ncell*(i+26), 28)

  # mn connections 
  for i in range(0, 12):
    exconnectcells(ncell*39+nIP, ncell*39+nIP+int(nMN*0.65), 0.05, 1, ncell*39+int(nIP/12)*i, ncell*39+int(nIP/12)*(i+1), 15)

  exconnectcells(ncell*39+nIP, ncell*39+nIP+nMN, 0.1, 1, ncell*39+nIP+nMN+nAff, ncell*39+nIP+nMN+2*nAff, 50)

def exconnectcells(start, end, weight, delay, srcstart, srcend, nsyn):
  ''' connection with excitatory synapse '''
  global exnclist
  # not efficient but demonstrates use of pc.gid_exists
  for i in range(start, end):
    if pc.gid_exists(i):
      for j in range(nsyn):
        srcgid = random.randint(srcstart, srcend-1)
        target = pc.gid2cell(i)
        syn = target.synlistex[j]
        nc = pc.gid_connect(srcgid, syn)
        exnclist.append(nc)
        nc.delay = random.gauss(delay, delay/15)
        nc.weight[0] = random.gauss(weight, weight/10)

def stimconnectcells(start, end, weight, delay, srcstart, srcend, nsyn):
  ''' connection with excitatory synapse '''
  global stimnclist
  # not efficient but demonstrates use of pc.gid_exists
  for i in range(start, end):
    if pc.gid_exists(i):
      for j in range(nsyn):
        srcgid = random.randint(srcstart, srcend)
        target = pc.gid2cell(i)
        syn = target.synlistees[j]
        nc = pc.gid_connect(srcgid, syn)
        stimnclist.append(nc)
        nc.delay = random.gauss(delay, delay/15)
        nc.weight[0] = random.gauss(weight, weight/10)

def inhconnectcells(start, end, weight, delay, srcstart, srcend, nsyn):
  ''' connection with inhibitory synapse '''
  global inhnclist
  # not efficient but demonstrates use of pc.gid_exists
  for i in range(start, end):
    if pc.gid_exists(i):
      for j in range(nsyn):
        srcgid = random.randint(srcstart, srcend-1) 
        target = pc.gid2cell(i)
        syn = target.synlistinh[j]
        nc = pc.gid_connect(srcgid, syn)
        inhnclist.append(nc)
        nc.delay = random.gauss(delay, 0.01)
        nc.weight[0] = random.gauss(weight, weight/10)

def addees():
  ''' stimulate afferents with NetStim '''
  global stim, ncstim, eesnclist
  stim = h.NetStim()
  stim.number = 100000000
  stim.start = 0
  stim.interval = 25
  for i in range(ncell*39+nIP+nMN, ncell*39+nIP+nMN+2*nAff):
    if pc.gid_exists(i):
      for j in range(50):
        ncstim = h.NetCon(stim, pc.gid2cell(i).synlistees[j])
        eesnclist.append(ncstim)
        ncstim.delay = 0
        ncstim.weight[0] = 1

addnetwork()

# run and recording

def spike_record():
  ''' record spikes from all gids '''
  global moto_v_vec, motoneurons, soma_v_vec, interneurons 
  for i in range(len(interneurons)):
    v_vec = h.Vector()
    v_vec.record(interneurons[i].soma(0.5)._ref_v)
    soma_v_vec.append(v_vec)
  
  for i in range(len(motoneurons)):
    moto_vec = h.Vector()
    moto_vec.record(motoneurons[i].soma(0.5)._ref_vext[0])
    moto_v_vec.append(moto_vec)

def prun(tstop):
  ''' simulation control '''
  pc.set_maxstep(10)
  h.stdinit()
  pc.psolve(tstop)

def spikeout():
  ''' report simulation results to stdout '''
  global rank, moto_v_vec, motoneurons, soma_v_vec, interneurons 
  pc.barrier()
  for i in range(nhost):
    if i == rank:
      for j in range(len(interneurons)):
        path=str('./res/vIn%dr%ds%dv%d'%(j, rank, speed, version))
        f = open(path, 'w')
        for v in list(soma_v_vec[j]):
          f.write(str(v)+"\n")
    pc.barrier()
  
  pc.barrier()
  for i in range(nhost):
    if i == rank:
      for j in range(len(motoneurons)):
        path=str('./res/vMN%dr%ds%dv%d'%(j, rank, speed, version))
        f = open(path, 'w')
        for v in list(moto_v_vec[j]):
          f.write(str(v)+"\n")
    pc.barrier()

def finish():
  ''' proper exit '''
  pc.runworker()
  pc.done()
  h.quit()

if __name__ == '__main__':
  spike_record()
  print("- "*10, "\nstart")
  prun(125)
  print("- "*10, "\nend")
  spikeout()
  if (nhost > 1):
    finish()