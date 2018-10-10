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

index = 0

ncell = 20       
nMN = 168      
nAff = 120
nIP = 120
ncells = ncell*39+nIP+nMN+2*nAff 

def addnetwork():
  ''' create cells and connections '''
  addinterneurons(0, ncell*39+nIP)
  addmotoneurons(ncell*39+nIP, ncell*39+nIP+nMN)
  addafferents(ncell*39+nIP+nMN, ncells+1)
  connectcells()

def addinterneurons(start, end):
  global interneurons, rank, nhost
  interneurons = []
  for i in range(rank+start, end, nhost):
    cell = interneuron()
    interneurons.append(cell)
    pc.set_gid2node(i, rank)
    nc = cell.connect2target(None)
    pc.cell(i, nc)

def addmotoneurons(start, end):
  global motoneurons, rank, nhost
  motoneurons = []
  for i in range(rank+start, end, nhost):
    cell = motoneuron()
    motoneurons.append(cell)
    pc.set_gid2node(i, rank)
    nc = cell.connect2target(None)
    pc.cell(i, nc)

def addafferents(start, end):
  global afferents, rank, nhost
  afferents = []
  for i in range(rank+start, end, nhost):
    cell = afferent()
    afferents.append(cell)
    pc.set_gid2node(i, rank)
    nc = cell.connect2target(None)
    pc.cell(i, nc)


def connectcells():
  ''' connection between cells '''
  # delays
  exconnectcells(0, ncell, 1, 1, ncell*39+nIP+nMN, ncell*39+nIP+nMN+nAff, 27)
  exconnectcells(ncell*9, ncell*10, 1, 1, ncell*39+nIP+nMN, ncell*39+nIP+nMN+nAff, 27)
  
  for i in range(2):
    exconnectcells(ncell*i, ncell*(i+1), 0.05, 2, ncell*(i+3), ncell*(i+4), 27)
    exconnectcells(ncell*(i+5), ncell*(i+6), 0.05, 2, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+10), ncell*(i+11), 0.05, 1, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+10), ncell*(i+11), 0.05, 1, ncell*(i+5), ncell*(i+6), 27)
    inhconnectcells(ncell*i, ncell*(i+1), 0.04, 1, ncell*(i+10), ncell*(i+11), 27)
    inhconnectcells(ncell*(i+5), ncell*(i+6), 0.04, 1, ncell*(i+10), ncell*(i+11), 27)

  # generators
  for i in range(12, 16):
    exconnectcells(ncell*i, ncell*(i+1), 0.05, 2, ncell*(i+5), ncell*(i+6), 27)
    exconnectcells(ncell*(i+5), ncell*(i+6), 0.05, 2, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+10), ncell*(i+11), 0.05, 1, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+10), ncell*(i+11), 0.05, 1, ncell*(i+5), ncell*(i+6), 27)
    inhconnectcells(ncell*i, ncell*(i+1), 0.04, 1, ncell*(i+10), ncell*(i+11), 27)
    inhconnectcells(ncell*(i+5), ncell*(i+6), 0.04, 1, ncell*(i+10), ncell*(i+11), 27)

  # subthreshold
  for i in range(27, 30):
    exconnectcells(ncell*i, ncell*(i+1), 0.05, 2, ncell*(i+4), ncell*(i+5), 27)
    exconnectcells(ncell*(i+4), ncell*(i+5), 0.05, 2, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+8), ncell*(i+9), 0.00025, 1, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+8), ncell*(i+9), 0.00025, 1, ncell*(i+4), ncell*(i+5), 27)
    inhconnectcells(ncell*(i+4), ncell*(i+5), 0.04, 1, ncell*(i+8), ncell*(i+9), 27)

  # between sub
  for i in range(28, 30):
    exconnectcells(ncell*i, ncell*(i+1), 0.05, 1, ncell*(i+7), ncell*(i+8), 27)
    exconnectcells(ncell*(i+8), ncell*(i+9), 0.00025, 1, ncell*(i+7), ncell*(i+8), 27)

  # between layers

  exconnectcells(ncell*27, ncell*28, 0.05, 1, ncell*6, ncell*7, 27)
  exconnectcells(ncell*35, ncell*36, 0.00025, 1, ncell*6, ncell*7, 27)
  exconnectcells(ncell*12, ncell*12, 0.05, 1, ncell*6, ncell*7, 27)

  exconnectcells(ncell*2, ncell*3, 0.05, 1, ncell*36, ncell*37, 27)
  exconnectcells(ncell*11, ncell*12, 0.05, 1, ncell*36, ncell*37, 27)

  exconnectcells(ncell*13, ncell*14, 0.05, 1, ncell*35, ncell*36, 27)
  exconnectcells(ncell*14, ncell*14, 0.05, 1, ncell*36, ncell*37, 27)

  exconnectcells(ncell*15, ncell*16, 0.05, 1, ncell*37, ncell*38, 27)
  exconnectcells(ncell*16, ncell*17, 0.05, 1, ncell*38, ncell*39, 27)

  # inhibitory projections
  for i in range(12, 14):
    inhconnectcells(ncell*i, ncell*(i+1), 0.05, 1, ncell*16, ncell*17, 27)
    inhconnectcells(ncell*(i+5), ncell*(i+6), 0.05, 1, ncell*16, ncell*17, 27)

  inhconnectcells(ncell*12, ncell*13, 0.05, 1, ncell*28, ncell*29, 27)
  inhconnectcells(ncell*12, ncell*13, 0.05, 1, ncell*32, ncell*33, 27)
  inhconnectcells(ncell*12, ncell*13, 0.05, 1, ncell*36, ncell*37, 27)
  inhconnectcells(ncell*13, ncell*14, 0.05, 1, ncell*29, ncell*30, 27)
  inhconnectcells(ncell*13, ncell*14, 0.05, 1, ncell*33, ncell*34, 27)
  inhconnectcells(ncell*13, ncell*14, 0.05, 1, ncell*37, ncell*38, 27)

  # ip connections
  for i in range(0, 4):
    exconnectcells(ncell*39+12*i, ncell*39+12*(i+1), 0.05, 1, ncell*(i+12), ncell*(i+13), 28)
    exconnectcells(ncell*39+12*(i+5), ncell*39+12*(i+6), 0.05, 1, ncell*(i+17), ncell*(i+18), 28)

  # mn connections 
  for i in range(0, 11):
    exconnectcells(ncell*39+nIP, ncell*39+nIP+nMN, 0.05, 1, ncell*39+10*i, ncell*39+10*(i+1), 12)

  exconnectcells(ncell*39+nIP, ncell*39+nIP+nMN, 1, 1, ncell*39+nIP+nMN+nAff, ncell*39+nIP+nMN+2*nAff-1, 50)
  exconnectcells(ncell*39+nIP+nMN, ncell*39+nIP+nMN+2*nAff, 1, 1, ncells, ncells, 50)

def exconnectcells(start, end, weight, delay, srcstart, srcend, nsyn):
  global index
  ''' connection with excitatory synapse '''
  global exnclist
  # not efficient but demonstrates use of pc.gid_exists
  for i in range(start, end):
    if pc.gid_exists(i):
      for j in range(nsyn):
        srcgid = random.randint(srcstart, srcend)
        target = pc.gid2cell(i)
        syn = target.synlistex[j]
        nc = pc.gid_connect(srcgid, syn)
        exnclist.append(nc)
        nc.delay = delay
        nc.weight[0] = weight

def inhconnectcells(start, end, weight, delay, srcstart, srcend, nsyn):
  ''' connection with inhibitory synapse '''
  global inhnclist
  # not efficient but demonstrates use of pc.gid_exists
  for i in range(start, end):
    if pc.gid_exists(i):
      for j in range(nsyn):
        srcgid = random.randint(srcstart, srcend) 
        target = pc.gid2cell(i)
        syn = target.synlistinh[j]
        nc = pc.gid_connect(srcgid, syn)
        inhnclist.append(nc)
        nc.delay = delay
        nc.weight[0] = weight

addnetwork()
#print(len(exnclist))

# run and recording

def addees():
  ''' stimulate afferents with NetStim '''
  global stim, ncstim
  if not pc.gid_exists(0):
    return
  stim = h.NetStim()
  stim.number = 1000000000
  stim.start = 0
  stim.interval = 25
  ncstim = h.NetCon(stim, pc.gid2cell(ncells).synlistees[0])
  ncstim.delay = 0
  ncstim.weight[0] = 1

addees()

def spike_record():
  ''' record spikes from all gids '''
  global soma_v_vec, motoneurons
  soma_v_vec = [None] * len(motoneurons)
  for i in range(len(motoneurons)):
    soma_v_vec[i] = h.Vector()
    soma_v_vec[i].record(motoneurons[i].soma(0.5)._ref_vext[1])

def prun(tstop):
  ''' simulation control '''
  pc.set_maxstep(10)
  h.stdinit()
  pc.psolve(tstop)

def spikeout():
  ''' report simulation results to stdout '''
  global rank, soma_v_vec, motoneurons
  pc.barrier()
  for i in range(nhost):
    if i == rank:
      for j in range(len(motoneurons)):
        path=str('./res/vMN%dr%dv%d'%(j, rank, 0))
        f = open(path, 'w')
        for v in list(soma_v_vec[i]):
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
  prun(150)
  print("- "*10, "\nend")
  spikeout()
  if (nhost > 1):
    finish()


