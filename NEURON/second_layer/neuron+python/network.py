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

ncell = 20       
nMN = 168      
nAff = 120
nIP = 240
ncells = ncell*35+nIP+nMN+2*nAff+5*ncell

def addnetwork():
  ''' create cells and connections '''
  addinterneurons(0, ncell*35+nIP)
  addmotoneurons(ncell*35+nIP, ncell*35+nIP+nMN)
  addafferents(ncell*35+nIP+nMN, ncell*35+nIP+nMN+2*nAff+5*ncell)
  addees()
  addskininputs()
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
  # delay
  exconnectcells(0, ncell, 1, 1, ncell*35+nIP+nMN, ncell*35+nIP+nMN+nAff, 27)
  exconnectcells(ncell*10, ncell*11, 1, 1, ncell*35+nIP+nMN, ncell*35+nIP+nMN+nAff, 27)
  
  for i in range(5):
    exconnectcells(ncell*i, ncell*(i+1), 0.05, 2, ncell*(i+5), ncell*(i+6), 27)
    exconnectcells(ncell*(i+5), ncell*(i+6), 0.05, 2, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+15), ncell*(i+16), 0.001, 1, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+15), ncell*(i+16), 0.001, 1, ncell*(i+5), ncell*(i+6), 27)
    inhconnectcells(ncell*(i+15), ncell*(i+16), (0.0001+i*0.0002), 1, ncell*(i+10), ncell*(i+11), 27)
    inhconnectcells(ncell*(i+5), ncell*(i+6), 0.5, 1, ncell*(i+15), ncell*(i+16), 27)

  # skin inputs

  #C1
  stimconnectcells(ncell*35+nIP+nMN+2*nAff, ncell*35+nIP+nMN+2*nAff+ncell, 1, 1, ncells, ncells, 50)
  stimconnectcells(0, ncell, 0.0007, 1, ncells, ncells, 50)
  stimconnectcells(ncell*10, ncell*11, 0.0007, 1, ncells, ncells, 50)
  
  #C2
  stimconnectcells(ncell*35+nIP+nMN+2*nAff+ncell, ncell*35+nIP+nMN+2*nAff+2*ncell, 1, 1, ncells+1, ncells+1, 50)
  stimconnectcells(0, ncell, 0.0007, 1, ncells+1, ncells+1, 27)
  stimconnectcells(ncell*10, ncell*11, 0.0007, 1, ncells+1, ncells+1, 27)
  stimconnectcells(ncell, ncell*2, 0.0007, 1, ncells+1, ncells+1, 27)
  stimconnectcells(ncell*11, ncell*12, 0.0007, 1, ncells+1, ncells+1, 27)
  
  #C3
  stimconnectcells(ncell*35+nIP+nMN+2*nAff+2*ncell, ncell*35+nIP+nMN+2*nAff+3*ncell, 1, 1, ncells+2, ncells+2, 50)
  stimconnectcells(ncell, ncell*2, 0.0007, 1, ncells+2, ncells+2, 27)
  stimconnectcells(ncell*11, ncell*12, 0.0007, 1, ncells+2, ncells+2, 27)
  stimconnectcells(ncell*2, ncell*3, 0.0005, 1, ncells+2, ncells+2, 27)
  stimconnectcells(ncell*12, ncell*13, 0.0005, 1, ncells+2, ncells+2, 27)

  #C4
  stimconnectcells(ncell*35+nIP+nMN+2*nAff+3*ncell, ncell*35+nIP+nMN+2*nAff+4*ncell, 1, 1, ncells+3, ncells+3, 50)
  stimconnectcells(ncell*2, ncell*3, 0.0003, 1, ncells+3, ncells+3, 27)
  stimconnectcells(ncell*12, ncell*13, 0.0003, 1, ncells+3, ncells+3, 27)
  stimconnectcells(ncell*3, ncell*4, 0.0003, 1, ncells+3, ncells+3, 27)
  stimconnectcells(ncell*13, ncell*14, 0.0003, 1, ncells+3, ncells+3, 27)

  #C5
  stimconnectcells(ncell*35+nIP+nMN+2*nAff+4*ncell, ncell*35+nIP+nMN+2*nAff+5*ncell, 1, 1, ncells+4, ncells+4, 50)
  stimconnectcells(ncell*4, ncell*5, 0.0003, 1, ncells+4, ncells+4, 27)
  stimconnectcells(ncell*14, ncell*15, 0.0003, 1, ncells+4, ncells+4, 27)
  
  # between delays
  for i in range(1, 5): 
    exconnectcells(ncell*i, ncell*(i+1), 0.0007, 1, ncell*(i+14), ncell*(i+15), 27)
    exconnectcells(ncell*(i+10), ncell*(i+11), 0.0007, 1, ncell*(i+14), ncell*(i+15), 27)
  
  # generators
  for i in range(20, 25):
    exconnectcells(ncell*i, ncell*(i+1), 0.05, 2, ncell*(i+5), ncell*(i+6), 27)
    exconnectcells(ncell*(i+5), ncell*(i+6), 0.05, 2, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+10), ncell*(i+11), 0.05, 1, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+10), ncell*(i+11), 0.05, 1, ncell*(i+5), ncell*(i+6), 27)
    inhconnectcells(ncell*(i+5), ncell*(i+6), 1, 1, ncell*(i+10), ncell*(i+11), 27)

  # delay -> generator
  for i in range(15, 20):
    exconnectcells(ncell*(i+5), ncell*(i+6), 0.05, 2, ncell*(i), ncell*(i+1), 27)


  # inhibitory projections
  for i in range(20, 24):
    inhconnectcells(ncell*i, ncell*(i+1), 1, 1, ncell*35+nIP+nMN+2*nAff+4*ncell, ncell*35+nIP+nMN+2*nAff+5*ncell, 27)
    inhconnectcells(ncell*(i+5), ncell*(i+6), 1, 1, ncell*35+nIP+nMN+2*nAff+4*ncell, ncell*35+nIP+nMN+2*nAff+5*ncell, 27)

  inhconnectcells(ncell*20, ncell*21, 1, 1, ncell*35+nIP+nMN+2*nAff+2*ncell, ncell*35+nIP+nMN+2*nAff+3*ncell, 50)
  inhconnectcells(ncell*25, ncell*26, 1, 1, ncell*35+nIP+nMN+2*nAff+2*ncell, ncell*35+nIP+nMN+2*nAff+3*ncell, 50)
  inhconnectcells(ncell*20, ncell*21, 1, 1, ncell*35+nIP+nMN+2*nAff+3*ncell, ncell*35+nIP+nMN+2*nAff+4*ncell, 50)
  inhconnectcells(ncell*25, ncell*26, 1, 1, ncell*35+nIP+nMN+2*nAff+3*ncell, ncell*35+nIP+nMN+2*nAff+4*ncell, 50)
  inhconnectcells(ncell*21, ncell*22, 1, 1, ncell*35+nIP+nMN+2*nAff+3*ncell, ncell*35+nIP+nMN+2*nAff+4*ncell, 50)
  inhconnectcells(ncell*26, ncell*27, 1, 1, ncell*35+nIP+nMN+2*nAff+3*ncell, ncell*35+nIP+nMN+2*nAff+4*ncell, 50)


  # ip connections
  for i in range(0, 5):
    exconnectcells(ncell*35+int(nIP/10)*i, ncell*35+int(nIP/10)*(i+1), 0.05, 1, ncell*(i+20), ncell*(i+21), 28)
    exconnectcells(ncell*35+int(nIP/10)*(i+5), ncell*35+int(nIP/10)*(i+6), 0.05, 1, ncell*(i+25), ncell*(i+26), 28)

  # mn connections 
  for i in range(0, 10):
    exconnectcells(ncell*35+nIP, ncell*35+nIP+int(nMN/2), 0.05, 1, ncell*35+24*i, ncell*35+24*(i+1), 12)

  exconnectcells(ncell*35+nIP, ncell*35+nIP+nMN, 1, 1, ncell*35+nIP+nMN+nAff, ncell*35+nIP+nMN+2*nAff-1, 50)

def exconnectcells(start, end, weight, delay, srcstart, srcend, nsyn):
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
        nc.delay = random.gauss(delay, 0.01)
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
        nc.delay = random.gauss(delay, 0.01)
        nc.weight[0] = random.gauss(weight, weight/10)

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
        nc.delay = random.gauss(delay, 0.01)
        nc.weight[0] = random.gauss(weight, weight/10)

def addees():
  ''' stimulate afferents with NetStim '''
  global stim, ncstim, eesnclist
  stim = h.NetStim()
  stim.number = 100000000
  stim.start = 1
  stim.interval = 25
  for i in range(ncell*35+nIP+nMN, ncell*35+nIP+nMN+2*nAff):
    if pc.gid_exists(i):
      for j in range(50):
        ncstim = h.NetCon(stim, pc.gid2cell(i).synlistees[j])
        eesnclist.append(ncstim)
        ncstim.delay = 0
        ncstim.weight[0] = 1

def addskininputs():
  global skinstim, skinstims, ncskin, rank
  for n in range(5):
    skinstim = h.NetStim()
    skinstim.number = 5
    if n == 3:
      skinstim.number = 10
    skinstim.start = 1+25*n
    if n == 4:
      skinstim.start = 25*(n+1)
    skinstim.interval = 5
    #skinstim.noise = 0.1
    skinstims.append(skinstim)
    pc.set_gid2node(ncells+n, rank)
    ncskin = h.NetCon(skinstim, None)
    pc.cell(ncells+n, ncskin)

addnetwork()

# run and recording

def spike_record():
  ''' record spikes from all gids '''
  global soma_v_vec, motoneurons
  soma_v_vec = [None] * len(motoneurons)
  for i in range(len(motoneurons)):
    soma_v_vec[i] = h.Vector()
    soma_v_vec[i].record(motoneurons[i].soma(0.5)._ref_vext[0])

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


