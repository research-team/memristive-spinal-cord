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
nclist = []

ncell = 20       
nMN = 169      
nAff = 120
nIP = 120
ncells = ncell*39+nIP+nMN+2*nAff 

def addnetwork():
  addinterneurons(0, ncell*39+nIP)
  addmotoneurons(ncell*39+nIP, ncell*39+nIP+nMN)
  addafferents(ncell*39+nIP+nMN, ncell*39+nIP+nMN+2*nAff)
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

# connection between cells

def connectcells():
  
  # delays
  exconnectcells(0, ncell, 0.05, 1, ncell*39+nIP+nMN, ncell*39+nIP+nMN+nAff, 27)
  exconnectcells(ncell*9, ncell*10, 0.05, 1, ncell*39+nIP+nMN, ncell*39+nIP+nMN+nAff, 27)

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
    exconnectcells(ncell*39+nIP, ncell*39+nIP+nMN/2, 0.05, 1, ncell*39+10*i, ncell*39+10*(i+1), 12)

  exconnectcells(ncell*39+nIP, ncell*39+nIP+nMN, 0.05, 1, ncell*39+nIP+nMN+nAff, ncell*39+nIP+nMN+2*nAff, 10)


def exconnectcells(tarstart, tarend, weight, delay, start, end, nsyn):
  global exnclist
  exnclist = []
  # not efficient but demonstrates use of pc.gid_exists
  for i in range(start, end):
    targid = random.randint(tarstart, tarend)
    if pc.gid_exists(targid):
      for j in range(nsyn):
        target = pc.gid2cell(targid)
        syn = target.synlistex[j]
        nc = pc.gid_connect(i, syn)
        exnclist.append(nc)
        nc.delay = delay
        nc.weight[0] = weight

def inhconnectcells(tarstart, tarend, weight, delay, start, end, nsyn):
  global inhnclist
  inhnclist = []
  # not efficient but demonstrates use of pc.gid_exists
  for i in range(start, end):
    targid = random.randint(tarstart, tarend)
    if pc.gid_exists(targid):
      for j in range(nsyn): 
        target = pc.gid2cell(targid)
        syn = target.synlistinh[j]
        nc = pc.gid_connect(i, syn)
        inhnclist.append(nc)
        nc.delay = delay
        nc.weight[0] = weight



addnetwork()

# run and recording

def addees():
  ''' stimulate gid 0 with NetStim to start ring '''
  global stim, ncstim
  if not pc.gid_exists(0):
    return
  stim = h.NetStim()
  stim.number = 1000000000
  stim.start = 0
  ncstim = h.NetCon(stim, pc.gid2cell(0).synlistees[0])
  ncstim.delay = 0
  ncstim.weight[0] = 0.1

addees()

def spike_record():
  ''' record spikes from all gids '''
  global tvec, idvec
  tvec = h.Vector()
  idvec = h.Vector()
  pc.spike_record(-1, tvec, idvec)

def prun(tstop):
  ''' simulation control '''
  pc.set_maxstep(10)
  h.stdinit()
  pc.psolve(tstop)


def spikeout():
  ''' report simulation results to stdout '''
  global rank, tvec, idvec
  pc.barrier()
  for i in range(nhost):
    if i == rank:
      for i in range(len(tvec)):
        print('%g %d' % (tvec.x[i], int(idvec.x[i])))
    pc.barrier()


def finish():
  ''' proper exit '''
  pc.runworker()
  pc.done()
  h.quit()

if __name__ == '__main__':
  spike_record()
  prun(150)
  spikeout()
  if (nhost > 1):
    finish()

