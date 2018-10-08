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
  for i in range(2):
    exconnectcells(ncell*i, ncell*(i+1), 0.05, 2, ncell*(i+3), ncell*(i+4), 27)
    exconnectcells(ncell*(i+5), ncell*(i+6), 0.05, 2, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+10), ncell*(i+11), 0.05, 1, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+10), ncell*(i+11), 0.05, 1, ncell*(i+5), ncell*(i+6), 27)
    inhconnectcells(ncell*i, ncell*(i+1), 0.04, 1, ncell*(i+10), ncell*(i+11), 27)
    inhconnectcells(ncell*(i+5), ncell*(i+6), 0.04, 1, ncell*(i+10), ncell*(i+11), 27)



def exconnectcells(start, end, weight, delay, tarstart, tarend, nsyn):
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

def inhconnectcells(start, end, weight, delay, tarstart, tarend, nsyn):
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
  prun(10)
  spikeout()
  if (nhost > 1):
    finish()

