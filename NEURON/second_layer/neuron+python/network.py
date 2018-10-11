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
sensnclist = []
sensstims = []

index = 0

ncell = 40       
nMN = 168      
nAff = 120
nIP = 240
ncells = ncell*23+nIP+nMN+2*nAff 

def addnetwork():
  ''' create cells and connections '''
  addinterneurons(0, ncell*23+nIP)
  addmotoneurons(ncell*23+nIP, ncell*23+nIP+nMN)
  addafferents(ncell*23+nIP+nMN, ncell*23+nIP+nMN+2*nAff)
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
  exconnectcells(0, ncell, 1, 1, ncell*23+nIP+nMN, ncell*23+nIP+nMN+nAff, 27)
  exconnectcells(ncell*3, ncell*4, 1, 1, ncell*23+nIP+nMN, ncell*23+nIP+nMN+nAff, 27)
  
  exconnectcells(0, ncell, 0.05, 2, ncell, ncell*2, 27)
  exconnectcells(ncell, ncell*2, 0.05, 2, 0, ncell, 27)
  exconnectcells(ncell*3, ncell*4, 0.05, 1, 0, ncell, 27)
  exconnectcells(ncell*3, ncell*4, 0.05, 1, ncell, ncell*2, 27)
  inhconnectcells(ncell*3, ncell*4, 0.05, 1, ncell*2, ncell*3, 27)
  inhconnectcells(ncell, ncell*2, 0.05, 1, ncell*3, ncell*4, 27)

  # generators
  for i in range(8, 13):
    exconnectcells(ncell*i, ncell*(i+1), 0.05, 2, ncell*(i+5), ncell*(i+6), 27)
    exconnectcells(ncell*(i+5), ncell*(i+6), 0.05, 2, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+10), ncell*(i+11), 0.05, 1, ncell*i, ncell*(i+1), 27)
    exconnectcells(ncell*(i+10), ncell*(i+11), 0.05, 1, ncell*(i+5), ncell*(i+6), 27)
    inhconnectcells(ncell*(i+5), ncell*(i+6), 0.04, 1, ncell*(i+10), ncell*(i+11), 27)

  # between layers
  for i in range(8, 13):
    exconnectcells(ncell*i, ncell*(i+1), 0.05, 2, ncell*(i-5), ncell*(i-4), 27)


  # inhibitory projections
  for i in range(8, 12):
    inhconnectcells(ncell*i, ncell*(i+1), 0.05, 1, ncell*7, ncell*8, 27)
    inhconnectcells(ncell*(i+5), ncell*(i+6), 0.05, 1, ncell*7, ncell*8, 27)

  inhconnectcells(ncell*8, ncell*9, 0.05, 1, ncell*5, ncell*6, 27)
  inhconnectcells(ncell*9, ncell*10, 0.05, 1, ncell*6, ncell*7, 27)
  inhconnectcells(ncell*13, ncell*14, 0.05, 1, ncell*5, ncell*6, 27)
  inhconnectcells(ncell*14, ncell*15, 0.05, 1, ncell*6, ncell*7, 27)

  # ip connections
  for i in range(0, 5):
    exconnectcells(ncell*23+int(nIP/10)*i, ncell*23+int(nIP/10)*(i+1), 0.05, 1, ncell*(i+8), ncell*(i+9), 28)
    exconnectcells(ncell*23+int(nIP/10)*(i+5), ncell*23+int(nIP/10)*(i+6), 0.05, 1, ncell*(i+13), ncell*(i+14), 28)

  # mn connections 
  for i in range(0, 12):
    exconnectcells(ncell*23+nIP, ncell*23+nIP+int(nMN/2), 0.05, 1, ncell*23+20*i, ncell*23+20*(i+1), 12)

  exconnectcells(ncell*23+nIP, ncell*23+nIP+nMN, 1, 1, ncell*23+nIP+nMN+nAff, ncell*23+nIP+nMN+2*nAff-1, 50)
  #exconnectcells(ncell*23+nIP+nMN, ncell*23+nIP+nMN+2*nAff, 1, 1, ncells, ncells, 50)

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
  global stim, ncstim, eesnclist
  stim = h.NetStim()
  stim.number = 1000000000
  stim.start = 0
  stim.interval = 25
  for i in range(ncell*23+nIP+nMN, ncell*23+nIP+nMN+2*nAff):
    if pc.gid_exists(i):
      for j in range(50):
        ncstim = h.NetCon(stim, pc.gid2cell(i).synlistees[j])
        eesnclist.append(ncstim)
        ncstim.delay = 0
        ncstim.weight[0] = 1

addees()

def addsensoryinputs():
  global sensstim, sensstims, ncsens, sensnclist
  for n in range(4):
    sensstim = h.NetStim()
    sensstim.number = 5
    sensstim.start = 25*(n+1)
    sensstim.interval = 3
    sensstims.append(sensstim)
    for i in range(ncell*(n+3), ncell*(n+5)):
      if pc.gid_exists(i):
        for j in range(50):
          ncsens = h.NetCon(sensstim, pc.gid2cell(i).synlistex[j])
          sensnclist.append(ncsens)
          ncsens.delay = 0
          ncsens.weight[0] = 1

addsensoryinputs()

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


