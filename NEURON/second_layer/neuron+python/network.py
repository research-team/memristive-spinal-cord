from neuron import h
h.load_file('nrngui.hoc')
pc = h.ParallelContext()
rank = int(pc.id())
nhost = int(pc.nhost())

from interneuron import interneuron
from motoneuron import motoneuron
from afferent import afferent


# Network Creation


interneurons = []
motoneurons = []
afferents = []
nclist = []

ncell = 20       
nsyn = 2
nMN = 169      
nAff = 120
nIP = 120
ncells = ncell*39+nIP+nMN+2*nAff 

def addnetwork():
  addinterneurons(0, ncell*39+nIP)
  addmotoneurons(ncell*39+nIP, ncell*39+nIP+nMN)
  addafferents(ncell*39+nIP+nMN, ncell*39+nIP+nMN+2*nAff)

def addinterneurons(start, end):
  global interneurons, rank, nhost
  interneurons = []
  for i in range(rank+start, ncell, nhost):
    cell = interneuron()
    interneurons.append(cell)
    pc.set_gid2node(i, rank)
    nc = cell.connect2target(None)
    pc.cell(i, nc)

def addmotoneurons(start, end):
  global motoneurons, rank, nhost
  motoneurons = []
  for i in range(rank+start, ncell, nhost):
    cell = motoneuron()
    motoneurons.append(cell)
    pc.set_gid2node(i, rank)
    nc = cell.connect2target(None)
    pc.cell(i, nc)

def addafferents(start, end):
  global afferents, rank, nhost
  afferents = []
  for i in range(rank+start, ncell, nhost):
    cell = afferent()
    afferents.append(cell)
    pc.set_gid2node(i, rank)
    nc = cell.connect2target(None)
    pc.cell(i, nc)

addnetwork()

#Instrumentation - stimulation and recording

def addees():
  ''' stimulate gid 0 with NetStim to start ring '''
  global stim, ncstim
  if not pc.gid_exists(0):
    return
  stim = h.NetStim()
  stim.number = 1000000000
  stim.start = 0
  #ncstim = h.NetCon(stim, pc.gid2cell(0).synlist[0])
  ncstim.delay = 0
  #ncstim.weight[0] = 0.01

addees()

print("network created")

