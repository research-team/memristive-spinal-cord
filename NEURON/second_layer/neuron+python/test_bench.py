import logging
logging.basicConfig(filename='logs_b.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
logging.info("let's get it started")
import numpy as np
from neuron import h
h.load_file('nrngui.hoc')
import time
import random


#paralleling NEURON stuff
pc = h.ParallelContext()
rank = int(pc.id())
nhost = int(pc.nhost())

exnclist = []
stims = []
stimnclist = []

from interneuron_izh import interneuron_izh

'''
network creation
see topology https://github.com/research-team/memristive-spinal-cord/blob/master/doc/diagram/cpg_generator_FE_paper.png
and all will be clear
'''
class CPG:
    def __init__(self):

        self.interneurons = []
        # self.ncell = N
        self.groups = []
        # self.nsyn = N

        self.Ia_gen_E = self.addgener(0, 150, 500, False)
        self.Ia_gen_F = self.addgener(650, 150, 500, False)

        self.Ia_aff_E = self.addpool(120, 10)
        self.Ia_aff_F = self.addpool(120, 10)

        self.Ia_E = self.addpool(196, 30)
        self.Ia_F = self.addpool(196, 30)
        self.R_E = self.addpool(196, 30)
        self.R_F = self.addpool(196, 30)
        self.mns_E = self.addpool(200, 30)
        self.mns_F = self.addpool(200, 30)

        genconnect(self.Ia_gen_E, self.Ia_aff_E, 0.00001, 1)
        genconnect(self.Ia_gen_F, self.Ia_aff_F, 0.00001, 1)

        connectcells(self.Ia_aff_E, self.Ia_E, 0.00001, 1)
        connectcells(self.mns_E, self.R_E, 0.00001, 1)
        connectcells(self.Ia_E, self.mns_F, 0.000001, 1, True)
        connectcells(self.R_E, self.mns_E, 0.000001, 1, True)
        connectcells(self.R_E, self.Ia_E, 0.000001, 1, True)

        connectcells(self.Ia_aff_F, self.Ia_F, 0.00001, 1)
        connectcells(self.mns_F, self.R_F, 0.00001, 1)
        connectcells(self.Ia_F, self.mns_E, 0.000001, 1, True)
        connectcells(self.R_F, self.mns_F, 0.000001, 1, True)
        connectcells(self.R_F, self.Ia_F, 0.000001, 1, True)

        connectcells(self.R_E, self.R_F, 0.000001, 1, True)
        connectcells(self.R_F, self.R_E, 0.000001, 1, True)
        connectcells(self.Ia_E, self.Ia_F, 0.000001, 1, True)
        connectcells(self.Ia_F, self.Ia_E, 0.000001, 1, True)

    def addpool(self, num, nsyn, name="test", neurontype="int"):
        '''
        Creates interneuronal pool and returns gids of pool
        Parameters
        ----------
        num: int
            neurons number in pool
        neurontype: string
            int: interneuron
            delay: interneuron with 5ht
            moto: motoneuron
            aff: afferent
        Returns
        -------
        gids: list
            the list of neurons gids
        '''
        gids = []
        gid = 0
        for i in range(rank, num, nhost):
            cell = interneuron_izh(nsyn)
            self.interneurons.append(cell)
            while pc.gid_exists(gid) != 0:
                gid += 1
            gids.append(gid)
            pc.set_gid2node(gid, rank)
            nc = cell.connect2target(None)
            pc.cell(gid, nc)

        self.groups.append((gids, name))

        return gids

    def addgener(self, start, freq, nums, r=True):
        '''
        Creates generator and returns generator gid
        Parameters
        ----------
        start: int
            generator start up
        freq: int
            generator frequency
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
            stim.start = random.uniform(start - 3, start)
            stim.noise = 0.05
        else:
            stim.start = start
        stim.interval = int(1000 / freq)
        #skinstim.noise = 0.1
        stims.append(stim)
        while pc.gid_exists(gid) != 0:
            gid += 1
        pc.set_gid2node(gid, rank)
        ncstim = h.NetCon(stim, None)
        pc.cell(gid, ncstim)
        return gid

def connectcells(pre, post, weight, delay, inhtype = False):
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
      inhtype: bool
          is this connection inhibitory?
    '''
    for i in post:
        if pc.gid_exists(i):
            for j in range(30):
                srcgid = random.randint(pre[0], pre[-1])
                target = pc.gid2cell(i)
                if inhtype:
                    syn = target.synlistinh[j]
                    nc = pc.gid_connect(srcgid, syn)
                    # nc.weight[0] = 0 # str
                else:
                    syn = target.synlistex[j]
                    nc = pc.gid_connect(srcgid, syn)
                    # nc.weight[0] = random.gauss(weight, weight / 6) # str
                exnclist.append(nc)
                nc.weight[0] = weight
                nc.delay = delay

def genconnect(gen_gid, afferents_gids, weight, delay, inhtype = False, N = 50):
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
      inhtype: bool
          is this connection inhibitory?
    '''
    nsyn = random.randint(N, N+5)
    for i in afferents_gids:
        if pc.gid_exists(i):
            for j in range(10):
                target = pc.gid2cell(i)
                syn = target.synlistex[j]
                nc = pc.gid_connect(gen_gid, syn)
                stimnclist.append(nc)
                nc.delay = delay
                nc.weight[0] = weight

def prun():
    ''' simulation control
    Parameters
    ----------
    speed: int
      duration of each layer
    '''
    tstop = 1000
    pc.set_maxstep(10)
    h.stdinit()
    pc.psolve(tstop)


def finish():
    ''' proper exit '''
    pc.runworker()
    pc.done()
    # print("hi after finish")
    h.quit()

if __name__ == '__main__':
    '''
    cpg_ex: cpg
        topology of central pattern generation + reflex arc
    '''
    k_nrns = 0
    k_name = 1

    versions = 10

    #for j in range(1000, 11000, 1000):
    for i in range(versions):
        cpg_ex = CPG()
        logging.info("test RA IZH")
        logging.info("test %d" %  (i))

        print("- " * 10, "\nstart")
        start_time = time.time()
        prun()
        logging.info("--- %s ms ---" % ((time.time() - start_time)*1000))
        print("- " * 10, "\nend")

        logging.info("done")

    finish()
