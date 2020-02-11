import logging
logging.basicConfig(filename='logs.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
logging.info("let's get it started")
import numpy as np
from neuron import h
h.load_file('nrngui.hoc')

#paralleling NEURON staff
pc = h.ParallelContext()
rank = int(pc.id())
nhost = int(pc.nhost())

#param
ees_fr = 40 # frequency of EES
nMN = 20
nAff = 12
nInt = 19
N = 50

exnclist = []
inhnclist = []
eesnclist = []
stimnclist = []

from interneuron import interneuron
from motoneuron import motoneuron
from bioaff import bioaff
from bioaffrat import bioaffrat


import random

'''
network creation
see topology https://github.com/research-team/memristive-spinal-cord/blob/master/doc/diagram/cpg_generator_FE_paper.png
and all will be clear
'''
class RA:
    def __init__(self, ees_fr, N):

        self.interneurons = []
        self.motoneurons = []
        self.afferents = []
        self.stims = []
        self.ncell = N
        self.groups = []
        self.motogroups = []
        self.affgroups = []

        self.Ia_aff_E = self.addpool(nAff, "Ia_aff_E", "aff")
        self.Ia_aff_F = self.addpool(nAff, "Ia_aff_F", "aff")

        self.mns_E = self.addpool(nMN, "mns_E", "moto")
        self.mns_F = self.addpool(nMN, "mns_F", "moto")

        '''reflex arc'''
        self.Ia_E = self.addpool(nInt, "Ia_E", "int")
        self.R_E = self.addpool(nInt, "R_E", "int")

        self.Ia_F = self.addpool(nInt, "Ia_F", "int")
        self.R_F = self.addpool(nInt, "R_F", "int")
        # self.Iagener_E = []
        # self.Iagener_F = []

        '''ees'''
        self.ees = self.addgener(1, ees_fr, 10000, False)

        self.C1 = self.addgener(50, 200, 15)
        self.C0 = self.addgener(150, 200, 15)

        self.Iagener_E = self.addIagener(self.mns_E)
        self.Iagener_F = self.addIagener(self.mns_F)

        genconnect(self.ees, self.Ia_aff_E, 0.65, 2)
        genconnect(self.ees, self.Ia_aff_F, 0.5, 2)
        genconnect(self.Iagener_E, self.Ia_aff_E, 0.5, 2)
        genconnect(self.Iagener_F, self.Ia_aff_F, 0.5, 2)

        connectcells(self.Ia_aff_E, self.mns_E, 0.65, 2)
        connectcells(self.Ia_aff_F, self.mns_F, 0.65, 2)
        genconnect(self.C1, self.mns_E, 0.5, 3)
        genconnect(self.C0, self.mns_F, 0.5, 3)
        genconnect(self.C1, self.Ia_aff_F, 0.8, 1, True)
        genconnect(self.C0, self.Ia_aff_E, 0.8, 1, True)

        '''reflex arc'''
        connectcells(self.Ia_aff_E, self.Ia_E, 0.08, 1)
        connectcells(self.mns_E, self.R_E, 0.00025, 1)
        connectcells(self.Ia_E, self.mns_F, 0.08, 1, True)
        connectcells(self.R_E, self.mns_E, 0.0005, 1, True)
        connectcells(self.R_E, self.Ia_E, 0.001, 1, True)

        connectcells(self.Ia_aff_F, self.Ia_F, 0.08, 1)
        connectcells(self.mns_F, self.R_F, 0.0004, 1)
        connectcells(self.Ia_F, self.mns_E, 0.04, 1, True)
        connectcells(self.R_F, self.mns_F, 0.0005, 1, True)
        connectcells(self.R_F, self.Ia_F, 0.001, 1, True)

        connectcells(self.R_E, self.R_F, 0.04, 1, True)
        connectcells(self.R_F, self.R_E, 0.04, 1, True)
        connectcells(self.Ia_E, self.Ia_F, 0.08, 1, True)
        connectcells(self.Ia_F, self.Ia_E, 0.08, 1, True)

    def addpool(self, num, name="test", neurontype="int"):
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
        if neurontype.lower() == "delay":
            delaytype = True
        else:
            delaytype = False
        if neurontype.lower() == "moto":
            diams = motodiams(num)
        for i in range(rank, num, nhost):
            if neurontype.lower() == "moto":
                cell = motoneuron(diams[i])
                self.motoneurons.append(cell)
            elif neurontype.lower() == "aff":
                cell = bioaffrat()
                self.afferents.append(cell)
            else:
                cell = interneuron(delaytype)
                self.interneurons.append(cell)
            while pc.gid_exists(gid) != 0:
                gid += 1
            gids.append(gid)
            pc.set_gid2node(gid, rank)
            nc = cell.connect2target(None)
            pc.cell(gid, nc)

        # ToDo remove me (Alex code) - NO
        if neurontype.lower() == "moto":
            self.motogroups.append((gids, name))
        elif neurontype.lower() == "aff":
            self.affgroups.append((gids, name))
        else:
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
            stim.noise = 0.1
        else:
            stim.noise = 0.0
        stim.interval = 1000 / freq
        stim.start = start
        #skinstim.noise = 0.1
        self.stims.append(stim)
        while pc.gid_exists(gid) != 0:
            gid += 1
        pc.set_gid2node(gid, rank)
        ncstim = h.NetCon(stim, None)
        pc.cell(gid, ncstim)
        return gid

    def addIagener(self, mn):
        '''
        Creates self.Ia generators and returns generator gids
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
        gid = 0
        srcgid = random.randint(mn[0], mn[-1])
        moto = pc.gid2cell(srcgid)
        print(moto)
        stim = h.IaGenerator(0.5)
        h.setpointer(moto.muscle.muscle_unit(0.5)._ref_F_fHill, 'fhill', stim)
        self.stims.append(stim)
        while pc.gid_exists(gid) != 0:
            gid += 1
        pc.set_gid2node(gid, rank)
        ncstim = h.NetCon(stim, None)
        pc.cell(gid, ncstim)
        print(gid)

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
    nsyn = random.randint(3, 5)
    for i in post:
        if pc.gid_exists(i):
            for j in range(nsyn):
                srcgid = random.randint(pre[0], pre[-1])
                target = pc.gid2cell(i)
                if inhtype:
                    syn = target.synlistinh[j]
                    nc = pc.gid_connect(srcgid, syn)
                    inhnclist.append(nc)
                    # str nc.weight[0] = 0
                else:
                    syn = target.synlistex[j]
                    nc = pc.gid_connect(srcgid, syn)
                    exnclist.append(nc)
                    # str nc.weight[0] = random.gauss(weight, weight / 10)
                nc.weight[0] = random.gauss(weight, weight / 10)
                nc.delay = random.gauss(delay, delay / 9)


def genconnect(gen_gid, afferents_gids, weight, delay, inhtype = False):
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
    nsyn = random.randint(3, 5)
    for i in afferents_gids:
        if pc.gid_exists(i):
            for j in range(nsyn):
                target = pc.gid2cell(i)
                if inhtype:
                    syn = target.synlistinh[j]
                else:
                    syn = target.synlistex[j]
                nc = pc.gid_connect(gen_gid, syn)
                stimnclist.append(nc)
                nc.delay = random.gauss(delay, delay / 7)
                nc.weight[0] = random.gauss(weight, weight / 10)

def spike_record(pool):
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
        vec.record(cell.soma(0.5)._ref_v)
        v_vec.append(vec)
    return v_vec

def motodiams(number):
    nrn_number = number
    standby_percent = 70
    active_percent = 100 - standby_percent

    standby_size = int(nrn_number * standby_percent / 100)
    active_size = nrn_number - standby_size

    loc_active, scale_active = 27, 3
    loc_stanby, scale_stanby = 44, 4

    x2 = np.concatenate([np.random.normal(loc=loc_active, scale=scale_active, size=active_size),
                     np.random.normal(loc=loc_stanby, scale=scale_stanby, size=standby_size)])

    return x2


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


def spikeout(pool, name, v_vec):
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
            path = str('./res/' + name + 'r%d_PLT' % (rank))
            f = open(path, 'w')
            for v in outavg:
                f.write(str(v) + "\n")
        pc.barrier()


def prun(t=300):
    ''' simulation control
    Parameters
    ----------
    speed: int
      duration of each layer
    '''
    tstop = t
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
    ra_ex: cpg
        topology of central pattern generation + reflex arc
    '''
    k_nrns = 0
    k_name = 1

    ra_ex = RA(ees_fr, N)
    logging.info("created")
    motorecorders = []
    for group in ra_ex.motogroups:
        motorecorders.append(spike_record(group[k_nrns]))
    affrecorders = []
    for group in ra_ex.affgroups:
      affrecorders.append(spike_record(group[k_nrns]))
    # recorders = []
    # for group in ra_ex.groups:
    #   recorders.append(spike_record(group[k_nrns], i))
    logging.info("added recorders")

    print("- " * 10, "\nstart")
    prun()
    print("- " * 10, "\nend")

    for group, recorder in zip(ra_ex.motogroups, motorecorders):
        spikeout(group[k_nrns], group[k_name], recorder)
    for group, recorder in zip(ra_ex.affgroups, affrecorders):
      spikeout(group[k_nrns], group[k_name], recorder)
    # for group, recorder in zip(ra_ex.groups, recorders):
    #   spikeout(group[k_nrns], group[k_name], i, recorder)
    logging.info("recorded")

    finish()
