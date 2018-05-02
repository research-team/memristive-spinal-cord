import os
from ows.src.tools.multimeter import add_multimeter
from ows.src.tools.spike_detector import add_spike_detector
from ows.src.paths import raw_data_path
from nest import Create, Connect, SetStatus
from ows.src.params import num_sublevels, num_spikes, rate
from random import uniform
from ows.src.components.sublevel import Sublevel
from ows.src.components.rybak import Rybak
from ows.src.tools.dummy_sensory import DummySensoryGenerator


def connect(pre, post, weight, num_synapses=100):
        Connect(
            pre=pre,
            post=post,
            syn_spec={
                'model': 'static_synapse',
                'delay': .1,
                'weight': weight
            },
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': num_synapses})

class Topology:

    def connect_ees(self, num_spikes, rate):

        period = round(1000 / rate, 1)

        if num_spikes:
            self.ees = Create(
                model='spike_generator',
                n=1,
                params={'spike_times': [period + i * period for i in range(num_spikes)]})
            for post in [
                self.rybak.aff_ia_flexor.gids,
                self.rybak.aff_ia_extensor.gids,
                self.rybak.sensory_flexor.gids,
                self.rybak.sensory_extensor.gids]:
                Connect(
                    pre=self.ees,
                    post=post,
                    syn_spec={
                        'model': 'static_synapse',
                        'delay': 0.1,
                        'weight': 500.
                    },
                    conn_spec={
                        'rule': 'fixed_outdegree',
                        'outdegree': 120,
                        'multapses': False})

    def __init__(self):

        self.sublevels = [Sublevel(index=i) for i in range(num_sublevels)]
        self.rybak = Rybak()
        # level 2 interconnections
        for i in range(num_sublevels-1):
            Connect(
                pre=self.sublevels[i].e0,
                post=self.sublevels[i+1].e0,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.4,
                    'weight': 75.},
                conn_spec={
                    'rule': 'one_to_one'})
            Connect(
                pre=self.sublevels[i].e3,
                post=self.sublevels[i+1].e0,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.4,
                    'weight': 0.},
                conn_spec={
                    'rule': 'one_to_one'})
            Connect(
                pre=self.sublevels[i].e2,
                post=self.sublevels[i+1].e2,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 0.7,
                    'weight': 150.},
                conn_spec={
                    'rule': 'one_to_one'})

        self.s0gen = DummySensoryGenerator(inversion=False, rate=100.)
        # connect(pre=self.s0gen.id, post=self.rybak.sensory_flexor.gids, weight=300)
        self.s1gen = DummySensoryGenerator(inversion=True, rate=100.)
        # connect(pre=self.s1gen.id, post=self.rybak.sensory_extensor.gids, weight=300)

        for pre, post in zip(
            [self.s0gen.id, self.s1gen.id],
            [self.rybak.sensory_flexor.gids, self.rybak.sensory_extensor.gids]):
            Connect(
                pre=pre,
                post=post,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': .1,
                    'weight': 1.},
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': 30,
                    'multapses': False})

        connect(pre=self.rybak.s1.gids, post=self.sublevels[0].e0, weight=0.5)

        for i in range(num_sublevels):
            connect(pre=self.sublevels[i].e2, post=self.rybak.pool_flexor.gids, weight=10.)
            connect(pre=self.sublevels[i].e2, post=self.rybak.pool_extensor.gids, weight=10.)

        self.connect_ees(num_spikes, rate)

        self.ia_extens_gen = Create(
            model='spike_generator',
            n=1,
            params={
                'spike_times': [10. + 10. * i for i in range(100)],
                'spike_weights': [500. for i in range(100)]})
        self.ia_flex_gen = Create(
            model='spike_generator',
            n=1,
            params={
                'spike_times': [20. + 10. * i for i in range(100)],
                'spike_weights': [500. for i in range(100)]})

        # for pre, post in zip(
        #     [self.ia_extens_gen, self.ia_flex_gen],
        #     [self.rybak.aff_ia_extensor.gids, self.rybak.aff_ia_flexor.gids]):
        #     Connect(
        #         pre=pre,
        #         post=post,
        #         syn_spec={
        #             'model': 'static_synapse',
        #             'delay': .1,
        #             'weight': 1.},
        #         conn_spec={
        #             'rule': 'fixed_outdegree',
        #             'outdegree': 15,
        #             'multapses': False})