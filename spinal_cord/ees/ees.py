from spinal_cord.afferents.afferent_fiber import AfferentFiber

__author__ = 'Alexey Sanin'


from spinal_cord.namespace import Group, Muscle
import numpy
from pkg_resources import resource_filename
import os
import nest
from random import sample


class EES:

    @staticmethod
    def compute_activated_number(amplitude, muscle, group, number):
        percent = EES.compute_activated_percent(amplitude=amplitude, muscle=muscle, group=group)
        return int(round(number * percent))

    @staticmethod
    def compute_activated_percent(amplitude, muscle, group):
        if amplitude < 0 or amplitude > 600:
            raise ValueError('EES amplitude param must be between 600 and 0')
        ruler_file = EES.get_ruler_file(muscle, group)
        ruler = numpy.loadtxt(ruler_file)
        available_currents = numpy.linspace(0, 600, 20)
        currents_delta = abs(available_currents - amplitude)
        closest_computed_current_index = currents_delta.argmin()
        percent = ruler[closest_computed_current_index] / max(ruler)
        return percent

    @staticmethod
    def get_ruler_file(muscle, group, datapath='data'):
        filename = '{muscle}_full_{group}_S1_wire1'.format(muscle=muscle.value, group=group.value)
        return resource_filename('ees', os.path.join(datapath, filename))

    @staticmethod
    def generate_spiketimes(frequency_hz, how_long_s):
        how_many = int(frequency_hz * how_long_s)
        start = 1000 // frequency_hz
        return numpy.linspace(start, how_long_s * 1000, how_many, dtype=numpy.int)

    def __init__(self):
        self.ees_id = nest.Create(
            model='spike_generator',
            n=1,
            params={
                'spike_times': EES.generate_spiketimes(frequency_hz=40, how_long_s=1000).astype(float)
            }
        )

    def connect(self, amplitude: int, *afferents: AfferentFiber) -> None:
        for afferent in afferents:
            activated_number = EES.compute_activated_number(
                amplitude=amplitude,
                muscle=afferent.muscle,
                group=afferent.afferent,
                number=len(afferent.neuron_ids)
            )
            nest.Connect(
                pre=self.ees_id,
                post=afferent.neuron_ids,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': 1.,
                    'weight': 10
                },
                conn_spec={
                    'rule': 'output_degree',
                    'degree': activated_number
                }
            )


def test() -> None:
    print(EES.compute_activated_percent(amplitude=300, muscle=Muscle.FLEX, group=Group.IA))
    print(EES.compute_activated_percent(amplitude=300, muscle=Muscle.FLEX, group=Group.II))
    print(EES.compute_activated_percent(amplitude=300, muscle=Muscle.FLEX, group=Group.MOTO))
    print(EES.compute_activated_percent(amplitude=300, muscle=Muscle.EXTENS, group=Group.IA))
    print(EES.compute_activated_percent(amplitude=300, muscle=Muscle.EXTENS, group=Group.II))
    print(EES.compute_activated_percent(amplitude=300, muscle=Muscle.EXTENS, group=Group.MOTO))