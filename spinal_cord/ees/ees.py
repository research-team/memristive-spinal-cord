from spinal_cord.afferents.afferent_fiber import AfferentFiber, DummySensoryAfferentFiber
from spinal_cord.params import Params

__author__ = 'Alexey Sanin'


from spinal_cord.namespace import Group, Muscle
import numpy
from pkg_resources import resource_filename
import os
import nest


class EES:

    @staticmethod
    def compute_activated_number(amplitude, muscle, group, number):
        percent = EES.compute_activated_percent(amplitude=amplitude, muscle=muscle, group=group)
        print('{}%'.format(round(percent * 100), 2))
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
        print('percent = {}'.format(percent))
        return percent

    @staticmethod
    def get_ruler_file(muscle, group, datapath='data'):
        filename = '{muscle}_full_{group}_S1_wire1'.format(muscle=muscle.value, group=group.value)
        return resource_filename('ees', os.path.join(datapath, filename))

    @staticmethod
    def generate_spiketimes(frequency_hz, how_long_s):
        how_many = int(frequency_hz * how_long_s)
        # start = 1000 // frequency_hz
        time_between_spikes = round(1000 / frequency_hz)
        # return numpy.linspace(start, how_long_s * 1000, how_many, dtype=numpy.int)
        return [0.1 + time_between_spikes * i for i in range(how_many)]

    def __init__(self, amplitude: float):
        spike_times = EES.generate_spiketimes(frequency_hz=Params.rate.value, how_long_s=20)
        self.amplitude = amplitude
        self.ees_id = nest.Create(
            model='spike_generator',
            n=1,
            params={
                'spike_times': spike_times,
            }
        )

    def connect(self, *afferents: AfferentFiber or DummySensoryAfferentFiber) -> None:
        for afferent in afferents:
            activated_number = EES.compute_activated_number(
                amplitude=self.amplitude,
                muscle=afferent.muscle,
                group=afferent.afferent,
                number=len(afferent.neuron_ids)
            )
            print('Activated number = {}'.format(activated_number))
            nest.Connect(
                pre=self.ees_id,
                post=afferent.neuron_ids,
                syn_spec={
                    'model': 'static_synapse',
                    'delay': .1,
                    'weight': 100.
                },
                conn_spec={
                    'rule': 'fixed_outdegree',
                    'outdegree': activated_number,
                    'multapses': False
                }
            )

    def connect_dummy(self, dsaf):
        nest.Connect(
            pre=self.ees_id,
            post=dsaf.neuron_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': .1,
                'weight': 100.
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 54,
                'multapses': False
            }
        )


def test() -> None:
    print(EES.compute_activated_number(amplitude=300, muscle=Muscle.FLEX, group=Group.IA, number=60))
    print(EES.compute_activated_number(amplitude=300, muscle=Muscle.FLEX, group=Group.II, number=60))
    print(EES.compute_activated_number(amplitude=300, muscle=Muscle.FLEX, group=Group.MOTO, number=60))
    print(EES.compute_activated_number(amplitude=300, muscle=Muscle.EXTENS, group=Group.IA, number=60))
    print(EES.compute_activated_number(amplitude=300, muscle=Muscle.EXTENS, group=Group.II, number=60))
    print(EES.compute_activated_number(amplitude=300, muscle=Muscle.EXTENS, group=Group.MOTO, number=60))
