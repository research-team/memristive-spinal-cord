__author__ = 'Alexey Sanin'

from spinal_cord.namespace import Group, Muscle
import numpy
from pkg_resources import resource_filename
import os


class EesStimulation:

    def __init__(self, datapath='data'):
        self.datapath = datapath

    def compute_activated_number(self, amplitude, muscle, group, number):
        percent = EesStimulation.compute_activated_percent(amplitude=amplitude, muscle=muscle, group=group)
        return int(round(number * percent))

    def compute_activated_percent(self, amplitude, muscle, group):
        if amplitude < 0 or amplitude > 600:
            raise ValueError('EES amplitude param must be between 600 and 0')
        ruler_file = self.get_ruler_file(muscle, group)
        ruler = numpy.loadtxt(ruler_file)
        available_currents = numpy.linspace(0, 600, 20)
        currents_delta = abs(available_currents - amplitude)
        closest_computed_current_index = currents_delta.argmin()
        percent = ruler[closest_computed_current_index] / max(ruler)
        return percent

    def get_ruler_file(self, muscle, group):
        filename = '{muscle}_full_{group}_S1_wire1'.format(muscle=muscle.value, group=group.value)
        return resource_filename('ees', os.path.join(self.datapath, filename))


def test() -> None:
    ees = EesStimulation()
    print(ees.compute_activated_percent(amplitude=300, muscle=Muscle.FLEX, group=Group.IA))
    print(ees.compute_activated_percent(amplitude=300, muscle=Muscle.FLEX, group=Group.II))
    print(ees.compute_activated_percent(amplitude=300, muscle=Muscle.FLEX, group=Group.MOTO))
    print(ees.compute_activated_percent(amplitude=300, muscle=Muscle.EXTENS, group=Group.IA))
    print(ees.compute_activated_percent(amplitude=300, muscle=Muscle.EXTENS, group=Group.II))
    print(ees.compute_activated_percent(amplitude=300, muscle=Muscle.EXTENS, group=Group.MOTO))