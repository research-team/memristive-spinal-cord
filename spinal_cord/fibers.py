from spinal_cord.afferents.afferent_fiber import AfferentFiber, DummySensoryAfferentFiber
from spinal_cord.afferents.receptor import DummySensoryReceptor
from spinal_cord.ees.ees import EES
from spinal_cord.namespace import Muscle, Afferent
from spinal_cord.toolkit.plotter import ResultsPlotter


class AfferentFibers:
    def __init__(self):
        self.afferent_fiber_ia_flex = AfferentFiber(muscle=Muscle.FLEX, afferent=Afferent.IA)
        self.afferent_fiber_ia_extens = AfferentFiber(muscle=Muscle.EXTENS, afferent=Afferent.IA)
        self.afferent_fiber_ii_flex = AfferentFiber(muscle=Muscle.FLEX, afferent=Afferent.II)
        self.afferent_fiber_ii_extens = AfferentFiber(muscle=Muscle.EXTENS, afferent=Afferent.II)
        self.dsr_flex = DummySensoryReceptor(Muscle.FLEX)
        self.dsr_extens = DummySensoryReceptor(Muscle.EXTENS)
        self.dsaf_flex = DummySensoryAfferentFiber(self.dsr_flex)
        self.dsaf_extens = DummySensoryAfferentFiber(self.dsr_extens)

        self.ees_amplitude = 300
        self.ees = EES(amplitude=self.ees_amplitude)
        self.ees.connect(
            self.afferent_fiber_ia_flex,
            self.afferent_fiber_ia_extens,
            self.afferent_fiber_ii_flex,
            self.afferent_fiber_ii_extens
        )
        self.ees.connect_dummy(self.dsaf_flex, self.dsaf_extens)

    def plot_afferents(self):
        plotter = ResultsPlotter(3, 'Average "V_m" of afferents (Stimulation amplitude: {})'.format(self.ees_amplitude), 'afferents')

        plotter.subplot(
            first_label='extensor',
            second_label='flexor',
            first=self.afferent_fiber_ia_extens.name,
            second=self.afferent_fiber_ia_flex.name,
            title='Ia'
        )
        plotter.subplot(
            first_label='extensor',
            second_label='flexor',
            first=self.afferent_fiber_ii_flex.name,
            second=self.afferent_fiber_ii_extens.name,
            title='II'
        )
        plotter.subplot(
            first_label='sensory_flex',
            first=self.dsaf_flex.name,
            second_label='sensory_extens',
            second=self.dsaf_extens.name,
            title='sensory'
        )
        plotter.save()
