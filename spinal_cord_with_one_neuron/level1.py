from spinal_cord_with_one_neuron.afferents.afferent_fiber import AfferentFiber
from spinal_cord_with_one_neuron.motogroups.motogroup import Motogroup
from spinal_cord_with_one_neuron.namespace import Muscle, Afferent
from spinal_cord_with_one_neuron.ees.ees import EES
from spinal_cord_with_one_neuron.toolkit.plotter import ResultsPlotter
from spinal_cord_with_one_neuron.fibers import AfferentFibers


class Level1:

    def __init__(self):

        self.flex_motogroup = Motogroup(muscle=Muscle.FLEX)
        self.extens_motogroup = Motogroup(muscle=Muscle.EXTENS)

    def connect_afferents(self, afferents: AfferentFibers):

        self.flex_motogroup.connect_motogroup(self.extens_motogroup)
        self.extens_motogroup.connect_motogroup(self.flex_motogroup)
        self.flex_motogroup.connect_afferents(afferent_ia=afferents.afferent_fiber_ia_flex,
                                              afferent_ii=afferents.afferent_fiber_ii_flex)
        self.extens_motogroup.connect_afferents(afferent_ia=afferents.afferent_fiber_ia_extens,
                                                afferent_ii=afferents.afferent_fiber_ii_extens)

    def plot_motogroups(self):
        plotter = ResultsPlotter(3, 'Average "V_m" of motogroups', 'level1')

        plotter.subplot(
            first_label='extensor',
            second_label='flexor',
            first=self.extens_motogroup.motoname,
            second=self.flex_motogroup.motoname,
            title='Moto'
        )
        plotter.subplot(
            first_label='extensor',
            second_label='flexor',
            first=self.extens_motogroup.ia_name,
            second=self.flex_motogroup.ia_name,
            title='Ia'
        )
        plotter.subplot(
            title='II',
            first_label='extensor',
            first=self.extens_motogroup.ii_name,
            second_label='flexor',
            second=self.flex_motogroup.ii_name
        )
        plotter.save()
