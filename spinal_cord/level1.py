from spinal_cord.afferents.afferent_fiber import AfferentFiber
from spinal_cord.motogroups.motogroup import Motogroup
from spinal_cord.namespace import Muscle, Afferent
from spinal_cord.ees.ees import EES
from spinal_cord.toolkit.plotter import ResultsPlotter
from spinal_cord.afferents.receptor import Receptor


class Level1:

    def __init__(self):

        self.flex_motogroup = Motogroup(muscle=Muscle.FLEX)
        self.extens_motogroup = Motogroup(muscle=Muscle.EXTENS)

        self.afferent_fiber_ia_flex = AfferentFiber(muscle=Muscle.FLEX, afferent=Afferent.IA)
        self.afferent_fiber_ia_extens = AfferentFiber(muscle=Muscle.EXTENS, afferent=Afferent.IA)
        self.afferent_fiber_ii_flex = AfferentFiber(muscle=Muscle.FLEX, afferent=Afferent.II)
        self.afferent_fiber_ii_extens = AfferentFiber(muscle=Muscle.EXTENS, afferent=Afferent.II)

        self.flex_motogroup.connect_motogroup(self.extens_motogroup)
        self.extens_motogroup.connect_motogroup(self.flex_motogroup)
        self.flex_motogroup.connect_afferents(afferent_ia=self.afferent_fiber_ia_flex, afferent_ii=self.afferent_fiber_ii_flex)
        self.extens_motogroup.connect_afferents(afferent_ia=self.afferent_fiber_ia_extens, afferent_ii=self.afferent_fiber_ii_extens)

        self.ees = EES()
        self.ees.connect(
            500,
            self.afferent_fiber_ia_flex,
            self.afferent_fiber_ia_extens,
            self.afferent_fiber_ii_flex,
            self.afferent_fiber_ii_extens
        )

    def plot_afferents(self):
        plotter = ResultsPlotter(2, 'Average "V_m" of afferents')

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
        plotter.show()

    def plot_motogroups(self):
        plotter = ResultsPlotter(3, 'Average "V_m" of motogroups')

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
        plotter.show()
