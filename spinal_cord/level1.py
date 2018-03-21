from spinal_cord.afferents.afferent_fiber import AfferentFiber
from spinal_cord.motogroups.motogroup import Motogroup
from spinal_cord.namespace import Muscle, Afferent
from spinal_cord.ees.ees import EES
from spinal_cord.params import Params
from spinal_cord.toolkit.plotter import ResultsPlotter
from spinal_cord.fibers import AfferentFibers


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
            title='Ia (inter inh)'
        )
        plotter.subplot(
            title='II (inter exc)',
            first_label='extensor',
            first=self.extens_motogroup.ii_name,
            second_label='flexor',
            second=self.flex_motogroup.ii_name
        )
        plotter.save()

    def plot_slices(self, afferent: str, time=40.):
        n_slices = 7
        plotter = ResultsPlotter(n_slices, 'Average "V_m" of Moto, stimulation rate: {}Hz'.format(Params.rate.value), 'moto_slices')
        plotter.subplot_with_slices(
            slices=n_slices,
            first_label='extensor',
            first=self.extens_motogroup.motoname,
            second_label='flexor',
            second=self.flex_motogroup.motoname,
            third_label='stimuli',
            third=afferent,
            title='Pool'
        )
        plotter.save()

    def plot_moto_only(self):
        plotter = ResultsPlotter(1, 'Average "V_m" of motoneurons, stimulation rate: {}Hz'.format(Params.rate.value), 'moto')

        plotter.subplot(
            first_label='extensor',
            second_label='flexor',
            first=self.extens_motogroup.motoname,
            second=self.flex_motogroup.motoname,
            title='Moto'
        )
        plotter.save()
