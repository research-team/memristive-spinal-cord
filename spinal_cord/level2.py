from spinal_cord.level1 import Level1
from spinal_cord.polysynaptic_circuit.pc import PolysynapticCircuit
from spinal_cord.pool.pool import Pool
from spinal_cord.toolkit.plotter import ResultsPlotter
from spinal_cord.fibers import AfferentFibers


class Level2:

    def __init__(self, level1: Level1, afferents: AfferentFibers=None):
        self.pc = PolysynapticCircuit()
        self.pool = Pool()
        self.pc.connect_pool(self.pool)
        self.pool.connect_level1(level1)
        if afferents:
            self.pool.connect_sensory(afferents.dsaf)

    def plot_pool(self):
        self.pool.plot_results()

    def plot_pc(self):
        for tier in range(len(self.pc.tiers)):
            plotter = ResultsPlotter(7, 'Tier{}'.format(tier+1), 'tier{}'.format(tier+1))
            total_excitatory_groups = len(self.pc.tiers[tier].e)
            total_inhibitory_groups = len(self.pc.tiers[tier].i)
            for e in range(total_excitatory_groups):
                plotter.subplot(
                    title='E{}'.format(e),
                    first_label='average V_m',
                    first='tier{}e{}'.format(tier, e)
                )
            for i in range(total_inhibitory_groups):
                plotter.subplot(
                    title='I{}'.format(i),
                    second_label='average V_m',
                    second='tier{}i{}'.format(tier, i)
                )
            plotter.save()

    def plot_slices(self, afferent: str):
        self.pool.plot_slices(afferent)