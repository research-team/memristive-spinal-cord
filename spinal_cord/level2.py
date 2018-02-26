from spinal_cord.level1 import Level1
from spinal_cord.polysynaptic_circuit.pc import PolysynapticCircuit
from spinal_cord.pool.pool import Pool


class Level2:

    def __init__(self, level1: Level1):
        self.pc = PolysynapticCircuit()
        self.pool = Pool()

        self.pc.connect_pool(self.pool)
        self.pool.connect_level1(level1)

    def plot_pool(self):
        self.pool.plot_results()