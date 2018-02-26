from spinal_cord.polysynaptic_circuit.pc import PolysynapticCircuit
from spinal_cord.pool.pool import Pool


class Level2:

    def __init__(self):
        pc = PolysynapticCircuit()
        pool = Pool()

        pc.connect_pool(pool)