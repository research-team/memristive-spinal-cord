from memristive_spinal_cord.layer2.schemes.mk3.tier import Tier


class PolysynapticCircuit:

    def __init__(self):
        self.tiers = [Tier(i+1) for i in range(1)]

    def get_input(self): return self.tiers[0].get_control_group()

    def get_output(self): return [self.tiers[i] for i in range(len(self.tiers))]

    def get_tiers(self): return self.tiers
