from memristive_spinal_cord.layer2.schemes.mk6.tier import Tier


class PolysynapticCircuit:

    def __init__(self):
        self.structure = [Tier(i+1) for i in range(6)]

    def set_interconnections(self):
        for tier in self.structure:
            tier.set_interconnections()
        for higher_tier, lower_tier in zip(self.structure[:-1], self.structure[1:]):
            higher_tier.connect(lower_tier)

    def connect_multimeters(self):
        for tier in self.structure:
            tier.connect_multimeters()

    def get_input(self):
        return [tier.get_e(0) for tier in self.structure]

    def get_output(self):
        return [tier.get_e(3) for tier in self.structure]
