from memristive_spinal_cord.layer2.schemes.mk5.tier import Tier


class PolysynapticCircuit:
    @staticmethod
    def get_number_of_neurons(self):
        return Tier.get_number_of_neurons()

    def __init__(self):
        self.tiers = [Tier(i + 1) for i in range(6)]

    def get_input(self):
        return self.get_tiers()[0].get_e(0)

    def get_output(self):
        return [self.get_tiers()[i].get_e(2) for i in range(6)]

    def set_connections(self):
        for tier, lower_tier in zip(self.get_tiers()[1:], self.get_tiers()[:-1]):
            tier.connect(lower_tier)
        for tier in self.get_tiers():
            tier.set_connections()

    def connect_multimeters(self):
        for tier in self.get_tiers():
            tier.connect_multimeters()

    def get_tiers(self):
        return self.tiers
