from memristive_spinal_cord.layer2.schemes.mk3.hidden_tier import HiddenTier
from memristive_spinal_cord.layer2.schemes.mk3.tier import Tier


class PolysynapticCircuit:
    def __init__(self):
        self.tiers = [Tier(i+1) for i in range(6)]
        self.hidden_tiers = [HiddenTier(i+1) for i in range(5)]

    def set_connections(self):
        for upper_tier, lower_tier, hidden_tier in zip(
                self.tiers[1:], self.tiers[:-1], self.hidden_tiers):
            upper_tier.connect(tier=lower_tier)
            hidden_tier.connect(upper_tier=upper_tier, lower_tier=lower_tier)
    
    def get_input(self): return self.tiers[0].get_control_group()

    def get_output(self): return [tier.get_left_group() for tier in self.tiers]

    def get_tiers(self): return self.tiers

    def get_hidden_tiers(self): return self.hidden_tiers

    @staticmethod
    def get_number_of_neurons():
        return Tier.get_number_of_neurons() + HiddenTier.get_number_of_neurons()