from neucogar.Nucleus import Nucleus
from memristive_spinal_cord.layer2.models import Neurotransmitters
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.parameters import Constants
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.neurons import Neurons
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.synapses import Synapses
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.parameters import Weights
from memristive_spinal_cord.layer2.models import ConnectionTypes


class Tier:
    def __init__(self, tier:int):
        """
        The tier in current scheme is three groups of neurons: excitatory left and right,
        and one inhibitory group
        Args:
            tier: a number of the tier in order
        """
        self.tier = tier

        self.left_group = Nucleus(nucleus_name='Tier{}Left'.format(tier))
        self.left_group.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=Constants.NEURONS_IN_GROUP.value,
            params=Neurons.NEUCOGAR.value
        )

        self.right_group = Nucleus(nucleus_name='Tier{}Right'.format(tier))
        self.right_group.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=Constants.NEURONS_IN_GROUP.value,
            params=Neurons.NEUCOGAR.value
        )

        self.inhibitory_group = Nucleus(nucleus_name='Tier{}Inhibitory'.format(tier))
        self.inhibitory_group.addSubNucleus(
            neurotransmitter=Neurotransmitters.GABA.value,
            number=Constants.NEURONS_IN_GROUP.value,
            params=Neurons.NEUCOGAR.value
        )

        # connect right group to left group
        self.right_group.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.left_group.nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.RL.value.reverse()[tier-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

        # connect left group to right group
        self.left_group.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.right_group.nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.LR.value.reverse()[tier-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

        # connect inhibitory group to right group
        self.inhibitory_group.nuclei(Neurotransmitters.GABA.value).connect(
            nucleus=self.right_group.nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GABAERGIC.value,
            weight=Weights.IR.value.reverse()[tier-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

        # connect right group to inhibitory group
        self.right_group.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.inhibitory_group.nuclei(Neurotransmitters.GABA.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.RI.value.reverse()[tier-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

    def connect_to_tier(self, tier):
        """
        Connects this tier to a lower tier
        Args:
            tier (Tier): a lower tier

        Returns:
            bool

        """

        if self.tier - tier.get_tier() == 1:

            self.left_group.nuclei(Neurotransmitters.GLU.value).connect(
                nucleus=tier.get_left_group().nuclei(Neurotransmitters.GLU.value),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=Weights.LL.value.reverse()[tier.get_tier()-1],
                conn_type=ConnectionTypes.ONE_TO_ONE.value
            )

            tier.get_right_group().nuclei(Neurotransmitters.GLU.value).connect(
                nucleus=self.right_group.nuclei(Neurotransmitters.GLU.value),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=Weights.RR.value.reverse()[tier.get_tier()-2],
                conn_type=ConnectionTypes.ONE_TO_ONE.value
            )

            return True
        else:
            return False

    def get_tier(self): return self.tier

    def get_left_group(self): return self.left_group

    def get_right_group(self): return self.right_group

    def get_inhibitory_group(self): return self.inhibitory_group if self.inhibitory_group else None
