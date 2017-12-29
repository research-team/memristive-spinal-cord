from neucogar.Nucleus import Nucleus
from memristive_spinal_cord.layer2.models import Neurotransmitters
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.parameters import Constants
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.neurons import Neurons
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.synapses import Synapses
from memristive_spinal_cord.layer2.schemes.hidden_tiers.components.parameters import Weights
from memristive_spinal_cord.layer2.models import ConnectionTypes


class Tier:
    __number_of_neurons = 0

    @classmethod
    def add_neurons(cls, number:int):
        cls.__number_of_neurons += number

    @classmethod
    def get_number_of_neurons(cls):
        return cls.__number_of_neurons

    def __init__(self, index:int):
        """
        Args:
            index: a number of the index in order
        """
        self.index = index

        self.left_group = Nucleus(nucleus_name='Tier{}Left'.format(index))
        self.left_group.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=Constants.NEURONS_IN_GROUP.value,
            params=Neurons.NEUCOGAR.value
        )

        self.right_group = Nucleus(nucleus_name='Tier{}Right'.format(index))
        self.right_group.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=Constants.NEURONS_IN_GROUP.value,
            params=Neurons.NEUCOGAR.value
        )

        self.inhibitory_group = Nucleus(nucleus_name='Tier{}Inhibitory'.format(index))
        self.inhibitory_group.addSubNucleus(
            neurotransmitter=Neurotransmitters.GABA.value,
            number=Constants.NEURONS_IN_GROUP.value,
            params=Neurons.NEUCOGAR.value
        )

        self.add_neurons(Constants.NEURONS_IN_GROUP.value * 3)

    def connect_multimeters(self):
        self.right_group.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        self.left_group.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
        self.inhibitory_group.nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()

    def set_connections(self):
        # connect right group to left group
        self.right_group.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.left_group.nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.RL.value[::-1][self.get_index()-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

        # connect left group to right group
        self.left_group.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.right_group.nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.LR.value[::-1][self.get_index()-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

        # connect inhibitory group to right group
        self.inhibitory_group.nuclei(Neurotransmitters.GABA.value).connect(
            nucleus=self.right_group.nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GABAERGIC.value,
            weight=Weights.IR.value[::-1][self.get_index()-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

        # connect right group to inhibitory group
        self.right_group.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.inhibitory_group.nuclei(Neurotransmitters.GABA.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.RI.value[::-1][self.get_index()-1],
            conn_type=ConnectionTypes.ONE_TO_ONE.value
        )

    def connect(self, tier):
        """
        Connects this tier to a lower tier
        Args:
            tier (Tier): a lower tier
        Returns:
            bool

        """

        if self.index - tier.get_index() == 1:

            self.left_group.nuclei(Neurotransmitters.GLU.value).connect(
                nucleus=tier.get_left_group().nuclei(Neurotransmitters.GLU.value),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=Weights.LL.value[::-1][tier.get_index()-1],
                conn_type=ConnectionTypes.ONE_TO_ONE.value
            )

            tier.get_right_group().nuclei(Neurotransmitters.GLU.value).connect(
                nucleus=self.right_group.nuclei(Neurotransmitters.GLU.value),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=Weights.RR.value[::-1][tier.get_index()-2],
                conn_type=ConnectionTypes.ONE_TO_ONE.value
            )

            return True
        else:
            return False

    def get_index(self): return self.index

    def get_left_group(self): return self.left_group

    def get_right_group(self): return self.right_group

    def get_inhibitory_group(self): return self.inhibitory_group