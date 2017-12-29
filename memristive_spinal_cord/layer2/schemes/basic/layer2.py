from memristive_spinal_cord.layer2.schemes.basic import Tier
from neucogar.Nucleus import Nucleus
from memristive_spinal_cord.layer2.models import Neurotransmitters
from memristive_spinal_cord.layer2.components.synapses import Synapses
from memristive_spinal_cord.layer2.components.parameters import Weights
from memristive_spinal_cord.layer2.models import ConnectionTypes


class Layer2:

    __structure = []

    def __init__(self):
        self.__structure = [Tier(i + 1) for i in range(6)]
        for higher_tier, lower_tier in zip(self.__structure[1:], self.__structure[:-1]):
            higher_tier.connect_to_tier(lower_tier)

    def connect_to_input(self, source: Nucleus):
        source.nuclei(Neurotransmitters.GLU.value).connect(
            nucleus=self.__structure[0].get_right_group().nuclei(Neurotransmitters.GLU.value),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.MR.value,
            conn_type=ConnectionTypes.ALL_TO_ALL.value
        )

    def connect_output_to_pool(self, pool: Nucleus):
        for tier in self.__structure:
            tier.get_left_group().nuclei(Neurotransmitters.GLU.value).connect(
                nucleus=pool.nuclei(Neurotransmitters.GLU.value),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=Weights.LIP.value[tier.get_tier()+1],
                conn_type=ConnectionTypes.ALL_TO_ALL.value
            )

    def connect_multimeters(self):
        for tier in self.__structure:
            tier.get_left_group().nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
            tier.get_right_group().nuclei(Neurotransmitters.GLU.value).ConnectMultimeter()
            if tier.get_inhibitory_group():
                tier.get_inhibitory_group().nuclei(Neurotransmitters.GABA.value).ConnectMultimeter()

    def connect_detectors(self):
        for tier in self.__structure:
            tier.get_left_group().nuclei(Neurotransmitters.GLU.value).ConnectDetector()
            tier.get_right_group().nuclei(Neurotransmitters.GLU.value).ConnectDetector()
            if tier.get_inhibitory_group():
                tier.get_inhibitory_group().nuclei(Neurotransmitters.GABA.value).ConnectDetector()