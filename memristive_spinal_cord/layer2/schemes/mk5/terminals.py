from neucogar.Nucleus import Nucleus
from neucogar import api_kernel

from memristive_spinal_cord.layer2.models import ConnectionTypes
from memristive_spinal_cord.layer2.schemes.mk5.components.neurons import Neurons
from memristive_spinal_cord.layer2.schemes.mk5.components.parameters import Weights, Constants
from memristive_spinal_cord.layer2.schemes.mk5.components.synapses import Synapses


class Terminals:

    number_of_neurons = 2

    @classmethod
    def get_number_of_neurons(cls):
        return cls.number_of_neurons

    def __init__(self):
        self.mediator = Nucleus('Mediator')
        self.mediator.addSubNucleus(
            neurotransmitter='Glu',
            number=1,
            params=Neurons.NEUCOGAR.value
        )
        self.pool = Nucleus('InterneuronalPool')
        self.pool.addSubNucleus(
            neurotransmitter='Glu',
            number=1,
            params=Neurons.NEUCOGAR.value
        )

    def connect(self, input: Nucleus, output: list):
        self.mediator.nuclei('Glu').connect(
            nucleus=input.nuclei('Glu'),
            synapse=Synapses.GLUTAMATERGIC.value,
            weight=Weights.MR.value,
            conn_type=ConnectionTypes.ALL_TO_ALL.value
        )
        for i in range(len(output)):
            output[i].nuclei('Glu').connect(
                nucleus=self.pool.nuclei('Glu'),
                synapse=Synapses.GLUTAMATERGIC.value,
                weight=Weights.P.value[i],
                conn_type=ConnectionTypes.ALL_TO_ALL.value
            )

    def connect_multimeters(self):
        self.mediator.nuclei('Glu').ConnectMultimeter()
        self.pool.nuclei('Glu').ConnectMultimeter()

    def connect_spike_generator(self):
        spike_generator = api_kernel.NEST.Create('spike_generator', 1, {
            'spike_times': Constants.SPIKE_GENERATOR_TIMES.value,
            'spike_weights': Constants.SPIKE_GENERATOR_WEIGHTS.value
        })

        api_kernel.NEST.Connect(
            spike_generator,
            self.mediator.nuclei('Glu').getNeurons())

    def connect_noise_generator(self):
        self.mediator.nuclei('Glu').ConnectPoissonGenerator(
            weight=Weights.SG.value,
            start=1.,
            stop=5.,
            rate=100.
        )
