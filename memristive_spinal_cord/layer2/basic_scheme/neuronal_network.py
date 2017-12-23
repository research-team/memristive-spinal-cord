from memristive_spinal_cord.layer2.basic_scheme.layer2 import Layer2
from neucogar.Nucleus import Nucleus
from memristive_spinal_cord.layer2.models import Neurotransmitters
from memristive_spinal_cord.layer2.basic_scheme.neurons import Neurons
from neucogar import api_kernel
from memristive_spinal_cord.layer2.basic_scheme.parameters import Constants


def connect_spike_generator(target: Nucleus):
    spike_generator = api_kernel.NEST.Create(
        'spike_generator',
        1,
        {
            'spike_times': Constants.SPIKE_GENERATOR_TIMES.value,
            'spike_weights': Constants.SPIKE_GENERATOR_WEIGHTS.value
        }
    )

    api_kernel.NEST.Connect(
        spike_generator,
        target.nuclei(Neurotransmitters.GLU.value).getNeurons()
    )


class NeuronalNetwork:

    def __init__(self):
        layer2 = Layer2()

        mediator = Nucleus(nucleus_name='Mediator')
        mediator.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=1,
            params=Neurons.NEUCOGAR.value
        )
        mediator.ConnectMultimeter()
        mediator.ConnectDetector()
        connect_spike_generator(mediator)
        layer2.connect_to_input(mediator)

        interneuronal_pool = Nucleus(nucleus_name='Interneuronal_Pool')
        interneuronal_pool.addSubNucleus(
            neurotransmitter=Neurotransmitters.GLU.value,
            number=1,
            params=Neurons.NEUCOGAR.value
        )
        interneuronal_pool.ConnectMultimeter()
        interneuronal_pool.ConnectDetector()
        layer2.connect_output_to_pool(interneuronal_pool)
