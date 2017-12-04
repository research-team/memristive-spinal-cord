from neucogar.Nucleus import Nucleus
from neucogar.api_kernel import CreateNetwork
import memristive_spinal_cord.layer1.params.neucogar as layer1_params


def create_neuron_group(group_name, group_params):
    """
    Function for building spike diagrams

    Args:
        group_name (str)
        group_params (NeuronGroupParameters)
    """

    neuron_group = Nucleus(group_name)
    neuron_group.addSubNucleus(group_params.get_type(),
                               params=group_params.get_model(),
                               number=group_params.get_number())
    return neuron_group


r_motor = create_neuron_group("R Motoneurons", layer1_params.motor_neurons)
l_motor = create_neuron_group("L Motoneurons", layer1_params.motor_neurons)

r_renshaw = create_neuron_group("R Renshaw", layer1_params.renshaw_neurons)
l_renshaw = create_neuron_group("L Renshaw", layer1_params.renshaw_neurons)

r_inter_1a = create_neuron_group("R 1A Interneurons", layer1_params.inter_neurons_1a)
l_inter_1a = create_neuron_group("L 1A Interneurons", layer1_params.inter_neurons_1a)

r_inter_1b = create_neuron_group("R 1B Interneurons", layer1_params.inter_neurons_1b)
l_inter_1b = create_neuron_group("L 1B Interneurons", layer1_params.inter_neurons_1b)

CreateNetwork(10000)
