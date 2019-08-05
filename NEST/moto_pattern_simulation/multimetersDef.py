def multimeters_creater(nest, neuron_groups):
    multimeters_dict = dict()
    for i in neuron_groups:
        multimeter1 = nest.Create("multimeter")
        nest.SetStatus(multimeter1, {"withtime": True, "record_from": ["V_m", "g_in", "g_ex"], "interval": 0.025})
        multimeters_dict[i] = multimeter1

    return multimeters_dict


def spike_det_creater(nest, neuron_groups):
    spike_dec_dict = dict()
    for i in neuron_groups:
        spikedetector1 = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
        spike_dec_dict[i] = spikedetector1

    return spike_dec_dict


def multimeters_connector(nest, mult_dict, neurons_group_dict):
    for i in mult_dict:
        nest.Connect(mult_dict[i], neurons_group_dict[i])


def spike_det_connector(nest, spike_det_dict, neurons_group_dict):
    for i in spike_det_dict:
        nest.Connect(neurons_group_dict[i], spike_det_dict[i])