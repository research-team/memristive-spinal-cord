import pylab

import nest


def neuron_groups_generator(nest):
    ndict = {"I_e": 0.0, "C_m": 200.0, "E_L": -70.0, "g_L": 75.0, "V_m": -70.0, 't_ref': 3.0}

    all_neurons = dict()
    all_neurons['ilP_F'] = nest.Create("hh_cond_exp_traub", 20, ndict)  # круги
    all_neurons['la_F'] = nest.Create("hh_cond_exp_traub", 20, ndict)
    all_neurons['R_F'] = nest.Create("hh_cond_exp_traub", 20, ndict)
    all_neurons['elP_F'] = nest.Create("hh_cond_exp_traub", 20, ndict)
    all_neurons['ilP_E'] = nest.Create("hh_cond_exp_traub", 20, ndict)
    all_neurons['la_E'] = nest.Create("hh_cond_exp_traub", 20, ndict)
    all_neurons['R_E'] = nest.Create("hh_cond_exp_traub", 20, ndict)
    all_neurons['elP_E'] = nest.Create("hh_cond_exp_traub", 20, ndict)

    all_neurons['MP_F'] = nest.Create("hh_cond_exp_traub", 169, ndict)  # ромбы
    all_neurons['MP_E'] = nest.Create("hh_cond_exp_traub", 169, ndict)

    all_neurons['la_1'] = nest.Create("hh_cond_exp_traub", 196, ndict)  # цилиндры
    all_neurons['la_2'] = nest.Create("hh_cond_exp_traub", 196, ndict)

    return all_neurons


def spike_generators(nest, time, hg):
    # time в миллисекундах
    # hg герц
    noise_ex1 = nest.Create("spike_generator")
    noise_ex2 = nest.Create("spike_generator")

    times = [1.0 * i * 1 / hg * 1000 for i in range(int(time / 1000 * hg)) if i != 0]
    weights = [5.0 * i for i in range(int(time / 1000 * hg)) if i != 0]

    nest.SetStatus(noise_ex1,
                   {"spike_times": times,
                    "spike_weights": weights})

    nest.SetStatus(noise_ex2,
                   {"spike_times": times,
                    "spike_weights": weights})

    return noise_ex1, noise_ex2


def neur_groups_connect(nest, noise_ex1, noise_ex2, neuron_groups):
    ndict_exc = {'weight': 12.0, 'delay': 1.0}
    ndict_inh = {'weight': -20.0, 'delay': 1.0}

    nest.Connect(noise_ex1, neuron_groups['la_1'], "all_to_all", ndict_exc)
    nest.Connect(noise_ex2, neuron_groups['la_2'], "all_to_all", ndict_exc)

    nest.Connect(neuron_groups['la_1'], neuron_groups['MP_F'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_exc)
    nest.Connect(neuron_groups['la_1'], neuron_groups['la_F'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_exc)
    nest.Connect(neuron_groups['MP_F'], neuron_groups['R_F'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_exc)
    nest.Connect(neuron_groups['la_F'], neuron_groups['ilP_F'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_exc)
    nest.Connect(neuron_groups['elP_F'], neuron_groups['MP_F'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_exc)
    nest.Connect(neuron_groups['elP_F'], neuron_groups['la_F'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_exc)

    nest.Connect(neuron_groups['la_2'], neuron_groups['MP_E'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_exc)
    nest.Connect(neuron_groups['la_2'], neuron_groups['la_E'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_exc)
    nest.Connect(neuron_groups['MP_E'], neuron_groups['R_E'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_exc)
    nest.Connect(neuron_groups['la_E'], neuron_groups['ilP_E'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_exc)
    nest.Connect(neuron_groups['elP_E'], neuron_groups['MP_E'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_exc)
    nest.Connect(neuron_groups['elP_E'], neuron_groups['la_E'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_exc)

    nest.Connect(neuron_groups['ilP_F'], neuron_groups['elP_E'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_inh)
    nest.Connect(neuron_groups['la_F'], neuron_groups['MP_E'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_inh)
    nest.Connect(neuron_groups['la_F'], neuron_groups['la_E'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_inh)
    nest.Connect(neuron_groups['R_F'], neuron_groups['la_F'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_inh)
    nest.Connect(neuron_groups['R_F'], neuron_groups['MP_F'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_inh)
    nest.Connect(neuron_groups['R_F'], neuron_groups['R_E'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_inh)

    nest.Connect(neuron_groups['ilP_E'], neuron_groups['elP_F'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_inh)
    nest.Connect(neuron_groups['la_E'], neuron_groups['MP_F'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_inh)
    nest.Connect(neuron_groups['la_E'], neuron_groups['la_F'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_inh)
    nest.Connect(neuron_groups['R_E'], neuron_groups['la_E'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_inh)
    nest.Connect(neuron_groups['R_E'], neuron_groups['MP_E'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_inh)
    nest.Connect(neuron_groups['R_E'], neuron_groups['R_F'], {'rule': 'fixed_outdegree', 'outdegree': 27}, ndict_inh)


def multimeters_creater(nest, neuron_groups):
    multimeters_dict = dict()
    for i in neuron_groups:
        multimeter1 = nest.Create("multimeter")
        nest.SetStatus(multimeter1, {"withtime": True, "record_from": ["V_m"]})
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


def graph_volts_data_creator(nest, multimeter, neurons_groups):
    volts_data_moving = []

    dmm = nest.GetStatus(multimeter)[0]
    vms = dmm['events']["V_m"]
    ts = dmm["events"]["times"]
    ts = ts[::len(neurons_groups)]
    n_len = len(neurons_groups)
    for i in range(0, len(vms), n_len):  # среднее арифметическое значений
        volts_data_moving.append(sum(vms[i:i+n_len])/n_len)

    return volts_data_moving, ts


def graph_spike_data_creator(nest, spike_det, neurons_groups):
    spike_time = []

    dSD = nest.GetStatus(spike_det)[0]
    spike_evs = dSD['events']["senders"]
    ts = dSD['events']["times"]
    # ts = ts[::len(neurons_groups)]
    n_len = len(neurons_groups)

    s = 0
    last = 0
    for i in range(len(ts)):
        i = int(ts[i])
        if i == last:
            s += 1
        else:
            last = i
            if s > n_len/2:
                spike_time.append(i)
            s = 0

    spike_data_moving = [0 for i in spike_time]
    return spike_data_moving, spike_time

hg = 40.0  # герц
all_time = 100.0

pics_folder = 'pics(volts)/'

all_neurons = neuron_groups_generator(nest)
noise_ex1, noise_ex2 = spike_generators(nest, all_time, hg)

neur_groups_connect(nest, noise_ex1, noise_ex2, all_neurons)

multimeters = multimeters_creater(nest, all_neurons)
spike_detectors = spike_det_creater(nest, all_neurons)

multimeters_connector(nest, multimeters, all_neurons)
spike_det_connector(nest, spike_detectors, all_neurons)
nest.Simulate(all_time)

for i in all_neurons:
    volts_data_moving, ts = graph_volts_data_creator(nest, multimeters[i], all_neurons)
    spike_data_moving, ts2 = graph_spike_data_creator(nest, spike_detectors[i], all_neurons)

    pylab.figure(1)
    pylab.plot(ts, volts_data_moving)
    pylab.plot(ts2, spike_data_moving, '.', color='r')
    pylab.savefig(pics_folder + i, fmt='png')
    pylab.show()

# # print(MP_F)
# nest.Connect(multimeter1, [MP_F[0]])
# nest.Connect([MP_F[0]], spikedetector1)
#
# # nest.Connect(voltmeter1, elP_F)
#
# nest.Simulate(SIM_TIME)
#
# dmm = nest.GetStatus(multimeter1)[0]
# Vms = dmm['events']["V_m"]
# ts = dmm["events"]["times"]
#
# import pylab
# pylab.figure(1)
# pylab.plot(ts, Vms)
# # pylab.show()
#
#
# dSD = nest . GetStatus (spikedetector1 , keys = "events" )
# print(len(dSD))
# print(dSD)
# dSD = dSD[0]
# for i in range(len(dSD['senders'])):
#     dSD['senders'][i] = 0
# print(dSD)
# print('____')
# print(dSD)
# evs = dSD [ "senders" ]
# ts = dSD [ "times" ]
# pylab . figure ( 1 )
# pylab . plot ( ts , evs , ".", color ='r' )
# pylab . show ()

# nest.voltage_trace.from_device(voltmeter1)
# print(nest.GetStatus(voltmeter1))
# nest.voltage_trace.show()