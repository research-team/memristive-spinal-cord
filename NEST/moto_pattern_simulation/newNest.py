import nest

import nestData


class NeurGroupsContainer:
    def add_groups(self, group_name, value):
        setattr(self, group_name, value)


all_neurons = {}
all_generators = {}


def form_group(group_name, nrns_in_group):
    neur_type = 'hh_cond_exp_traub'
    ndict = {'t_ref': 3.0,
             'V_m': -70.0,
             'E_L': - 70.,
             'g_L': 75.0,
             'tau_syn_ex': 0.2,
             'tau_syn_in': 2.0,
             'C_m': 200.0}
    neurs = nest.Create(neur_type, nrns_in_group, ndict)
    all_neurons[group_name] = neurs
    return neurs


def much_groups_creater(groups_names, neur_num):
    n_list = [form_group(i, neur_num) for i in groups_names]
    return n_list


def spike_generator_creator(name, frequency, current):
    generator = nest.Create("spike_generator")
    nest.SetStatus(generator, {
        "spike_times": [(1.0 * i + 0.025) / frequency * 1000 for i in range(int(nestData.SIM_TIME / 1000 * frequency))],
        "spike_weights": [1.0 * current for i in range(int(nestData.SIM_TIME * frequency / 1000))]})
    all_generators[name] = generator


def poisson_generator_creator(name, origin, stop=26):
    generator = nest.Create("poisson_generator")

    nest.SetStatus(generator, {"rate": 8000.0, "origin": 1.0 * origin, "start": 1.0, "stop": 1.0 * stop})
    all_generators[name] = generator


def connect_fixed_outdegree(group1, group2, delay, weight, syn_outdegree=27):
    rule = {'rule': 'fixed_outdegree', 'outdegree': nestData.syn_outdegree}
    data1 = {'weight': weight, 'delay': delay}
    nest.Connect(group1, group2, rule, data1)
    pass


def connect_one_to_all(group1, group2, delay, weight):
    rule = {'rule': 'all_to_all'}
    data1 = {'weight': weight, 'delay': delay}
    nest.Connect(group1, group2, rule, data1)
    pass


def create_network():
    spike_generator_creator('EES', 40, 1.0)
    poisson_generator_creator('CV1', 1)
    poisson_generator_creator('CV2', 26)
    poisson_generator_creator('CV3', 51)
    poisson_generator_creator('CV4', 76, 51)
    poisson_generator_creator('CV5', 126)
    poisson_generator_creator('iIP_F', 151, 126)

    much_groups_creater(['E1', 'E2', 'E3', 'E4', 'E5'], nestData.neurons_in_group)
    much_groups_creater(['CD4', 'CD5'], 1)
    much_groups_creater(['OM1_0', 'OM1_1', 'OM1_2_E', 'OM1_2_F', 'OM1_3'], nestData.syn_outdegree)
    much_groups_creater(['OM2_0', 'OM2_1', 'OM2_2_E', 'OM2_2_F', 'OM2_3'], nestData.syn_outdegree)
    much_groups_creater(['OM3_0', 'OM3_1', 'OM3_2_E', 'OM3_2_F', 'OM3_3'], nestData.syn_outdegree)
    much_groups_creater(['OM4_0', 'OM4_1', 'OM4_2_E', 'OM4_2_F', 'OM4_3'], nestData.syn_outdegree)
    much_groups_creater(['OM5_0', 'OM5_1', 'OM5_2_E', 'OM5_2_F', 'OM5_3'], nestData.syn_outdegree)
    much_groups_creater(['MN_E', 'MN_F'], nestData.neurons_in_moto)
    much_groups_creater(['Ia_E_aff', 'Ia_F_aff'], nestData.neurons_in_afferent)
    much_groups_creater(['R_E', 'R_F'], nestData.neurons_in_group)
    much_groups_creater(['Ia_E_pool', 'Ia_F_pool'], nestData.neurons_in_aff_ip)
    much_groups_creater(['eIP_E', 'eIP_F'], nestData.neurons_in_ip)
    much_groups_creater(['iIP_E', 'iIP_F'], nestData.neurons_in_ip)

    connect_one_to_all(all_generators['EES'], all_neurons['E1'], 1, 370)
    connect_fixed_outdegree(all_neurons['E1'], all_neurons['E2'], 1, 80, nestData.syn_outdegree)
    connect_fixed_outdegree(all_neurons['E2'], all_neurons['E3'], 1, 80, nestData.syn_outdegree)
    connect_fixed_outdegree(all_neurons['E3'], all_neurons['E4'], 1, 80, nestData.syn_outdegree)
    connect_fixed_outdegree(all_neurons['E4'], all_neurons['E5'], 1, 80, nestData.syn_outdegree)

    connect_one_to_all(all_generators['CV1'], all_neurons['iIP_E'], 0.5, 20)
    connect_one_to_all(all_generators['CV2'], all_neurons['iIP_E'], 0.5, 20)
    connect_one_to_all(all_generators['CV3'], all_neurons['iIP_E'], 0.5, 20)
    connect_one_to_all(all_generators['CV4'], all_neurons['iIP_E'], 0.5, 20)
    connect_one_to_all(all_generators['CV5'], all_neurons['iIP_E'], 0.5, 20)

    connect_fixed_outdegree(all_neurons['E1'], all_neurons['OM1_0'], 1, 20, nestData.syn_outdegree)

    connect_one_to_all(all_generators['CV1'], all_neurons['OM1_0'], 0.5, 4)
    connect_one_to_all(all_generators['CV2'], all_neurons['OM1_0'], 0.5, 4)

    connect_one_to_all(all_generators['CV3'], all_neurons['OM1_3'], 1, 80)
    connect_one_to_all(all_generators['CV4'], all_neurons['OM1_3'], 1, 80)
    connect_one_to_all(all_generators['CV5'], all_neurons['OM1_3'], 1, 80)

    connect_fixed_outdegree(all_neurons['OM1_0'], all_neurons['OM1_1'], 1, 30)
    connect_fixed_outdegree(all_neurons['OM1_1'], all_neurons['OM1_2_E'], 1, 20)
    connect_fixed_outdegree(all_neurons['OM1_1'], all_neurons['OM1_2_F'], 0.1, 20)
    connect_fixed_outdegree(all_neurons['OM1_1'], all_neurons['OM1_3'], 3.5, 4)
    connect_fixed_outdegree(all_neurons['OM1_2_E'], all_neurons['OM1_1'], 2.5, 18.3)
    connect_fixed_outdegree(all_neurons['OM1_2_F'], all_neurons['OM1_1'], 2.5, 16)
    connect_fixed_outdegree(all_neurons['OM1_2_E'], all_neurons['OM1_3'], 1, 2)
    connect_fixed_outdegree(all_neurons['OM1_2_F'], all_neurons['OM1_3'], 0.3, 15.5)
    connect_fixed_outdegree(all_neurons['OM1_3'], all_neurons['OM1_1'], 1.5, -3)
    connect_fixed_outdegree(all_neurons['OM1_3'], all_neurons['OM1_2_E'], 0.3, -60)
    connect_fixed_outdegree(all_neurons['OM1_3'], all_neurons['OM1_2_F'], 1, -1)

    connect_fixed_outdegree(all_neurons['OM1_2_F'], all_neurons['OM2_2_F'], 1, 50)

    connect_fixed_outdegree(all_neurons['OM1_2_E'], all_neurons['eIP_E'], 1, 45, nestData.neurons_in_ip)
    connect_fixed_outdegree(all_neurons['OM1_2_F'], all_neurons['eIP_F'], 1, 30, nestData.neurons_in_ip)

    connect_fixed_outdegree(all_neurons['E2'], all_neurons['OM2_0'], 1, 12)

    connect_one_to_all(all_generators['CV2'], all_neurons['OM2_0'], 0.5, 5)
    connect_one_to_all(all_generators['CV3'], all_neurons['OM2_0'], 0.5, 5)

    connect_one_to_all(all_generators['CV4'], all_neurons['OM2_3'], 1, 80)
    connect_one_to_all(all_generators['CV5'], all_neurons['OM2_3'], 1, 80)

    connect_fixed_outdegree(all_neurons['OM2_0'], all_neurons['OM2_1'], 1, 27)
    connect_fixed_outdegree(all_neurons['OM2_1'], all_neurons['OM2_2_E'], 1, 21.5)
    connect_fixed_outdegree(all_neurons['OM2_1'], all_neurons['OM2_2_F'], 0.1, 15.5)
    connect_fixed_outdegree(all_neurons['OM2_1'], all_neurons['OM2_3'], 3.5, 4)
    connect_fixed_outdegree(all_neurons['OM2_2_E'], all_neurons['OM2_1'], 2.5, 17.5)
    connect_fixed_outdegree(all_neurons['OM2_2_F'], all_neurons['OM2_1'], 2.5, 16)
    connect_fixed_outdegree(all_neurons['OM2_2_E'], all_neurons['OM2_3'], 1, 2)
    connect_fixed_outdegree(all_neurons['OM2_2_F'], all_neurons['OM2_3'], 0.3, 12)
    connect_fixed_outdegree(all_neurons['OM2_3'], all_neurons['OM2_1'], 1.5, -3)
    connect_fixed_outdegree(all_neurons['OM2_3'], all_neurons['OM2_2_E'], 0.3, -20)
    connect_fixed_outdegree(all_neurons['OM2_3'], all_neurons['OM2_2_F'], 1, -1)

    connect_fixed_outdegree(all_neurons['OM2_2_F'], all_neurons['OM3_2_F'], 1, 50)

    connect_fixed_outdegree(all_neurons['OM2_2_E'], all_neurons['eIP_E'], 3, 30, nestData.neurons_in_ip)
    connect_fixed_outdegree(all_neurons['OM2_2_F'], all_neurons['eIP_F'], 2, 30, nestData.neurons_in_ip)

    connect_fixed_outdegree(all_neurons['E3'], all_neurons['OM3_0'], 1, 12)

    connect_one_to_all(all_generators['CV3'], all_neurons['OM3_0'], 0.5, 5)
    connect_one_to_all(all_generators['CV4'], all_neurons['OM3_0'], 0.5, 5)

    connect_one_to_all(all_generators['CV5'], all_neurons['OM3_3'], 1, 80)

    connect_one_to_all(all_neurons['CD4'], all_neurons['OM3_0'], 1, 11)

    connect_fixed_outdegree(all_neurons['OM3_0'], all_neurons['OM3_1'], 1, 35)
    connect_fixed_outdegree(all_neurons['OM3_1'], all_neurons['OM3_2_E'], 1, 21.4)
    connect_fixed_outdegree(all_neurons['OM3_1'], all_neurons['OM3_2_F'], 0.1, 15.5)
    connect_fixed_outdegree(all_neurons['OM3_1'], all_neurons['OM3_3'], 3.5, 3)
    connect_fixed_outdegree(all_neurons['OM3_2_E'], all_neurons['OM3_1'], 2.5, 17)
    connect_fixed_outdegree(all_neurons['OM3_2_F'], all_neurons['OM3_1'], 2.5, 16)
    connect_fixed_outdegree(all_neurons['OM3_2_E'], all_neurons['OM3_3'], 1, 2)
    connect_fixed_outdegree(all_neurons['OM3_2_F'], all_neurons['OM3_3'], 0.3, 12)
    connect_fixed_outdegree(all_neurons['OM3_3'], all_neurons['OM3_1'], 1.5, -3)
    connect_fixed_outdegree(all_neurons['OM3_3'], all_neurons['OM3_2_E'], 0.3, -24)
    connect_fixed_outdegree(all_neurons['OM3_3'], all_neurons['OM3_2_F'], 1, -2)

    connect_fixed_outdegree(all_neurons['OM3_2_F'], all_neurons['OM4_2_F'], 1, 50)
    connect_fixed_outdegree(all_neurons['OM3_2_E'], all_neurons['eIP_E'], 2, 28, nestData.neurons_in_ip)
    connect_fixed_outdegree(all_neurons['OM3_2_F'], all_neurons['eIP_F'], 3, 30, nestData.neurons_in_ip)

    connect_fixed_outdegree(all_neurons['E4'], all_neurons['OM4_0'], 2, 12)

    connect_one_to_all(all_generators['CV4'], all_neurons['OM4_0'], 0.5, 5)
    connect_one_to_all(all_generators['CV5'], all_neurons['OM4_0'], 0.5, 5)

    connect_one_to_all(all_neurons['CD4'], all_neurons['OM4_0'], 1, 11)
    connect_one_to_all(all_neurons['CD5'], all_neurons['OM4_0'], 1, 11)

    connect_fixed_outdegree(all_neurons['OM4_0'], all_neurons['OM4_1'], 1, 30)
    connect_fixed_outdegree(all_neurons['OM4_1'], all_neurons['OM4_2_E'], 1, 21.5)
    connect_fixed_outdegree(all_neurons['OM4_1'], all_neurons['OM4_2_F'], 0.1, 13)
    connect_fixed_outdegree(all_neurons['OM4_1'], all_neurons['OM4_3'], 3.5, 4)
    connect_fixed_outdegree(all_neurons['OM4_2_E'], all_neurons['OM4_1'], 2.5, 18.3)
    connect_fixed_outdegree(all_neurons['OM4_2_F'], all_neurons['OM4_1'], 2.5, 16)
    connect_fixed_outdegree(all_neurons['OM4_2_E'], all_neurons['OM4_3'], 1, 2)
    connect_fixed_outdegree(all_neurons['OM4_2_F'], all_neurons['OM4_3'], 0.3, 12)
    connect_fixed_outdegree(all_neurons['OM4_3'], all_neurons['OM4_1'], 1.5, -3)
    connect_fixed_outdegree(all_neurons['OM4_3'], all_neurons['OM4_2_E'], 0.3, -23)
    connect_fixed_outdegree(all_neurons['OM4_3'], all_neurons['OM4_2_F'], 1, -2)

    connect_fixed_outdegree(all_neurons['OM4_2_F'], all_neurons['OM5_2_F'], 1, 50)
    connect_fixed_outdegree(all_neurons['OM4_2_E'], all_neurons['eIP_E'], 2, 35, nestData.neurons_in_ip)
    connect_fixed_outdegree(all_neurons['OM4_2_F'], all_neurons['eIP_F'], 1, 30, nestData.neurons_in_ip)

    connect_fixed_outdegree(all_neurons['E5'], all_neurons['OM5_0'], 3, 12)

    connect_one_to_all(all_generators['CV5'], all_neurons['OM5_0'], 0.5, 5)

    connect_one_to_all(all_neurons['CD5'], all_neurons['OM5_0'], 1, 11)

    connect_fixed_outdegree(all_neurons['OM5_0'], all_neurons['OM5_1'], 1, 30)
    connect_fixed_outdegree(all_neurons['OM5_1'], all_neurons['OM5_2_E'], 1, 20.5)
    connect_fixed_outdegree(all_neurons['OM5_1'], all_neurons['OM5_2_F'], 0.1, 15.5)
    connect_fixed_outdegree(all_neurons['OM5_1'], all_neurons['OM5_3'], 3.5, 4)
    connect_fixed_outdegree(all_neurons['OM5_2_E'], all_neurons['OM5_1'], 2.5, 17.5)
    connect_fixed_outdegree(all_neurons['OM5_2_F'], all_neurons['OM5_1'], 2.5, 16)
    connect_fixed_outdegree(all_neurons['OM5_2_E'], all_neurons['OM5_3'], 1, 2)
    connect_fixed_outdegree(all_neurons['OM5_2_F'], all_neurons['OM5_3'], 0.3, 15)
    connect_fixed_outdegree(all_neurons['OM5_3'], all_neurons['OM5_1'], 1.5, -3)
    connect_fixed_outdegree(all_neurons['OM5_3'], all_neurons['OM5_2_E'], 0.3, -20)
    connect_fixed_outdegree(all_neurons['OM5_3'], all_neurons['OM5_2_F'], 1, -2)

    connect_fixed_outdegree(all_neurons['OM5_2_E'], all_neurons['eIP_E'], 2, 30, nestData.neurons_in_ip)
    connect_fixed_outdegree(all_neurons['OM5_2_F'], all_neurons['eIP_F'], 3, 30, nestData.neurons_in_ip)

    connect_fixed_outdegree(all_neurons['iIP_E'], all_neurons['eIP_F'], 0.5, -50, nestData.neurons_in_ip)
    # connect_fixed_outdegree(all_neurons['iIP_F'], all_neurons['eIP_E'], 0.5, -50, nestData.neurons_in_ip)
    connect_one_to_all(all_generators['iIP_F'], all_neurons['eIP_E'], 0.5, -50)

    connect_fixed_outdegree(all_neurons['iIP_E'], all_neurons['OM1_2_F'], 0.5, -500, nestData.neurons_in_ip)
    connect_fixed_outdegree(all_neurons['iIP_E'], all_neurons['OM2_2_F'], 0.5, -100, nestData.neurons_in_ip)
    connect_fixed_outdegree(all_neurons['iIP_E'], all_neurons['OM3_2_F'], 0.5, -100, nestData.neurons_in_ip)
    connect_fixed_outdegree(all_neurons['iIP_E'], all_neurons['OM4_2_F'], 0.5, -100, nestData.neurons_in_ip)

    connect_one_to_all(all_generators['EES'], all_neurons['Ia_E_aff'], 1, 500)
    connect_one_to_all(all_generators['EES'], all_neurons['Ia_F_aff'], 1, 500)

    connect_fixed_outdegree(all_neurons['eIP_E'], all_neurons['MN_E'], 2.5, 400, nestData.neurons_in_moto)

    connect_fixed_outdegree(all_neurons['eIP_F'], all_neurons['MN_F'], 1, 200, nestData.neurons_in_moto)

    connect_fixed_outdegree(all_neurons['iIP_E'], all_neurons['Ia_E_pool'], 1, 30, nestData.neurons_in_ip)
    connect_one_to_all(all_generators['iIP_F'], all_neurons['Ia_F_pool'], 1, 30)
    # Возможно нужно заменить на connect_fixed_outdegree

    connect_fixed_outdegree(all_neurons['Ia_E_pool'], all_neurons['MN_F'], 1, -150, nestData.neurons_in_ip)
    connect_fixed_outdegree(all_neurons['Ia_E_pool'], all_neurons['Ia_F_pool'], 1, -1, nestData.neurons_in_ip)
    connect_fixed_outdegree(all_neurons['Ia_F_pool'], all_neurons['MN_E'], 1, -50, nestData.neurons_in_ip)
    connect_fixed_outdegree(all_neurons['Ia_F_pool'], all_neurons['Ia_E_pool'], 1, -1, nestData.neurons_in_ip)

    connect_fixed_outdegree(all_neurons['Ia_E_aff'], all_neurons['MN_E'], 2, 30, nestData.neurons_in_moto)
    connect_fixed_outdegree(all_neurons['Ia_F_aff'], all_neurons['MN_F'], 2, 22, nestData.neurons_in_moto)

    connect_fixed_outdegree(all_neurons['MN_E'], all_neurons['R_E'], 2, 4)
    connect_fixed_outdegree(all_neurons['MN_F'], all_neurons['R_F'], 0.5, 6)

    connect_fixed_outdegree(all_neurons['R_E'], all_neurons['MN_E'], 2, -10, nestData.neurons_in_moto)
    connect_fixed_outdegree(all_neurons['R_E'], all_neurons['R_F'], 2, -1)

    connect_fixed_outdegree(all_neurons['R_F'], all_neurons['MN_F'], 0.5, -50, nestData.neurons_in_moto)
    connect_fixed_outdegree(all_neurons['R_F'], all_neurons['R_E'], 2, -1)

    return all_neurons