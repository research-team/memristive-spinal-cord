import nest
import nest
import os


from reflex_arc.nest.Cheme_2.pl_2 import plot
from reflex_arc.nest.Cheme_2.Receptor import DummySensoryReceptor
import shutil

nest.ResetKernel()

nest.SetKernelStatus({"print_time": True,
                     "local_num_threads": 4,
                      "resolution": 0.1})

T = 100.

glu_weight = 10.
# glu_weight_1 = 15.
gaba_weight = -10.
gaba_weight_1 = 1.
static_weight = 60.
gen_rate = 60.

glu = {'model': 'static_synapse',
        'delay': 1.,
        'weight': glu_weight}
# glu_1 = {'model': 'static_synapse',
#         'delay': 1.,
#         'weight': glu_weight_1}
gaba = {'model': 'static_synapse',
        'delay': 1.,
        'weight': gaba_weight}
gaba_1 = {'model': 'static_synapse',
        'delay': 1.,
        'weight': gaba_weight_1}
static_syn = {'weight': static_weight,
              'delay': 1.}
nrn_parameters = {
    't_ref': 7.,  # Refractory period
    'V_m': -70.0,  #
    'E_L': -70.0,  #
    'E_K': -77.0,  #
    'g_L': 30.0,  #
    'g_Na': 12000.0,  #
    'g_K': 3600.0,  #
    'C_m': 134.0,  # Capacity of membrane (pF)
    'tau_syn_ex': 0.2,  # Time of excitatory action (ms)
    'tau_syn_in': 2.0}  # Time of inhibitory action (ms)


#Create neurons
#Sensory
S_0 = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)
S_l = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)
S_t = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)
S_h = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)
S_r = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)
S_1 = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)
#Afferent
I_a_0 = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)
I_I_0 = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)
I_I_1 = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)
I_a_1 = nest.Create("hh_cond_exp_traub", 60, nrn_parameters)
#Moto
Mn_L = nest.Create("hh_cond_exp_traub", 169, nrn_parameters)
Mn_E = nest.Create("hh_cond_exp_traub", 169, nrn_parameters)
Mn_F = nest.Create("hh_cond_exp_traub", 169, nrn_parameters)
Mn_R = nest.Create("hh_cond_exp_traub", 169, nrn_parameters)
#Excitatory
Ex_L = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
Ex_E = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
Ex_F = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
Ex_R = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
Ex_0 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
Ex_1 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
Ex_2 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
Ex_3 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
Ex_4 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
#Inhibitory
In_0 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
In_1 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
In_2 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
In_3 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
In_4 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
In_5 = nest.Create("hh_cond_exp_traub", 196, nrn_parameters)
#Noc
Noc = nest.Create("hh_cond_exp_traub", 20, nrn_parameters)


#Create multimeter
#Sensory
mm_S_0 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/s_0'})
mm_S_l = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/s_l'})
mm_S_t = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/s_t'})
mm_S_1 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/s_1'})
mm_S_r = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/s_r'})
mm_S_h = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/s_h'})
#Afferent
mm_I_a_0 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/i_a_0'})
mm_I_I_0 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/i_i_0'})
mm_I_a_1 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/i_a_1'})
mm_I_I_1 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/i_i_1'})
#Moto
mm_Mn_L = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/mn_l'})
mm_Mn_E = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/mn_e'})
mm_Mn_R = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/mn_r'})
mm_Mn_F = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/mn_f'})
#Excitatory
mm_Ex_L = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ex_l'})
mm_Ex_E = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ex_e'})
mm_Ex_R = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ex_r'})
mm_Ex_F = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ex_f'})
mm_Ex_0 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ex_0'})
mm_Ex_1 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ex_1'})
mm_Ex_2 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ex_2'})
mm_Ex_3 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ex_3'})
mm_Ex_4 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/ex_4'})
#Inhibitory
mm_In_0 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/in_0'})
mm_In_1 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/in_1'})
mm_In_2 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/in_2'})
mm_In_3 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/in_3'})
mm_In_4 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/in_4'})
mm_In_5 = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/in_5'})
#Noc
mm_Noc = nest.Create("multimeter", 1, {'record_from': ['V_m'], "interval": 0.1, 'to_file': True, 'label':
                        'results/noc'})


#Connections neurons
#Sensory
nest.Connect(pre=S_0, post=Ex_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=S_l, post=Ex_L, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=S_l, post=In_4, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=S_t, post=In_2, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=S_h, post=In_3, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=S_r, post=Ex_R, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=S_r, post=In_5, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=S_1, post=Ex_4, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=S_1, post=Ex_0, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
#Afferent
nest.Connect(pre=I_a_0, post=Mn_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=I_a_0, post=In_0, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=I_I_0, post=Ex_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=I_I_0, post=In_0, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=I_I_1, post=Ex_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=I_I_1, post=In_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=I_a_1, post=Mn_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
nest.Connect(pre=I_a_1, post=In_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 60},
             syn_spec=glu)
#Excitatory
nest.Connect(pre=Ex_0, post=Noc, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=Ex_L, post=Mn_L, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)
nest.Connect(pre=Ex_E, post=Mn_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)
nest.Connect(pre=Ex_F, post=Mn_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)
nest.Connect(pre=Ex_R, post=Mn_R, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)
nest.Connect(pre=Ex_1, post=Ex_2, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)
nest.Connect(pre=Ex_1, post=Ex_3, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba_1)
nest.Connect(pre=Ex_2, post=In_0, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)
nest.Connect(pre=Ex_2, post=Mn_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)
nest.Connect(pre=Ex_2, post=Ex_3, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=Ex_3, post=In_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)
nest.Connect(pre=Ex_3, post=Mn_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)
nest.Connect(pre=Ex_3, post=Ex_2, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=Ex_4, post=Ex_3, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)
nest.Connect(pre=Ex_4, post=Ex_2, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba_1)
#Inhibitory
nest.Connect(pre=In_0, post=Ex_4, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)
nest.Connect(pre=In_0, post=In_1, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=In_0, post=Mn_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=In_1, post=Ex_4, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=glu)
nest.Connect(pre=In_1, post=In_0, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=In_1, post=Mn_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=In_2, post=In_3, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=In_2, post=Mn_F, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=In_3, post=In_2, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=In_3, post=Mn_E, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=In_4, post=In_5, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=In_4, post=Mn_R, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=In_5, post=In_4, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
nest.Connect(pre=In_5, post=Mn_L, conn_spec={'rule': 'fixed_indegree', 'indegree': 196},
             syn_spec=gaba)
#Noc
nest.Connect(pre=Noc, post=Ex_3, conn_spec={'rule': 'fixed_indegree', 'indegree': 20},
             syn_spec=glu)
nest.Connect(pre=Noc, post=Ex_2, conn_spec={'rule': 'fixed_indegree', 'indegree': 20},
             syn_spec=gaba)


#Create generator
time_between_spikes = 1000 / gen_rate  # time between spikes
spike_times = [round(4.5 + i * time_between_spikes, 1) for i in range(int(T / time_between_spikes))]
generators = nest.Create("spike_generator", 1, {'spike_times': spike_times,
                                                'spike_weights': [10.0 for i in spike_times]})
generator_1 = nest.Create("spike_generator", 1, {'spike_times': spike_times,
                                                'spike_weights': [10.0 for i in spike_times]})
print(spike_times)


receptor_left = DummySensoryReceptor(stand_coef=0.5, inversion=True)
nest.Connect(pre=receptor_left.receptor_id, post=S_l)

receptor_toe = DummySensoryReceptor(stand_coef=0.3, inversion=True)
nest.Connect(pre=receptor_toe.receptor_id, post=S_t)

receptor_heel = DummySensoryReceptor(stand_coef=0.3, inversion=False)
nest.Connect(pre=receptor_heel.receptor_id, post=S_h)

receptor_right = DummySensoryReceptor(stand_coef=0.5, inversion=False)
nest.Connect(pre=receptor_right.receptor_id, post=S_r)

receptor_noc = DummySensoryReceptor(stand_coef=0.3, inversion=False)
nest.Connect(pre=receptor_noc.receptor_id, post=Noc)


# Connect generator
nest.Connect(pre=generators, post=S_0, syn_spec=static_syn)
nest.Connect(pre=generators, post=I_a_0, syn_spec=static_syn)
nest.Connect(pre=generators, post=I_I_0, syn_spec=static_syn)
nest.Connect(pre=generators, post=I_I_1, syn_spec=static_syn)
nest.Connect(pre=generators, post=I_a_1, syn_spec=static_syn)
nest.Connect(pre=generators, post=S_1, syn_spec=static_syn)


#Connect multimeter
#Sensory
nest.Connect(pre=mm_S_0, post=S_0)
nest.Connect(pre=mm_S_l, post=S_l)
nest.Connect(pre=mm_S_t, post=S_t)
nest.Connect(pre=mm_S_h, post=S_h)
nest.Connect(pre=mm_S_r, post=S_r)
nest.Connect(pre=mm_S_1, post=S_1)
#Afferent
nest.Connect(pre=mm_I_a_0, post=I_a_0)
nest.Connect(pre=mm_I_I_0, post=I_I_0)
nest.Connect(pre=mm_I_a_1, post=I_a_1)
nest.Connect(pre=mm_I_I_1, post=I_I_1)
#Moto
nest.Connect(pre=mm_Mn_L, post=Mn_L)
nest.Connect(pre=mm_Mn_E, post=Mn_E)
nest.Connect(pre=mm_Mn_F, post=Mn_F)
nest.Connect(pre=mm_Mn_R, post=Mn_R)
#Excitatory
nest.Connect(pre=mm_Ex_L, post=Ex_L)
nest.Connect(pre=mm_Ex_E, post=Ex_E)
nest.Connect(pre=mm_Ex_F, post=Ex_F)
nest.Connect(pre=mm_Ex_R, post=Ex_R)
nest.Connect(pre=mm_Ex_0, post=Ex_0)
nest.Connect(pre=mm_Ex_1, post=Ex_1)
nest.Connect(pre=mm_Ex_2, post=Ex_2)
nest.Connect(pre=mm_Ex_3, post=Ex_3)
nest.Connect(pre=mm_Ex_4, post=Ex_4)
#Inhibitory
nest.Connect(pre=mm_In_0, post=In_0)
nest.Connect(pre=mm_In_1, post=In_1)
nest.Connect(pre=mm_In_2, post=In_2)
nest.Connect(pre=mm_In_3, post=In_3)
nest.Connect(pre=mm_In_4, post=In_4)
nest.Connect(pre=mm_In_5, post=In_5)
#Noc
nest.Connect(pre=mm_Noc, post=Noc)


if os.path.isdir('results'):
    shutil.rmtree('results')
    os.mkdir('results')
else:
    os.mkdir('results')
nest.Simulate(T)


plot(gen_rate, glu_weight, gaba_weight, static_weight, group='mnl_mnr')
plot(gen_rate, glu_weight, gaba_weight, static_weight, group='mne')
plot(gen_rate, glu_weight, gaba_weight, static_weight, group='mnf')

# plot(gen_rate, glu_weight, gaba_weight, static_weight, group='mn_n')
# plot(gen_rate, glu_weight, gaba_weight, static_weight, group='in')
# plot(gen_rate, glu_weight, gaba_weight, static_weight, group='s')
# plot(gen_rate, glu_weight, gaba_weight, static_weight, group='af')
# plot(gen_rate, glu_weight, gaba_weight, static_weight, group='ex')
# plot(gen_rate, glu_weight, gaba_weight, static_weight, group='ex_2')


