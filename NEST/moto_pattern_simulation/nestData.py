global_id = 0
# SIM_TIME_IN_STEPS ????
LEG_STEPS = 3             # [step] number of full cycle steps
SIM_STEP = 0.025   # [s] simulation step
SIM_TIME = 600
CV_timing = 10000     # frequency that depends on CV stimulation period (1000 / 8 = 125 ms - CV stimulation period)

# stuff variables
syn_outdegree = 27        # synapse number outgoing from one neuron
neurons_in_ip = 196       # number of neurons in interneuronal pool
neurons_in_aff_ip = 196   # number of neurons in interneuronal pool
neurons_in_moto = 169     # motoneurons number
neurons_in_group = 20     # number of neurons in a group
neurons_in_afferent = 120 # number of neurons in afferent

# neuron parameters
g_Na = 20000.0          # [nS] Maximal conductance of the Sodium current
g_K = 6000.0            # [nS] Maximal conductance of the Potassium current
g_L = 30.0              # [nS] Conductance of the leak current
C_m = 200.0             # [pF] Capacity of the membrane
E_Na = 50.0             # [mV] Reversal potential for the Sodium current
E_K = -100.0            # [mV] Reversal potential for the Potassium current
E_L = -72.0             # [mV] Reversal potential for the leak current
E_ex = 0.0              # [mV] Reversal potential for excitatory input
E_in = -80.0            # [mV] Reversal potential for inhibitory input
tau_syn_exc = 0.2       # [ms] Decay time of excitatory synaptic current (ms)
tau_syn_inh = 2.0       # [ms] Decay time of inhibitory synaptic current (ms)
V_adj = -63.0           # adjusts threshold to around -50 mV
g_bar = 1500.0          # [nS] the maximal possible conductivity


