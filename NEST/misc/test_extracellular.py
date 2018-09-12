import nest
import pylab

nest.ResetKernel()
nest.SetKernelStatus({
		'total_num_virtual_procs': 4,
		'print_time': False,
		'resolution': 0.1,
		'overwrite_files': True})

nrn_params = {
	't_ref': 2.0,       # [ms] refractory period
	'V_m': -70.,        # [mV] membrane potential
	'E_L': -70.,        # [mV] leak reversal potential
	'E_K': -77.,        # [mV] potassium reversal potential
	'g_L': 30.,         # [nS] leak conductance
	'g_Na': 12000.,     # [nS] sodium peak conductance
	'g_K': 3600.,       # [nS] potassium peak conductance
	'C_m': 134.,        # [pf] capacity of membrane
	'tau_syn_ex': 0.2,  # [ms] time of excitatory action
	'tau_syn_in': 2.0,  # [ms] time of inhibitory action
}
multimeter_params = {
	'record_from': ['extracellular', 'V_m'],
	'withgid': True,
	'withtime': True,
	'interval': 0.1,
	'to_file': False,
	'to_memory': True
}
n = nest.Create("hh_cond_exp_traub", 1, nrn_params)
m = nest.Create("multimeter", 1, multimeter_params)
g = nest.Create("spike_generator", 1, {'spike_weights': [400.0, 200.0],
                                       'spike_times': [2.5, 12.5]})
nest.Connect(g, n, syn_spec={'weight': 1.0, 'delay': 0.1})
nest.Connect(m, n)
nest.Simulate(20.0)

ex_cell_raw = []
with open("/home/alex/test", 'r') as file:
	for line in file:
		ex_cell_raw.append(-float(line))
volt_normal = nest.GetStatus(m, "events")[0]['V_m']
ex_cell_normal = nest.GetStatus(m, "events")[0]['extracellular']


te_raw = [i/100 for i in range(len(ex_cell_raw))]
te_norm = [i/10 for i in range(len(ex_cell_normal))]
tv_norm = [i/10 for i in range(len(volt_normal))]
print(len(te_raw))
print(len(te_norm))
# 5.0 = 150
# 2.5 = 75

pylab.subplot(311)
pylab.title("Extracellular RAW")
pylab.axvline(x=2.5, color='r')
pylab.axvline(x=12.5, color='r')
pylab.xlim(0, 20)
pylab.plot(te_raw, ex_cell_raw)

pylab.subplot(312)
pylab.title("Extracellular normal")
pylab.axvline(x=2.5, color='r')
pylab.axvline(x=12.5, color='r')
pylab.xlim(0, 20)
pylab.plot(te_norm, ex_cell_normal)

pylab.subplot(313)
pylab.title("V_m normal")
pylab.axvline(x=2.5, color='r')
pylab.axvline(x=12.5, color='r')
pylab.xlim(0, 20)
pylab.plot(tv_norm, volt_normal)
pylab.show()
