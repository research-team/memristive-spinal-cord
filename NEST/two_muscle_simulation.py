import os
import traceback
import nest
from time import time
from multiprocessing import cpu_count
from NEST.functions import Functions
from NEST.functions import Parameters
from GRAS.tests_runner import convert_to_hdf5, plot_results
from GRAS import plot_results as draw


class V3(Functions):
	def __init__(self, parameters, iteration):
		"""
        ToDo add info
        Args:
            parameters (Parameters):
            iteration (int):
        """
		nest.ResetKernel()
		nest.SetKernelStatus({'data_path': "dat",
		                      'print_time': True,
		                      'resolution': 0.025,
		                      'overwrite_files': True,
		                      'data_prefix': f"{iteration}_",
		                      'total_num_virtual_procs': cpu_count(),
		                      'rng_seeds': [int(time() * 10000 % 10000)] * cpu_count()})

		super().__init__(parameters)
		self.P = parameters

		# checking

		self.init_network()
		self.simulate()
		self.resave()

	def init_network(self):
		"""
        TODO add info
        """

		if self.P.air:
			self.P.toe = True

		inh_coef = self.P.inh / 100
		quadru_coef = 0.5 if self.P.ped == 4 else 1
		sero_coef = 5.3 if self.P.ht5 else 1
		air_coef = 0.5 if self.P.air else 1
		toe_coef = 0.5 if self.P.toe else 1

		neurons_in_ip = 196  # number of neurons in interneuronal pool
		neurons_in_moto_extensor = 210  # motoneurons number
		neurons_in_moto_flexor = 180
		neurons_in_aff_ip = 196  # number of neurons in interneuronal pool
		neurons_in_afferent = 120  # number of neurons in afferent

		# groups of neurons
		EES = self.form_group("EES")
		E1 = self.form_group("E1")
		E2 = self.form_group("E2")
		E3 = self.form_group("E3")
		E4 = self.form_group("E4")
		E5 = self.form_group("E5")

		CV1 = self.form_group("CV1")
		CV2 = self.form_group("CV2")
		CV3 = self.form_group("CV3")
		CV4 = self.form_group("CV4")
		CV5 = self.form_group("CV5")
		CD4 = self.form_group("CD4")
		CD5 = self.form_group("CD5")

		OM1_0 = self.form_group("OM1_0")
		OM1_1 = self.form_group("OM1_1")
		OM1_2_E = self.form_group("OM1_2_E")
		OM1_2_F = self.form_group("OM1_2_F")
		OM1_3 = self.form_group("OM1_3")

		OM2_0 = self.form_group("OM2_0")
		OM2_1 = self.form_group("OM2_1")
		OM2_2_E = self.form_group("OM2_2_E")
		OM2_2_F = self.form_group("OM2_2_F")
		OM2_3 = self.form_group("OM2_3")

		OM3_0 = self.form_group("OM3_0")
		OM3_1 = self.form_group("OM3_1")
		OM3_2_E = self.form_group("OM3_2_E")
		OM3_2_F = self.form_group("OM3_2_F")
		OM3_3 = self.form_group("OM3_3")

		OM4_0 = self.form_group("OM4_0")
		OM4_1 = self.form_group("OM4_1")
		OM4_2_E = self.form_group("OM4_2_E")
		OM4_2_F = self.form_group("OM4_2_F")
		OM4_3 = self.form_group("OM4_3")

		OM5_0 = self.form_group("OM5_0")
		OM5_1 = self.form_group("OM5_1")
		OM5_2_E = self.form_group("OM5_2_E")
		OM5_2_F = self.form_group("OM5_2_F")
		OM5_3 = self.form_group("OM5_3")

		MN_E = self.form_group("MN_E", neurons_in_moto_extensor, d_distr="bimodal")
		MN_F = self.form_group("MN_F", neurons_in_moto_flexor, d_distr="bimodal")

		Ia_E_aff = self.form_group("Ia_E_aff", neurons_in_afferent)
		Ia_F_aff = self.form_group("Ia_F_aff", neurons_in_afferent)

		R_E = self.form_group("R_E")
		R_F = self.form_group("R_F")

		Ia_E_pool = self.form_group("Ia_E_pool", neurons_in_aff_ip)
		Ia_F_pool = self.form_group("Ia_F_pool", neurons_in_aff_ip)

		eIP_E_0 = self.form_group("eIP_E_0", int(neurons_in_ip / 2))
		eIP_E_1 = self.form_group("eIP_E_1", int(neurons_in_ip / 2))
		eIP_F = self.form_group("eIP_F", neurons_in_ip)

		iIP_E = self.form_group("iIP_E", neurons_in_ip)
		iIP_F = self.form_group("iIP_F", neurons_in_ip)

		self.connect_spike_generator(EES, rate=self.P.EES)
		self.connect_noise_generator(CV1, rate=5000, t_end=self.P.skin_stim - 2)
		self.connect_noise_generator(CV2, rate=5000, t_start=self.P.skin_stim, t_end=2 * self.P.skin_stim - 2)
		self.connect_noise_generator(CV3, rate=5000, t_start=2 * self.P.skin_stim, t_end=3 * self.P.skin_stim - 2)
		self.connect_noise_generator(CV4, rate=5000, t_start=3 * self.P.skin_stim, t_end=5 * self.P.skin_stim - 2)
		self.connect_noise_generator(CV5, rate=5000, t_start=5 * self.P.skin_stim, t_end=6 * self.P.skin_stim - 2)
		self.connect_noise_generator(iIP_F, rate=3000, t_start=6 * self.P.skin_stim,
		                             t_end=6 * self.P.skin_stim + self.P.flexor_time - 5)
		self.connect_noise_generator(MN_E, rate=200, weight=75)
		# self.connect_noise_generator(MN_F, rate=200, weight=125)

		# connectomes
		self.connect_fixed_outdegree(EES, E1, 1, 370, no_distr=True)
		self.connect_fixed_outdegree(E1, E2, 1, 80, no_distr=True)
		self.connect_fixed_outdegree(E2, E3, 1, 80, no_distr=True)
		self.connect_fixed_outdegree(E3, E4, 1, 80, no_distr=True)
		self.connect_fixed_outdegree(E4, E5, 1, 80, no_distr=True)

		self.connect_one_to_all(CV1, iIP_E, 0.5, 20)
		self.connect_one_to_all(CV2, iIP_E, 0.5, 20)
		self.connect_one_to_all(CV3, iIP_E, 0.5, 20)
		self.connect_one_to_all(CV4, iIP_E, 0.5, 20)
		self.connect_one_to_all(CV5, iIP_E, 0.5, 20)

		def connect_to_ip(neuron, weight, delay):
			self.connect_fixed_outdegree(neuron, eIP_E_0, delay + 0.5, weight, neurons_in_ip)
			self.connect_fixed_outdegree(eIP_E_0, eIP_E_1, 1.2, 9, neurons_in_ip)

		def connect_to_moto(weight):
			self.connect_fixed_outdegree(eIP_E_0, MN_E, 2, weight, neurons_in_moto_extensor)
			self.connect_fixed_outdegree(eIP_E_1, MN_E, 3, weight * 2.1, neurons_in_moto_extensor)

		# extensor

		coef2 = 1.8

		connect_to_ip(OM1_2_E, 14 * coef2, 1.4)
		connect_to_ip(OM2_2_E, 14 * coef2, 1.3)
		connect_to_ip(OM3_2_E, 14 * coef2, 1.9)
		connect_to_ip(OM4_2_E, 15.5 * coef2, 2.3)
		connect_to_ip(OM5_2_E, 16.5 * coef2, 2.7)

		connect_to_moto(6)

		# flexor
		self.connect_fixed_outdegree(OM1_2_F, eIP_F, 1, 7, neurons_in_ip)
		self.connect_fixed_outdegree(OM2_2_F, eIP_F, 1, 7, neurons_in_ip)
		self.connect_fixed_outdegree(OM3_2_F, eIP_F, 1, 7, neurons_in_ip)
		self.connect_fixed_outdegree(OM4_2_F, eIP_F, 1, 7, neurons_in_ip)
		self.connect_fixed_outdegree(OM5_2_F, eIP_F, 1, 7, neurons_in_ip)

		self.connect_fixed_outdegree(eIP_F, MN_F, 1, 3, neurons_in_moto_flexor)

		# self.connect_fixed_outdegree(MN_F, R_F, 0.5, 2)
		# self.connect_fixed_outdegree(R_F, MN_F, 2, 0, neurons_in_moto_flexor)

		self.connect_fixed_outdegree(Ia_F_aff, MN_F, 2, 0.5, neurons_in_moto_flexor)
		self.connect_fixed_outdegree(Ia_E_pool, MN_F, 1, -50, neurons_in_ip)

		# OM 1
		# input from EES group 1
		self.connect_fixed_outdegree(E1, OM1_0, 3, 0.65)
		# input from sensory
		self.connect_one_to_all(CV1, OM1_0, 2.5, 0.75 * quadru_coef * sero_coef * toe_coef)
		self.connect_one_to_all(CV2, OM1_0, 2.5, 0.75 * quadru_coef * sero_coef * toe_coef)
		# [inhibition]
		self.connect_one_to_all(CV3, OM1_3, 2.5, 100)
		self.connect_one_to_all(CV4, OM1_3, 2.5, 100)
		self.connect_one_to_all(CV5, OM1_3, 2.5, 100)
		# inner connectomes
		self.connect_fixed_outdegree(OM1_0, OM1_1, 2.5, 50)
		self.connect_fixed_outdegree(OM1_1, OM1_2_E, 2, 7)
		self.connect_fixed_outdegree(OM1_1, OM1_2_F, 2.4, 8)
		self.connect_fixed_outdegree(OM1_1, OM1_3, 1.5, 3)
		self.connect_fixed_outdegree(OM1_2_E, OM1_1, 1.5, 6.5)
		self.connect_fixed_outdegree(OM1_2_F, OM1_1, 2.5, 30)
		self.connect_fixed_outdegree(OM1_2_E, OM1_3, 1.5, 8)
		self.connect_fixed_outdegree(OM1_2_F, OM1_3, 0.4, 8)
		self.connect_fixed_outdegree(OM1_3, OM1_1, 1.5, -2 * inh_coef)
		self.connect_fixed_outdegree(OM1_3, OM1_2_E, 1.5, -2 * inh_coef)
		self.connect_fixed_outdegree(OM1_3, OM1_2_F, 1, -1 * inh_coef)
		# output to OM2
		self.connect_fixed_outdegree(OM1_2_F, OM2_2_F, 1, 10)

		# OM 2
		# input from EES group 2
		self.connect_fixed_outdegree(E2, OM2_0, 3, 0.4)
		# input from sensory [CV]
		self.connect_one_to_all(CV2, OM2_0, 2.5, 0.8 * quadru_coef * sero_coef * toe_coef)
		self.connect_one_to_all(CV3, OM2_0, 2.5, 0.8 * quadru_coef * sero_coef * toe_coef)
		# [inhibition]
		self.connect_one_to_all(CV4, OM2_3, 3, 80)
		self.connect_one_to_all(CV5, OM2_3, 3, 80)
		# # inner connectomes
		self.connect_fixed_outdegree(OM2_0, OM2_1, 2.5, 35)
		self.connect_fixed_outdegree(OM2_1, OM2_2_E, 2, 7.3)
		self.connect_fixed_outdegree(OM2_1, OM2_2_F, 2.4, 8)
		self.connect_fixed_outdegree(OM2_1, OM2_3, 1.5, 3)
		self.connect_fixed_outdegree(OM2_2_E, OM2_1, 1.5, 6.4)
		self.connect_fixed_outdegree(OM2_2_F, OM2_1, 2.5, 30)
		self.connect_fixed_outdegree(OM2_2_E, OM2_3, 1.5, 8)
		self.connect_fixed_outdegree(OM2_2_F, OM2_3, 0.4, 8)
		self.connect_fixed_outdegree(OM2_3, OM2_1, 1.5, -2 * inh_coef)
		self.connect_fixed_outdegree(OM2_3, OM2_2_E, 1.5, -2 * inh_coef)
		self.connect_fixed_outdegree(OM2_3, OM2_2_F, 1, -1 * inh_coef)
		# output to OM3
		self.connect_fixed_outdegree(OM2_2_F, OM3_2_F, 1, 10)

		# OM 3
		# input from EES group 3
		self.connect_fixed_outdegree(E3, OM3_0, 3, 0.42)
		# input from sensory [CV]
		self.connect_one_to_all(CV3, OM3_0, 2.5, 0.8 * quadru_coef * sero_coef * toe_coef)
		self.connect_one_to_all(CV4, OM3_0, 2.5, 0.8 * quadru_coef * sero_coef * toe_coef)
		# [inhibition]
		self.connect_one_to_all(CV5, OM3_3, 3, 80)
		# input from sensory [CD]
		self.connect_one_to_all(CD4, OM3_0, 3, 11)
		# inner connectomes
		self.connect_fixed_outdegree(OM3_0, OM3_1, 2.5, 40)
		self.connect_fixed_outdegree(OM3_1, OM3_2_E, 2, 7.2)
		self.connect_fixed_outdegree(OM3_1, OM3_2_F, 2.4, 8)
		self.connect_fixed_outdegree(OM3_1, OM3_3, 1.5, 3)
		self.connect_fixed_outdegree(OM3_2_E, OM3_1, 1.5, 6.3)
		self.connect_fixed_outdegree(OM3_2_F, OM3_1, 2.5, 30)
		self.connect_fixed_outdegree(OM3_2_E, OM3_3, 1.5, 8)
		self.connect_fixed_outdegree(OM3_2_F, OM3_3, 0.4, 8)
		self.connect_fixed_outdegree(OM3_3, OM3_1, 1.5, -2 * inh_coef)
		self.connect_fixed_outdegree(OM3_3, OM3_2_E, 1.5, -2 * inh_coef)
		self.connect_fixed_outdegree(OM3_3, OM3_2_F, 1, -1 * inh_coef)
		# output to OM4
		self.connect_fixed_outdegree(OM3_2_F, OM4_2_F, 1, 10)

		# OM 4
		# input from EES group 4
		self.connect_fixed_outdegree(E4, OM4_0, 3, 0.425)
		# input from sensory [CV]
		self.connect_one_to_all(CV4, OM4_0, 3.5, 0.8 * quadru_coef * sero_coef * air_coef)
		self.connect_one_to_all(CV5, OM4_0, 3.5, 0.8 * quadru_coef * sero_coef * air_coef)
		# input from sensory [CD]
		self.connect_one_to_all(CD4, OM4_0, 3, 11)
		self.connect_one_to_all(CD5, OM4_0, 3, 11)
		# inner connectomes
		self.connect_fixed_outdegree(OM4_0, OM4_1, 3, 40)
		self.connect_fixed_outdegree(OM4_1, OM4_2_E, 2, 6.7)
		self.connect_fixed_outdegree(OM4_1, OM4_2_F, 2.4, 8)
		self.connect_fixed_outdegree(OM4_1, OM4_3, 1.5, 3)
		self.connect_fixed_outdegree(OM4_2_E, OM4_1, 1.5, 5.5)
		self.connect_fixed_outdegree(OM4_2_F, OM4_1, 2.5, 30)
		self.connect_fixed_outdegree(OM4_2_E, OM4_3, 1.5, 8)
		self.connect_fixed_outdegree(OM4_2_F, OM4_3, 0.4, 8)
		self.connect_fixed_outdegree(OM4_3, OM4_1, 1.5, -2 * inh_coef)
		self.connect_fixed_outdegree(OM4_3, OM4_2_E, 1.5, -2 * inh_coef)
		self.connect_fixed_outdegree(OM4_3, OM4_2_F, 1, -1 * inh_coef)
		# output to OM5
		self.connect_fixed_outdegree(OM4_2_F, OM5_2_F, 1, 10)

		# OM 5
		# input from EES group 5
		self.connect_fixed_outdegree(E5, OM5_0, 3.5, 0.46)
		# input from sensory [CV]
		self.connect_one_to_all(CV5, OM5_0, 3, 0.85 * quadru_coef * sero_coef * air_coef)
		# input from sensory [CD]
		self.connect_one_to_all(CD5, OM5_0, 3, 11)
		# inner connectomes
		self.connect_fixed_outdegree(OM5_0, OM5_1, 3.5, 40)
		self.connect_fixed_outdegree(OM5_1, OM5_2_E, 2, 8)
		self.connect_fixed_outdegree(OM5_1, OM5_2_F, 2.4, 8)
		self.connect_fixed_outdegree(OM5_1, OM5_3, 1.5, 3)
		self.connect_fixed_outdegree(OM5_2_E, OM5_1, 1.5, 7)
		self.connect_fixed_outdegree(OM5_2_F, OM5_1, 2.5, 30)
		self.connect_fixed_outdegree(OM5_2_E, OM5_3, 1.5, 8)
		self.connect_fixed_outdegree(OM5_2_F, OM5_3, 0.4, 8)
		self.connect_fixed_outdegree(OM5_3, OM5_1, 1.5, -3 * inh_coef)
		self.connect_fixed_outdegree(OM5_3, OM5_2_E, 1.5, -3 * inh_coef)
		self.connect_fixed_outdegree(OM5_3, OM5_2_F, 1, -1 * inh_coef)

		# reflex arc
		self.connect_fixed_outdegree(iIP_E, eIP_F, 3.3, -14, neurons_in_ip)
		connect_to_ip(iIP_F, -40, 0.5)
		# self.connect_fixed_outdegree(iIP_F, eIP_E, 0.5, -40, neurons_in_ip)

		self.connect_fixed_outdegree(iIP_E, OM1_2_F, 0.5, -3, neurons_in_ip)
		self.connect_fixed_outdegree(iIP_E, OM2_2_F, 0.5, -3, neurons_in_ip)
		self.connect_fixed_outdegree(iIP_E, OM3_2_F, 0.5, -3, neurons_in_ip)
		self.connect_fixed_outdegree(iIP_E, OM4_2_F, 0.5, -3, neurons_in_ip)
		self.connect_fixed_outdegree(iIP_E, OM5_2_F, 0.5, -3, neurons_in_ip)

		self.connect_fixed_outdegree(EES, Ia_E_aff, 1, 500)
		self.connect_fixed_outdegree(EES, Ia_F_aff, 1, 500)

		self.connect_fixed_outdegree(iIP_E, Ia_E_pool, 1, 30, neurons_in_ip)
		self.connect_fixed_outdegree(iIP_F, Ia_F_pool, 1, 30, neurons_in_ip)

		self.connect_fixed_outdegree(Ia_E_pool, Ia_F_pool, 1, -1, neurons_in_ip)
		self.connect_fixed_outdegree(Ia_F_pool, MN_E, 1, -10, neurons_in_ip)
		self.connect_fixed_outdegree(Ia_F_pool, Ia_E_pool, 1, -1, neurons_in_ip)
		self.connect_fixed_outdegree(Ia_E_aff, MN_E, 3.4, 14, neurons_in_moto_extensor)

		self.connect_fixed_outdegree(R_E, R_F, 2, -1)
		self.connect_fixed_outdegree(R_F, R_E, 2, -1)


if __name__ == "__main__":
	parameters = Parameters()
	parameters.tests = 4
	parameters.steps = 10
	parameters.cms = 15
	parameters.EES = 40
	parameters.inh = 100
	parameters.ped = 2
	parameters.ht5 = False
	parameters.air = False
	parameters.toe = False
	parameters.save_all = False

	save_folder = f"{os.getcwd()}/dat"

	for i in range(parameters.tests):
		while True:
			try:
				V3(parameters, iteration=i)
				break
			except Exception:
				print(traceback.format_exc())
				continue

	convert_to_hdf5(save_folder)
	plot_results(save_folder, ees_hz=parameters.EES)
	# draw.run()
