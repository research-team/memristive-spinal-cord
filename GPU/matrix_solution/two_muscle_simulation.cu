#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <ctime>
#include <cmath>
#include <stdexcept>
#include <random>

// for file writing
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unistd.h>

// my classes
#include "Group.cpp"

// jetbrains cuda
#ifdef __JETBRAINS_IDE__
	#define __host__
	#define __device__
	#define __global__
	#define __shared__
#endif

/**
 6 cm/s = 125 [ms] has 30 slices
15 cm/s = 50 [ms] has 15 slices
21 cm/s = 25 [ms] has 6 slices

References:
  [1] https://en.wikipedia.org/wiki/Hodgkin–Huxley_model

**/

//float mV = 1e-3;
//float ms = 1e-3;
//float mSiemens = 1e-3;
//float uFarad = 1e-6;

// parameters for variability of the simulation
const float INH_COEF = 1.0f;         // strength coefficient of inhibitory synapses
const int EES_FREQ = 100;            // [hz] spike frequency of EES
const int SENSORY_FREQ = 200;        // [hz] spike frequency of C1-C5
const float T_SIMULATION = 20;       // [ms] simulation time
const float SIM_STEP = 0.025;        // [s] simulation step
const int SPEED = 21;                // [cm/s] speed of rat moving
const int skin_stim_time = 25;       // [ms] time of stimulating sensory (based on speed)
const int slices_number = 6;         // number of slices (based on speed)

// stuff variables
unsigned int global_id = 0;          // iter to count neurons one by one
const int syn_outdegree = 27;        // synapse number outgoing from one neuron
const int neurons_in_ip = 196;       // number of neurons in interneuronal pool
const int neurons_in_moto = 169;     // motoneurons number
const int neurons_in_group = 20;     // number of neurons in a group
const int neurons_in_afferent = 60;  // number of neurons in afferent

// neuron parameters
const float g_Na = 20000.0;          // [nS] Maximal conductance of the Sodium current
const float g_K = 6000.0;            // [nS] Maximal conductance of the Potassium current
const float g_L = 10.0;              // [nS] Conductance of the leak current
const float C_m = 200.0;             // [pF] Capacity of the membrane
const float E_Na = 40.0;             // [mV] Reversal potential for the Sodium current
const float E_K = -90.0;             // [mV] Reversal potential for the Potassium current
const float E_L = -72.0;             // [mV] Reversal potential for the leak current
const float E_ex = 0.0;              // [mV] Reversal potential for excitatory input
const float E_in = -80.0;            // [mV] Reversal potential for inhibitory input
const float tau_syn_exc = 0.2;       // [ms] Decay time of excitatory synaptic current (ms)
const float tau_syn_inh = 2.0;       // [ms] Decay time of inhibitory synaptic current (ms)
const float V_adj = -63.0;           // adjusts threshold to around -50 mV
const float g_bar = 1500;            // [nS] the maximal possible conductivity

// calculate spike frequency in steps [steps]
const unsigned int sensory_spike_each_step = (unsigned int)(1000 / SENSORY_FREQ / SIM_STEP);
const unsigned int ees_spike_each_step = (unsigned int)(1000 / EES_FREQ / SIM_STEP);
// calculate start time of CV spiking [steps]
const unsigned int CV1_begin_spiking_time = (unsigned int)(0.1 / SIM_STEP);
const unsigned int CV2_begin_spiking_time = (unsigned int)(skin_stim_time / SIM_STEP);
const unsigned int CV3_begin_spiking_time = (unsigned int)(2 * skin_stim_time / SIM_STEP);
const unsigned int CV4_begin_spiking_time = (unsigned int)(3 * skin_stim_time / SIM_STEP);
const unsigned int CV5_begin_spiking_time = (unsigned int)(5 * skin_stim_time / SIM_STEP);
// calculate end time of CV spiking [steps]
const unsigned int CV1_end_spiking_time = (unsigned int)(skin_stim_time / SIM_STEP);
const unsigned int CV2_end_spiking_time = (unsigned int)(2 * skin_stim_time / SIM_STEP);
const unsigned int CV3_end_spiking_time = (unsigned int)(3 * skin_stim_time / SIM_STEP);
const unsigned int CV4_end_spiking_time = (unsigned int)(5 * skin_stim_time / SIM_STEP);
const unsigned int CV5_end_spiking_time = (unsigned int)(6 * skin_stim_time / SIM_STEP);
// calculate steps activation of C0 and C1
const unsigned int steps_activation_C0 = (unsigned int)(skin_stim_time * 5 / SIM_STEP);
const unsigned int steps_activation_C1 = (unsigned int)(skin_stim_time * slices_number / SIM_STEP);
// calculate how much steps in simulation time [steps]
const unsigned int sim_time_in_steps = (unsigned int)(T_SIMULATION / SIM_STEP);

// struct for human-readable initialization of connectomes
struct SynapseMetadata {
	int post_id;           // post neuron ID
	int synapse_delay;     // [step] synaptic delay of the synapse (axonal delay is included to this delay)
	float synapse_weight;  // [nS] synaptic weight. Interpreted as changing conductivity of neuron membrane

	SynapseMetadata() = default;
	SynapseMetadata(int post_id, float synapse_delay, float synapse_weight){
		this->post_id = post_id;
		this->synapse_delay = static_cast<int>(synapse_delay * (1 / SIM_STEP) + 0.5);  // round
		this->synapse_weight = synapse_weight;
	}
};

// form structs of neurons global ID and groups name
Group form_group(string group_name, int nrns_in_group = neurons_in_group) {
	Group group = Group();
	group.group_name = group_name;     // name of a neurons group
	group.id_start = global_id;        // first ID in the group
	group.id_end = global_id + nrns_in_group - 1;  // the latest ID in the group
	group.group_size = nrns_in_group;  // size of the neurons group

	global_id += nrns_in_group;

	printf("Formed %s IDs [%d ... %d] = %d\n", group_name.c_str(), global_id - nrns_in_group, global_id - 1, nrns_in_group);

	return group;
}

/* Nodes with changable connectomes
C=1 :: disable for neurons 0 <= tid <= 100 and their first 27 synapses. Slice as [54:]
[D2_3, D4_3, D1_3, G2_1, G2_2, G3_1, G3_2, G4_1, G4_2, G5_1, G5_2] */

/// Form neuron groups
// At first init nodes with changable connectomes to reduce "and" operators at synapse checking (by tid)

// inhibited by C=1 group
Group D1_3 = form_group("D1_3");  // D1_3 IDs [0 ... 19]
Group D2_3 = form_group("D2_3");
Group D4_3 = form_group("D4_3");
Group G3_1 = form_group("G3_1");
Group G3_2 = form_group("G3_2");  // G3_2 IDs [80 ... 99]

// groups of neurons with generators
Group CV1 = form_group("CV1");
Group CV2 = form_group("CV2");
Group CV3 = form_group("CV3");
Group CV4 = form_group("CV4");
Group CV5 = form_group("CV5");
Group EES = form_group("EES");

// groups of neurons without changable synapses
Group D1_1 = form_group("D1_1");
Group D1_2 = form_group("D1_2");
// Group D1_3 inited in the group above
Group D1_4 = form_group("D1_4");

Group D2_1 = form_group("D2_1");
Group D2_2 = form_group("D2_2");
// Group D2_3 inited in the group above
Group D2_4 = form_group("D2_4");

Group D3_1 = form_group("D3_1");
Group D3_2 = form_group("D3_2");
Group D3_3 = form_group("D3_3");
Group D3_4 = form_group("D3_4");

Group D4_1 = form_group("D4_1");
Group D4_2 = form_group("D4_2");
// Group D4_3 inited in the group above
Group D4_4 = form_group("D4_4");

Group D5_1 = form_group("D5_1");
Group D5_2 = form_group("D5_2");
Group D5_3 = form_group("D5_3");
Group D5_4 = form_group("D5_4");

Group G1_1 = form_group("G1_1");
Group G1_2 = form_group("G1_2");
Group G1_3 = form_group("G1_3");

Group G2_1 = form_group("G2_1");
Group G2_2 = form_group("G2_2");
Group G2_3 = form_group("G2_3");

// Group G3_1 inited in the group above
// Group G3_2 inited in the group above
Group G3_3 = form_group("G3_3");

Group G4_1 = form_group("G4_1");
Group G4_2 = form_group("G4_2");
Group G4_3 = form_group("G4_3");

Group G5_1 = form_group("G5_1");
Group G5_2 = form_group("G5_2");
Group G5_3 = form_group("G5_3");

Group IP_E = form_group("IP_E", neurons_in_ip);
Group IP_F = form_group("IP_F", neurons_in_ip);

Group MP_E = form_group("MP_E", neurons_in_moto);
Group MP_F = form_group("MP_F", neurons_in_moto);

Group Ia_Extensor = form_group("Ia_Extensor", neurons_in_afferent);
Group Ia_Flexor = form_group("Ia_Flexor", neurons_in_afferent);

Group inh_group3 = form_group("inh_group3");
Group inh_group4 = form_group("inh_group4");
Group inh_group5 = form_group("inh_group5");

Group ees_group1 = form_group("ees_group1");
Group ees_group2 = form_group("ees_group2");
Group ees_group3 = form_group("ees_group3");
Group ees_group4 = form_group("ees_group4");

Group R_E = form_group("R_E");
Group R_F = form_group("R_F");

Group Ia_E = form_group("Ia_E");
Group Ia_F = form_group("Ia_F");
Group Ib_E = form_group("Ib_E");
Group Ib_F = form_group("Ib_F");

// global vectors of SynapseMetadata of synapses for each neuron
vector<vector<SynapseMetadata>> metadatas(global_id, vector<SynapseMetadata>());

__host__
int ms_to_step(float ms) { return (int)(ms / SIM_STEP); }

__host__
float step_to_ms(int step) { return step * SIM_STEP; }


__global__
void GPU_neurons_kernel(float *V_m,
                        float *h,
                        float *m,
                        float *n,
                        float *g_exc,
                        float *g_inh,
                        bool *has_spike,
                        int *nrn_ref_time,
                        int *nrn_ref_time_timer,
                        int neurons_number,
                        int shifted_sim_iter,
                        int activated_C_,
                        int sim_iter){

	__shared__ short decrease_lvl_Ia_spikes;
	__shared__ bool sensory_spike_flag;

	// get ID of the thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// init shared variables by "master" thread
	if (tid == 0) {
		decrease_lvl_Ia_spikes = 1;
		sensory_spike_flag = sim_iter % sensory_spike_each_step == 0;
	}

	__syncthreads();

	// activate only sensory C1-CV5 and control Ia afferent spikes
	if (100 <= tid && tid <= 199) {
		// CV1
		if (100 <= tid && tid <= 119 && (CV1_begin_spiking_time <= shifted_sim_iter) &&
			(shifted_sim_iter < CV1_end_spiking_time)) {
			if (tid == 100) {
				decrease_lvl_Ia_spikes = 3;
			}
			if (activated_C_ == 1 && sensory_spike_flag) {
				g_exc[tid] = g_bar; // set spike state
			}
		} else {
			// CV2
			if (120 <= tid && tid <= 139 && (CV2_begin_spiking_time <= shifted_sim_iter) &&
				(shifted_sim_iter < CV2_end_spiking_time)) {
				if (tid == 120) {
					decrease_lvl_Ia_spikes = 2;
				}
				if (activated_C_ == 1 && sensory_spike_flag) {
					g_exc[tid] = g_bar; // set spike state
				}
			} else {
				// CV3
				if (140 <= tid && tid <= 159 && (CV3_begin_spiking_time <= shifted_sim_iter) &&
					(shifted_sim_iter < CV3_end_spiking_time)) {
					if (tid == 140) {
						decrease_lvl_Ia_spikes = 1;
					}
					if (activated_C_ == 1 && sensory_spike_flag) {
						g_exc[tid] = g_bar; // set spike state
					}
				} else {
					// CV4
					if (160 <= tid && tid <= 179 && (CV4_begin_spiking_time <= shifted_sim_iter) &&
						(shifted_sim_iter < CV4_end_spiking_time)) {
						if (tid == 160) {
							decrease_lvl_Ia_spikes = 2;
						}
						if (activated_C_ == 1 && sensory_spike_flag) {
							g_exc[tid] = g_bar; // set spike state
						}
					} else {
						// CV5
						if (180 <= tid && tid <= 199 && (CV5_begin_spiking_time <= shifted_sim_iter) &&
							(shifted_sim_iter < CV5_end_spiking_time)) {
							if (tid == 180) {
								decrease_lvl_Ia_spikes = 3;
							}
							if (activated_C_ == 1 && sensory_spike_flag) {
								g_exc[tid] = g_bar; // set spike state
							}
						}
					}
				}
			}
		}
	}

	__syncthreads();

	// generating spikes for EES
	if (200 <= tid && tid <= 219 && ((sim_iter + 1) % ees_spike_each_step == 0)) {
		g_exc[tid] = g_bar;  // set spike state
	}

	// Ia IDs [1550 ... 1669], control spike number of Ia afferent by resetting neuron current
	if (1550 <= tid && tid <= 1669) {
		// rule for the 2nd level
		if (decrease_lvl_Ia_spikes == 2 && tid % 3 == 0) {
			// reset current of 1/3 of neurons
			g_inh[tid] = g_bar;  // set maximal inhibitory conductivity
		} else {
			// rule for the 3rd level
			if (decrease_lvl_Ia_spikes == 3 && tid % 2 == 0) {
				// reset current of 1/2 of neurons
				g_inh[tid] = g_bar;  // set maximal inhibitory conductivity
			}
		}
	}

	// inhibit IP_E IDs [820 ... 1015] by C0
	if (activated_C_ == 0 && 820 <= tid && tid <= 1015) {
		g_inh[tid] = g_bar * INH_COEF;
	}
	// inhibit IP_F IDs [1016 ... 1211] by C1
	if (activated_C_ == 1 && 1016 <= tid && tid <= 1211) {
		g_inh[tid] = g_bar * INH_COEF;
	}
	// inhibit Ia_Extensor IDs [1550 ... 1609] by C0
	if (activated_C_ == 0 && 1550 <= tid && tid <= 1609) {
		g_inh[tid] = g_bar * INH_COEF;
	}
	// inhibit Ia_Flexor IDs [1610 ... 1669] by C1
	if (activated_C_ == 1 && 1610 <= tid && tid <= 1669) {
		g_inh[tid] = g_bar * INH_COEF;
	}

	__syncthreads();

	// ignore threads which ID is greater than neurons number
	if (tid < neurons_number) {
		// reset spike flag of the current neuron before calculations
		has_spike[tid] = false;

		// the maximal value of input current
		if (g_exc[tid] > g_bar)
			g_exc[tid] = g_bar;
		if (g_inh[tid] > g_bar)
			g_inh[tid] = g_bar;

		// previous membrane potential
		const float prev_V = V_m[tid];

		// use temporary V variable as V_m with adjust
		const float V = V_m[tid] - V_adj;

		// transition rates between open and closed states of the potassium channels
		float alpha_n = 0.032 * (15.0 - V) / (std::exp((15.0 - V) / 5.0) - 1.0);
		float beta_n = 0.5 * std::exp((10.0 - V) / 40.0);

		// transition rates between open and closed states of the activation of sodium channels
		float alpha_m = 0.32 * (13.0 - V) / (std::exp((13.0 - V) / 4.0) - 1.0);
		float beta_m = 0.28 * (V - 40.0) / (std::exp((V - 40.0) / 5.0) - 1.0);

		// transition rates between open and closed states of the inactivation of sodium channels
		float alpha_h = 0.128 * std::exp((17.0 - V) / 18.0);
		float beta_h = 4.0 / (1.0 + std::exp((40.0 - V) / 5.0));

		// re-calculate activation variables
		n[tid] += (alpha_n - (alpha_n + beta_n) * n[tid]) * SIM_STEP;
		m[tid] += (alpha_m - (alpha_m + beta_m) * m[tid]) * SIM_STEP;
		h[tid] += (alpha_h - (alpha_h + beta_h) * h[tid]) * SIM_STEP;

		// ionic currents
		float I_NA = g_Na * std::pow(m[tid], 3) * h[tid] * (V_m[tid] - E_Na);
		float I_K = g_K * std::pow(n[tid], 4) * (V_m[tid] - E_K);
		float I_L = g_L * (V_m[tid] - E_L);
		float I_syn_exc = g_exc[tid] * (V_m[tid] - E_ex);
		float I_syn_inh = g_inh[tid] * (V_m[tid] - E_in);

		// if neuron in the refractory state -- ignore synaptic inputs. Re-calculate membrane potential
		if (nrn_ref_time_timer[tid] > 0) {
			V_m[tid] += -(I_L + I_K + I_NA) / C_m * SIM_STEP;
		} else {
			V_m[tid] += -(I_L + I_K + I_NA + I_syn_exc + I_syn_inh) / C_m * SIM_STEP;
		}

		// re-calculate conductance
		g_exc[tid] += -g_exc[tid] / tau_syn_exc * SIM_STEP;
		g_inh[tid] += -g_inh[tid] / tau_syn_inh * SIM_STEP;

		// (threshold && maximal peak)
		if (V_m[tid] >= V_adj + 30 && prev_V < V_m[tid]) {
			has_spike[tid] = true;  // set spike state. It will be used in the "GPU_synapses_kernel"
			nrn_ref_time_timer[tid] = nrn_ref_time[tid];  // set the refractory period
		}

		// update the refractory period timer
		if (nrn_ref_time_timer[tid] > 0)
			nrn_ref_time_timer[tid]--;
	}
}

__global__
void GPU_synapses_kernel(bool* has_spike,
                         int *synapses_number,
                         int **synapses_post_nrn_id,
                         int **synapses_delay,
                         int **synapses_delay_timer,
                         float **synapses_weight,
                         float *g_exc,
                         float *g_inh,
                         int activated_C_,
                         int neurons_number){

	// get ID of the thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// ignore threads which ID is greater than neurons number
	if (tid < neurons_number) {
		// init basic synapse IDs
		int syn_id_begin = 0;
		int syn_id_end = synapses_number[tid];

		// C=1 -- "slice" as [54:] -- skip the first 54 synapses because they must be inhibited
		if (activated_C_ == 1 && 0 <= tid && tid <= 99) {
			syn_id_begin = 54; // 27 * 2
		}

		// pointers to current neuronID synapses_delay_timer (decrease array calls)
		int *ptr_delay_timers = synapses_delay_timer[tid];
		// synapse updating loop (with formed begin/end borders)
		for (int syn_id = syn_id_begin; syn_id < syn_id_end; syn_id++) {
			// add synaptic delay if neuron has spike
			if (has_spike[tid] && ptr_delay_timers[syn_id] == -1) {
				ptr_delay_timers[syn_id] = synapses_delay[tid][syn_id];
			}
			// if synaptic delay is zero it means the time when synapse increase I by synaptic weight
			if (ptr_delay_timers[syn_id] == 0) {
				// post neuron ID = synapses_post_nrn_id[tid][syn_id], thread-safe (!)
				if (synapses_weight[tid][syn_id] >= 0) {
					atomicAdd(&g_exc[synapses_post_nrn_id[tid][syn_id]], synapses_weight[tid][syn_id]);
				} else {
					atomicAdd(&g_inh[synapses_post_nrn_id[tid][syn_id]], -synapses_weight[tid][syn_id]);
				}
				// make synapse timer a "free" for next spikes
				ptr_delay_timers[syn_id] = -1;
			}
			// update synapse delay timer
			if (ptr_delay_timers[syn_id] > 0) {
				ptr_delay_timers[syn_id]--;
			}
		}
	}
}

void connect_fixed_outdegree(Group pre_neurons, Group post_neurons,
                             float syn_delay, float weight, int outdegree = syn_outdegree) {
	// connect neurons with uniform distribution and normal distributon for syn delay and weight
	// weight *= (100 * 0.7);

	random_device rd;
	mt19937 gen(rd());  // initialize pseudo-random number generator

	uniform_int_distribution<int> id_distr(post_neurons.id_start, post_neurons.id_end);
	normal_distribution<float> delay_distr(syn_delay, syn_delay / 10);
	normal_distribution<float> weight_distr(weight, weight / 10);

	#ifdef DEBUG
	printf("pre group %s (%d, %d) to post %s (%d, %d)\n",
	       pre_neurons.group_name.c_str(),
	       pre_neurons.id_start,
	       pre_neurons.id_end,
	       post_neurons.group_name.c_str(),
	       post_neurons.id_start,
	       post_neurons.id_end);
	#endif

	for (int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++) {
		for (int i = 0; i < outdegree; i++) {
			int rand_post_id = id_distr(gen);
			float syn_delay_dist = syn_delay; //delay_distr(gen);
			float syn_weight_dist = weight; //weight_distr(gen);
			#ifdef DEBUG
			printf("weight %f (%f), delay %f (%f) \n", syn_weight_dist, weight, syn_delay_dist, syn_delay);
			#endif
			metadatas.at(pre_id).push_back(SynapseMetadata(rand_post_id, syn_delay_dist, syn_weight_dist));
		}
	}

	printf("Connect %s with %s (1:%d). W=%.2f, D=%.1f\n", pre_neurons.group_name.c_str(),
	                                                      post_neurons.group_name.c_str(),
	                                                      outdegree,
	                                                      weight,
	                                                      syn_delay);
}


void init_extensor_flexor() {
	/// connections which are inhibited by C=1. REMOVED AS [54:]
	// D1 -> G2
	connect_fixed_outdegree(D1_3, G2_1, 0.5, 13);
	connect_fixed_outdegree(D1_3, inh_group5, sim_time_in_steps, 0);  // FixME FAKE connectome
	// D2 -> D3
	connect_fixed_outdegree(D2_3, D3_1, 0.5, 12.5);
	connect_fixed_outdegree(D2_3, D3_4, 0.5, 12.5);
	// D4 -> D5
	connect_fixed_outdegree(D4_3, D5_1, 1, 10);
	connect_fixed_outdegree(D4_3, D5_4, 1, 10);
	// G3 -> G4
	connect_fixed_outdegree(G3_1, G4_1, 1.0, 65);
	connect_fixed_outdegree(G3_1, inh_group5, sim_time_in_steps, 0);	 // FixME FAKE connectome
	connect_fixed_outdegree(G3_2, G4_1, 1.0, 65);
	connect_fixed_outdegree(G3_2, inh_group5, sim_time_in_steps, 0);	 // FixME FAKE connectome
	/// end

	connect_fixed_outdegree(CV3, inh_group3, 0.5, 15); // 0.5
	connect_fixed_outdegree(CV4, inh_group4, 0.5, 15); // 0.5
	connect_fixed_outdegree(CV5, inh_group5, 0.5, 15); // 0.5

	connect_fixed_outdegree(inh_group3, G1_3, 0.5, 20); // 20

	connect_fixed_outdegree(inh_group4, G1_3, 0.5, 20);
	connect_fixed_outdegree(inh_group4, G2_3, 0.5, 20);

	connect_fixed_outdegree(inh_group5, G1_3, 0.5, 20);
	connect_fixed_outdegree(inh_group5, G2_3, 0.5, 20);
	connect_fixed_outdegree(inh_group5, G3_3, 0.5, 20);
	connect_fixed_outdegree(inh_group5, G4_3, 0.5, 20);

	/// D1
	// input from sensory
	connect_fixed_outdegree(CV1, D1_1, 1, 0.4);
	connect_fixed_outdegree(CV1, D1_4, 1, 0.4);
	connect_fixed_outdegree(CV2, D1_1, 1, 0.4);
	connect_fixed_outdegree(CV2, D1_4, 1, 0.4);
	// input from EES
	connect_fixed_outdegree(EES, D1_1, 2, 50); // ST value (?) // was 10
	connect_fixed_outdegree(EES, D1_4, 2, 20); // ST value (?) // was 10
	// inner connectomes
	connect_fixed_outdegree(D1_1, D1_2, 1, 1);
	connect_fixed_outdegree(D1_1, D1_3, 1, 10);
	connect_fixed_outdegree(D1_2, D1_1, 1, 7);
	connect_fixed_outdegree(D1_2, D1_3, 1, 13);
	connect_fixed_outdegree(D1_3, D1_1, 1, -30 * INH_COEF); // -10
	connect_fixed_outdegree(D1_3, D1_2, 1, -30 * INH_COEF); // -10
	connect_fixed_outdegree(D1_4, D1_3, 3, -30 * INH_COEF); // -20
	// output to
	connect_fixed_outdegree(D1_3, G1_1, 3, 6); // 8
	connect_fixed_outdegree(D1_3, ees_group1, 1.0, 60);

	// EES group connectomes
	connect_fixed_outdegree(ees_group1, ees_group2, 1, 20);

	/// D2
	// input from Sensory
	connect_fixed_outdegree(CV2, D2_1, 1, 0.6); // was 8
	connect_fixed_outdegree(CV2, D2_4, 1, 0.8);
	connect_fixed_outdegree(CV3, D2_1, 1, 0.6); // was 8
	connect_fixed_outdegree(CV3, D2_4, 1, 0.8);
	// input from Group (1)
	connect_fixed_outdegree(ees_group1, D2_1, 1.7, 0.8);
	connect_fixed_outdegree(ees_group1, D2_4, 1.7, 1);
	// inner connectomes
	connect_fixed_outdegree(D2_1, D2_2, 1, 3);
	connect_fixed_outdegree(D2_1, D2_3, 1, 10);
	connect_fixed_outdegree(D2_2, D2_1, 1, 7);
	connect_fixed_outdegree(D2_2, D2_3, 1, 20);
	connect_fixed_outdegree(D2_3, D2_1, 1, -20 * INH_COEF);
	connect_fixed_outdegree(D2_3, D2_2, 1, -20 * INH_COEF);
	connect_fixed_outdegree(D2_4, D2_3, 2, -20 * INH_COEF);
	// output to generator
	connect_fixed_outdegree(D2_3, G2_1, 1, 8);

	// EES group connectomes
	connect_fixed_outdegree(ees_group2, ees_group3, 1, 20);

	/// D3
	// input from Sensory
	connect_fixed_outdegree(CV3, D3_1, 1, 0.4); // was 0.5
	connect_fixed_outdegree(CV3, D3_4, 1, 0.5);
	connect_fixed_outdegree(CV4, D3_1, 1, 0.4); // was 0.5
	connect_fixed_outdegree(CV4, D3_4, 1, 0.5);
	// input from Group (2)
	connect_fixed_outdegree(ees_group2, D3_1, 1, 1.0); // was 1.2
	connect_fixed_outdegree(ees_group2, D3_4, 1, 1.2);
	// inner connectomes
	connect_fixed_outdegree(D3_1, D3_2, 1, 3);
	connect_fixed_outdegree(D3_1, D3_3, 1, 10);
	connect_fixed_outdegree(D3_2, D3_1, 1, 7);
	connect_fixed_outdegree(D3_2, D3_3, 1, 20);
	connect_fixed_outdegree(D3_3, D3_1, 1, -10 * INH_COEF);
	connect_fixed_outdegree(D3_3, D3_2, 1, -10 * INH_COEF);
	connect_fixed_outdegree(D3_4, D3_3, 2, -10 * INH_COEF);
	// output to generator
	connect_fixed_outdegree(D3_3, G3_1, 1, 25);
	// suppression of the generator
	connect_fixed_outdegree(D3_3, G1_3, 1.5, 30);

	// EES group connectomes
	connect_fixed_outdegree(ees_group3, ees_group4, 2, 20);

	/// D4
	// input from Sensory
	connect_fixed_outdegree(CV4, D4_1, 1, 0.4);
	connect_fixed_outdegree(CV4, D4_4, 1, 0.5);
	connect_fixed_outdegree(CV5, D4_1, 1, 0.4);
	connect_fixed_outdegree(CV5, D4_4, 1, 0.5);
	// input from Group (3)
	connect_fixed_outdegree(ees_group3, D4_1, 1, 1.0);
	connect_fixed_outdegree(ees_group3, D4_4, 1, 1.2);
	// inner connectomes
	connect_fixed_outdegree(D4_1, D4_2, 1.0, 3);
	connect_fixed_outdegree(D4_1, D4_3, 1.0, 10);
	connect_fixed_outdegree(D4_2, D4_1, 1.0, 7);
	connect_fixed_outdegree(D4_2, D4_3, 1.0, 20);
	connect_fixed_outdegree(D4_3, D4_1, 1.0, -20 * INH_COEF);
	connect_fixed_outdegree(D4_3, D4_2, 1.0, -20 * INH_COEF);
	connect_fixed_outdegree(D4_4, D4_3, 2.0, -20 * INH_COEF);
	// output to the generator
	connect_fixed_outdegree(D4_3, G4_1, 3, 20);
	// suppression of the generator
	connect_fixed_outdegree(D4_3, G2_3, 1, 30);

	/// D5
	// input from Sensory
	connect_fixed_outdegree(CV5, D5_1, 1, 0.5);
	connect_fixed_outdegree(CV5, D5_4, 1, 0.5);
	// input from Group (4)
	connect_fixed_outdegree(ees_group4, D5_1, 1, 0.8); // was 1.1
	connect_fixed_outdegree(ees_group4, D5_4, 1, 1);
	// inner connectomes
	connect_fixed_outdegree(D5_1, D5_2, 1, 3);
	connect_fixed_outdegree(D5_1, D5_3, 1, 15);
	connect_fixed_outdegree(D5_2, D5_1, 1, 7);
	connect_fixed_outdegree(D5_2, D5_3, 1, 20);
	connect_fixed_outdegree(D5_3, D5_1, 1, -20 * INH_COEF);
	connect_fixed_outdegree(D5_3, D5_2, 1, -20 * INH_COEF);
	connect_fixed_outdegree(D5_4, D5_3, 2.5, -20 * INH_COEF);
	// output to the generator
	connect_fixed_outdegree(D5_3, G5_1, 3, 8);
	// suppression of the generators
	connect_fixed_outdegree(D5_3, G1_3, 1, 30);
	connect_fixed_outdegree(D5_3, G2_3, 1, 30);
	connect_fixed_outdegree(D5_3, G3_3, 1, 30);
	connect_fixed_outdegree(D5_3, G4_3, 1, 30);

	/// G1
	// inner connectomes
	connect_fixed_outdegree(G1_1, G1_2, 1, 10);
	connect_fixed_outdegree(G1_1, G1_3, 1, 15);
	connect_fixed_outdegree(G1_2, G1_1, 1, 10);
	connect_fixed_outdegree(G1_2, G1_3, 1, 15);
	connect_fixed_outdegree(G1_3, G1_1, 0.25, -20 * INH_COEF); // -70 - 40 // syn was 0.7
	connect_fixed_outdegree(G1_3, G1_2, 0.25, -20 * INH_COEF); // -70 - 40 // syn was 0.7
	// G1 -> IP_E
	connect_fixed_outdegree(G1_1, IP_E, 3, 20);
	connect_fixed_outdegree(G1_2, IP_E, 3, 20);
	// G1 -> IP_F
	connect_fixed_outdegree(G1_1, IP_F, 2.5, 15);
	connect_fixed_outdegree(G1_2, IP_F, 2.5, 15);
	/// G2
	// inner connectomes
	connect_fixed_outdegree(G2_1, G2_2, 1, 10);
	connect_fixed_outdegree(G2_1, G2_3, 1, 20);
	connect_fixed_outdegree(G2_2, G2_1, 1, 10);
	connect_fixed_outdegree(G2_2, G2_3, 1, 20);
	connect_fixed_outdegree(G2_3, G2_1, 0.5, -30 * INH_COEF);
	connect_fixed_outdegree(G2_3, G2_2, 0.5, -30 * INH_COEF);
	// G2 -> IP_E
	connect_fixed_outdegree(G2_1, IP_E, 1, 20);
	connect_fixed_outdegree(G2_2, IP_E, 1, 20);
	// G2 -> IP_F
	connect_fixed_outdegree(G2_1, IP_F, 3, 20);
	connect_fixed_outdegree(G2_2, IP_F, 3, 20);

	/// G3
	// inner connectomes
	connect_fixed_outdegree(G3_1, G3_2, 1, 14);
	connect_fixed_outdegree(G3_1, G3_3, 1, 20);
	connect_fixed_outdegree(G3_2, G3_1, 1, 12);
	connect_fixed_outdegree(G3_2, G3_3, 1, 20);
	connect_fixed_outdegree(G3_3, G3_1, 0.5, -30 * INH_COEF);
	connect_fixed_outdegree(G3_3, G3_2, 0.5, -30 * INH_COEF);
	// G3 -> IP_E
	connect_fixed_outdegree(G3_1, IP_E, 2, 25);
	connect_fixed_outdegree(G3_2, IP_E, 2, 25);
	// G3 -> IP_F
	connect_fixed_outdegree(G3_1, IP_F, 2.5, 20);
	connect_fixed_outdegree(G3_2, IP_F, 2.5, 20);

	/// G4
	// inner connectomes
	connect_fixed_outdegree(G4_1, G4_2, 1, 10);
	connect_fixed_outdegree(G4_1, G4_3, 1, 10);
	connect_fixed_outdegree(G4_2, G4_1, 1, 5);
	connect_fixed_outdegree(G4_2, G4_3, 1, 10);
	connect_fixed_outdegree(G4_3, G4_1, 0.5, -30 * INH_COEF);
	connect_fixed_outdegree(G4_3, G4_2, 0.5, -30 * INH_COEF);
	// G4 -> IP_E
	connect_fixed_outdegree(G4_1, IP_E, 1, 17);
	connect_fixed_outdegree(G4_2, IP_E, 1, 17);
	// G4 -> IP_F
	connect_fixed_outdegree(G4_1, IP_F, 3, 17);
	connect_fixed_outdegree(G4_2, IP_F, 3, 17);

	/// G5
	// inner connectomes
	connect_fixed_outdegree(G5_1, G5_2, 1, 10);
	connect_fixed_outdegree(G5_1, G5_3, 1, 10);
	connect_fixed_outdegree(G5_2, G5_1, 1, 7);
	connect_fixed_outdegree(G5_2, G5_3, 1, 10);
	connect_fixed_outdegree(G5_3, G5_1, 0.5, -30 * INH_COEF);
	connect_fixed_outdegree(G5_3, G5_2, 0.5, -30 * INH_COEF);
	// G5 -> IP_E
	connect_fixed_outdegree(G5_1, IP_E, 2, 20);
	connect_fixed_outdegree(G5_2, IP_E, 2, 20);
	// G5 -> IP_F
	connect_fixed_outdegree(G5_1, IP_F, 3, 20);
	connect_fixed_outdegree(G5_2, IP_F, 3, 20);
}

void init_ref_arc() {
	connect_fixed_outdegree(EES, Ia_Extensor, 1, 20); // was 20
	connect_fixed_outdegree(EES, Ia_Flexor, 1, 20); // was 20

	connect_fixed_outdegree(IP_E, MP_E, 1, 7); // 11 7
//	connect_fixed_outdegree(IP_E, Ia_E, 2.0, 20.0);
//
////	connect_fixed_outdegree(MP_E, Extensor, 2.0, 20.0);
//	connect_fixed_outdegree(MP_E, R_E, 2.0, 20.0);
//
	connect_fixed_outdegree(IP_F, MP_F, 1, 7); // 11 7
//	connect_fixed_outdegree(IP_F, Ia_F, 2.0, 20.0);
//
////	connect_fixed_outdegree(MP_F, Flexor, 2.0, 20.0);
//	connect_fixed_outdegree(MP_F, R_F, 2.0, 20.0);
//
//	connect_fixed_outdegree(Ib_F, Ib_E, 2.0, -20 * INH_COEF);
//	connect_fixed_outdegree(Ib_F, MP_F, 2.0, -20 * INH_COEF);
//	connect_fixed_outdegree(Ib_E, Ib_F, 2.0, -20 * INH_COEF);
//	connect_fixed_outdegree(Ib_E, MP_E, 2.0, -5 * INH_COEF);
//
//	connect_fixed_outdegree(Ia_F, Ia_E, 2.0, -20 * INH_COEF);
//	connect_fixed_outdegree(Ia_F, MP_E, 2.0, -5 * INH_COEF);
//	connect_fixed_outdegree(Ia_E, Ia_F, 2.0, -20 * INH_COEF);
//	connect_fixed_outdegree(Ia_E, MP_F, 2.0, -20 * INH_COEF);
//
//	connect_fixed_outdegree(R_F, R_E, 2.0, -20 * INH_COEF);
//	connect_fixed_outdegree(R_F, Ia_F, 2.0, -20 * INH_COEF);
//	connect_fixed_outdegree(R_F, MP_F, 2.0, -20 * INH_COEF);
//
//	connect_fixed_outdegree(R_E, R_F, 2.0, -20 * INH_COEF);
//	connect_fixed_outdegree(R_E, Ia_E, 2.0, -20 * INH_COEF);
//	connect_fixed_outdegree(R_E, MP_E, 2.0, -5 * INH_COEF);

	connect_fixed_outdegree(Ia_Flexor, MP_F, 1, 10);
//	connect_fixed_outdegree(Ia, Ia_F, 1.0, 10.0);
//	connect_fixed_outdegree(Ia, Ib_F, 1.0, 10.0);

	connect_fixed_outdegree(Ia_Extensor, MP_E, 1, 10); // was 1 and 10
//	connect_fixed_outdegree(Ia, Ia_E, 1.0, 10.0);
//	connect_fixed_outdegree(Ia, Ib_E, 1.0, 10.0);
}

void save_result(int test_index,
                 float* volt_rec,
                 float* h_rec,
                 float* m_rec,
                 float* n_rec,
                 float* g_exc_rec,
                 float* g_inh_rec,
                 float* spikes_rec,
                 int neurons_number,
                 int full_save) {

	char cwd[256];

	getcwd(cwd, sizeof(cwd));
	printf("[Test #%d] Save results to: %s ...\n", test_index, cwd);

	// creare streams for files
	ofstream volt_rec_file;
	ofstream h_rec_file;
	ofstream m_rec_file;
	ofstream n_rec_file;
	ofstream g_exc_rec_file;
	ofstream g_inh_rec_file;
	ofstream spikes_rec_file;

	// init filenames
	string volt_rec_file_name = "/volt_" + std::to_string(test_index) + ".dat";
	string h_rec_file_name = "/h_" + std::to_string(test_index) + ".dat";
	string m_rec_file_name = "/m_" + std::to_string(test_index) + ".dat";
	string n_rec_file_name = "/n_" + std::to_string(test_index) + ".dat";
	string g_exc_rec_file_name = "/g_exc_" + std::to_string(test_index) + ".dat";
	string g_inh_rec_file_name = "/g_inh_" + std::to_string(test_index) + ".dat";
	string spikes_rec_file_name = "/spikes_" + std::to_string(test_index) + ".dat";

	// open files for writing
	volt_rec_file.open(cwd + volt_rec_file_name);
	h_rec_file.open(cwd + h_rec_file_name);
	m_rec_file.open(cwd + m_rec_file_name);
	n_rec_file.open(cwd + n_rec_file_name);
	g_exc_rec_file.open(cwd + g_exc_rec_file_name);
	g_inh_rec_file.open(cwd + g_inh_rec_file_name);
	spikes_rec_file.open(cwd + spikes_rec_file_name);

	// fill files
	for(int nrn_id = 0; nrn_id < neurons_number; nrn_id++){
		for(int sim_iter = 0; sim_iter < sim_time_in_steps; sim_iter++) {
			int iter = sim_iter + nrn_id * sim_time_in_steps;
			volt_rec_file << volt_rec[iter] << " ";
			h_rec_file << h_rec[iter] << " ";
			m_rec_file << m_rec[iter] << " ";
			n_rec_file << n_rec[iter] << " ";
			g_exc_rec_file << g_exc_rec[iter] << " ";
			g_inh_rec_file << g_inh_rec[iter] << " ";
			spikes_rec_file << spikes_rec[iter] << " ";
		}

		volt_rec_file << std::endl;
		h_rec_file << std::endl;
		m_rec_file << std::endl;
		n_rec_file << std::endl;
		g_exc_rec_file << std::endl;
		g_inh_rec_file << std::endl;
		spikes_rec_file << std::endl;
	}

	// close file streams
	volt_rec_file.close();
	h_rec_file.close();
	m_rec_file.close();
	n_rec_file.close();
	g_exc_rec_file.close();
	g_inh_rec_file.close();
	spikes_rec_file.close();
}

// copy data from host to device
template <typename type>
void memcpyHtD(type* gpu, type* host, int size) {
	cudaMemcpy(gpu, host, sizeof(type) * size, cudaMemcpyHostToDevice);
}

// copy data from device to host
template <typename type>
void memcpyDtH(type* host, type* gpu, int size) {
	cudaMemcpy(host, gpu, sizeof(type) * size, cudaMemcpyDeviceToHost);
}

// get datasize of current variable type and its number
template <typename type>
unsigned int datasize(int size) {
	return sizeof(type) * size;
}

// fill array with current value
template <typename type>
void init_array(type *array, int size, type value){
	for(int i = 0; i < size; i++)
		array[i] = value;
}

__host__
void simulate(int test_index, int full_save) {
	const int neurons_number = static_cast<int>(metadatas.size());

	/// CPU variables
	// neuron variables
	float v_m[neurons_number];               // [mV] neuron membrane potential
	int nrn_ref_time[neurons_number];        // [step] neuron refractory time
	int nrn_ref_time_timer[neurons_number];  // [step] neuron refractory time timer
	bool has_spike[neurons_number];          // neuron state - has spike or not
	int synapses_number[neurons_number];     // prepare variable to keep synapses number per each neuron
	float n[neurons_number];                 // dimensionless quantity [0 .. 1] of potassium channel activation
	float h[neurons_number];                 // dimensionless quantity [0 .. 1] of sodium channel activation
	float m[neurons_number];                 // dimensionless quantity [0 .. 1] of sodium channel inactivation
	float g_exc[neurons_number];             // [nS] excitatory synapse exponential conductance
	float g_inh[neurons_number];             // [nS] inhibitory synapse exponential conductance

	// special arrays to keep mean data of each neuron's states
	float* spikes_rec = (float *)malloc(datasize<float *>(neurons_number * sim_time_in_steps));
	float* volt_rec = (float *)malloc(datasize<float *>(neurons_number * sim_time_in_steps));
	float* h_rec = (float *)malloc(datasize<float *>(neurons_number * sim_time_in_steps));
	float* m_rec = (float *)malloc(datasize<float *>(neurons_number * sim_time_in_steps));
	float* n_rec = (float *)malloc(datasize<float *>(neurons_number * sim_time_in_steps));
	float* g_exc_rec = (float *)malloc(datasize<float *>(neurons_number * sim_time_in_steps));
	float* g_inh_rec = (float *)malloc(datasize<float *>(neurons_number * sim_time_in_steps));

	// fill arrays by initial data
	init_array<float>(v_m, neurons_number, E_L);  // by default all neurons have E_L membrane state at start
	init_array<int>(nrn_ref_time, neurons_number, ms_to_step(3.0)); // by default all neurons have 3ms refractory period
	init_array<int>(nrn_ref_time_timer, neurons_number, -1); // by default all neurons have ref_t timers as -1
	init_array<bool>(has_spike, neurons_number, false);  // by default all neurons haven't spikes at start
	init_array<float>(n, neurons_number, 0);  // by default all neurons have closed potassium channel
	init_array<float>(h, neurons_number, 1);  // by default all neurons have opened sodium channel activation
	init_array<float>(m, neurons_number, 0);  // by default all neurons have closed sodium channel inactivation
	init_array<float>(g_exc, neurons_number, 0);  // by default all neurons have zero excitatory synaptic conductivity
	init_array<float>(g_inh, neurons_number, 0);  // by default all neurons have zero inhibitory synaptic conductivity

	// init connectomes
	init_extensor_flexor();
	init_ref_arc();

	// synapse variables
	int **synapses_post_nrn_id = (int **)malloc(datasize<int* >(neurons_number));  // 2D array of post neuron IDs
	int **synapses_delay = (int **)malloc(datasize<int* >(neurons_number));        // 2D array of synaptic delays
	int **synapses_delay_timer = (int **)malloc(datasize<int* >(neurons_number));  // 2D array of synaptic delays timer
	float **synapses_weight = (float **)malloc(datasize<float* >(neurons_number)); // 2D array of synaptic weights

	// fill arrays of synapses
	for(int neuron_id = 0; neuron_id < neurons_number; neuron_id++) {
		int syn_count = static_cast<int>(metadatas.at(neuron_id).size());
		int tmp_synapses_post_nrn_id[syn_count];
		int tmp_synapses_delay[syn_count];
		int tmp_synapses_delay_timer[syn_count];
		float tmp_synapses_weight[syn_count];

		int syn_id = 0;
		// for each metadata of current neuron fill data in array to copy in future pointers to the GPU
		for(SynapseMetadata metadata : metadatas.at(neuron_id)) {
			tmp_synapses_post_nrn_id[syn_id] = metadata.post_id;
			tmp_synapses_delay[syn_id] = metadata.synapse_delay;
			tmp_synapses_delay_timer[syn_id] = -1;
			tmp_synapses_weight[syn_id] = metadata.synapse_weight;
			syn_id++;
		}
		// set number of synapses to the current neuron array
		synapses_number[neuron_id] = syn_count;

		// allocate memory for pointers in GPU
		cudaMalloc((void**)&synapses_post_nrn_id[neuron_id], datasize<int>(syn_count));
		cudaMalloc((void**)&synapses_delay[neuron_id], datasize<int>(syn_count));
		cudaMalloc((void**)&synapses_delay_timer[neuron_id], datasize<int>(syn_count));
		cudaMalloc((void**)&synapses_weight[neuron_id], datasize<float>(syn_count));

		// copy pointers from CPU to GPU
		cudaMemcpy(synapses_post_nrn_id[neuron_id], &tmp_synapses_post_nrn_id, datasize<int>(syn_count), cudaMemcpyHostToDevice);
		cudaMemcpy(synapses_delay[neuron_id], &tmp_synapses_delay, datasize<int>(syn_count), cudaMemcpyHostToDevice);
		cudaMemcpy(synapses_delay_timer[neuron_id], &tmp_synapses_delay_timer, datasize<int>(syn_count), cudaMemcpyHostToDevice);
		cudaMemcpy(synapses_weight[neuron_id], &tmp_synapses_weight, datasize<float>(syn_count), cudaMemcpyHostToDevice);
	}

	/// GPU variables
	// neuron variables
	float* gpu_v_m;
	int* gpu_nrn_ref_time;
	int* gpu_nrn_ref_time_timer;
	bool* gpu_has_spike;
	int* gpu_synapses_number;
	float* gpu_n;
	float* gpu_h;
	float* gpu_m;
	float* gpu_g_exc;
	float* gpu_g_inh;
	// synapse variables
	int **gpu_synapses_post_nrn_id;
	int **gpu_synapses_delay;
	int **gpu_synapses_delay_timer;
	float **gpu_synapses_weight;

	// allocate memory in the GPU
	cudaMalloc(&gpu_v_m, datasize<float>(neurons_number));
	cudaMalloc(&gpu_h, datasize<float>(neurons_number));
	cudaMalloc(&gpu_m, datasize<float>(neurons_number));
	cudaMalloc(&gpu_n, datasize<float>(neurons_number));
	cudaMalloc(&gpu_has_spike, datasize<bool>(neurons_number));
	cudaMalloc(&gpu_nrn_ref_time, datasize<int>(neurons_number));
	cudaMalloc(&gpu_g_exc, datasize<float>(neurons_number));
	cudaMalloc(&gpu_g_inh, datasize<float>(neurons_number));

	cudaMalloc(&gpu_synapses_number, datasize<int>(neurons_number));
	cudaMalloc(&gpu_nrn_ref_time_timer, datasize<int>(neurons_number));
	cudaMalloc((void ***)&gpu_synapses_post_nrn_id, datasize<int *>(neurons_number));
	cudaMalloc((void ***)&gpu_synapses_delay, datasize<int *>(neurons_number));
	cudaMalloc((void ***)&gpu_synapses_delay_timer, datasize<int *>(neurons_number));
	cudaMalloc((void ***)&gpu_synapses_weight, datasize<float *>(neurons_number));

	// copy data from CPU to GPU
	memcpyHtD<float>(gpu_v_m, v_m, neurons_number);
	memcpyHtD<bool>(gpu_has_spike, has_spike, neurons_number);
	memcpyHtD<int>(gpu_nrn_ref_time, nrn_ref_time, neurons_number);
	memcpyHtD<float>(gpu_g_exc, g_exc, neurons_number);
	memcpyHtD<float>(gpu_g_inh, g_inh, neurons_number);
	memcpyHtD<float>(gpu_h, h, neurons_number);
	memcpyHtD<float>(gpu_m, m, neurons_number);
	memcpyHtD<float>(gpu_n, n, neurons_number);

	memcpyHtD<int>(gpu_synapses_number, synapses_number, neurons_number);
	memcpyHtD<int>(gpu_nrn_ref_time_timer, nrn_ref_time_timer, neurons_number);
	memcpyHtD<int *>(gpu_synapses_post_nrn_id, synapses_post_nrn_id, neurons_number);
	memcpyHtD<int *>(gpu_synapses_delay, synapses_delay, neurons_number);
	memcpyHtD<int *>(gpu_synapses_delay_timer, synapses_delay_timer, neurons_number);
	memcpyHtD<float *>(gpu_synapses_weight, synapses_weight, neurons_number);

	// preparations for simulation
	int threads_per_block = 512;
	int num_blocks = neurons_number / threads_per_block + 1;

	printf("Size of network: %i \n", neurons_number);
	printf("Start GPU with %d threads x %d blocks (Total: %d threads) \n",
	       threads_per_block, num_blocks, threads_per_block * num_blocks);

	// stuff variables for controlling C0/C1 activation
	int activated_C_ = 0;
	int shift_time_by_step = 0;
	int local_iter = 0;

	// the main simulation loop
	for (int sim_iter = 0; sim_iter < sim_time_in_steps; sim_iter++) {
		// if flexor C0 activated, find the end of it and change to C1
		if (activated_C_ == 0) {
			if (local_iter != 0 && local_iter % steps_activation_C0 == 0) {
				activated_C_ = 1; // change to C1
				local_iter = 0;  // reset local time iterator
				shift_time_by_step += steps_activation_C0;  // add constant 125 ms
			}
		// if extensor C1 activated, find the end of it and change to C0
		} else {
			if (local_iter != 0 && local_iter % steps_activation_C1 == 0) {
				activated_C_ = 0;  // change to C0
				local_iter = 0;  // reset local time iterator
				shift_time_by_step += steps_activation_C1;  // add time equal to n_layers * 25 ms
			}
		}
		// update local iter (warning: can be resetted at C0/C1 activation)
		local_iter++;

		// invoke GPU kernel for neurons
		GPU_neurons_kernel<<<num_blocks, threads_per_block>>>(
		        gpu_v_m,
		        gpu_h,
		        gpu_m,
		        gpu_n,
		        gpu_g_exc,
		        gpu_g_inh,
		        gpu_has_spike,
		        gpu_nrn_ref_time,
		        gpu_nrn_ref_time_timer,
		        neurons_number,
		        shift_time_by_step,
		        activated_C_,
		        sim_iter);

		// invoke GPU kernel for synapses
		GPU_synapses_kernel<<<num_blocks, threads_per_block>>>(
                gpu_has_spike,
		        gpu_synapses_number,
		        gpu_synapses_post_nrn_id,
		        gpu_synapses_delay,
		        gpu_synapses_delay_timer,
		        gpu_synapses_weight,
		        gpu_g_exc,
		        gpu_g_inh,
		        activated_C_,
		        neurons_number);

		// copy data from GPU
		memcpyDtH<bool >(has_spike, gpu_has_spike, neurons_number);
		memcpyDtH<float>(v_m, gpu_v_m, neurons_number);
		memcpyDtH<bool>(has_spike, gpu_has_spike, neurons_number);
		memcpyDtH<int>(nrn_ref_time, gpu_nrn_ref_time, neurons_number);
		memcpyDtH<float>(g_exc, gpu_g_exc, neurons_number);
		memcpyDtH<float>(g_inh, gpu_g_inh, neurons_number);
		memcpyDtH<float>(h, gpu_h, neurons_number);
		memcpyDtH<float>(m, gpu_m, neurons_number);
		memcpyDtH<float>(n, gpu_n, neurons_number);

		// fill records arrays
		for(int tid = 0; tid < neurons_number; tid++) {
			int iter = sim_iter + tid * sim_time_in_steps;
			volt_rec[iter] = v_m[tid];
			h_rec[iter] = h[tid];
			m_rec[iter] = m[tid];
			n_rec[iter] = n[tid];
			g_exc_rec[iter] = g_exc[tid];
			g_inh_rec[iter] = g_inh[tid];

			if (has_spike[tid]) {
				spikes_rec[iter] = step_to_ms(sim_iter);
			} else {
				spikes_rec[iter] = -1;
			}
		}
	} // end of the simulation iteration loop

	cudaDeviceSynchronize();  // tell the CPU to halt further processing until the CUDA has finished doing its business
	cudaDeviceReset();  // remove all all device allocations (destroy a CUDA context)

	// save recorded data
	save_result(test_index,
	            volt_rec,
	            h_rec,
	            m_rec,
	            n_rec,
	            g_exc_rec,
	            g_inh_rec,
	            spikes_rec,
	            neurons_number,
	            full_save);
}

// runner
int main(int argc, char* argv[]) {
	// arg 1: test index (0 equal to test with i=0, 1 equal to tests with i=0,1)
	// arg 2: 0 - save only motoneurons V_m, 1 - save all results
	simulate(std::atoi(argv[1]), std::atoi(argv[2]));

	return 0;
}