#define COLOR_RED "\x1b[1;31m"
#define COLOR_GREEN "\x1b[1;32m"
#define COLOR_RESET "\x1b[0m"

#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <time.h>
#include <cmath>
#include <stdexcept>
#include <random>
#include <curand_kernel.h>
#include <chrono>

// for file writing
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unistd.h>

// my classes
#include "Group.cpp"

#ifdef __JETBRAINS_IDE__
#define __host__
#define __global__
#endif

using namespace std;

/**
 6 cm/s = 125 [ms] has 30 slices
15 cm/s = 50 [ms] has 15 slices
21 cm/s = 25 [ms] has 6 slices

References:
  [1] https://en.wikipedia.org/wiki/Hodgkin-Huxley_model

**/

// parameters for variability of the simulation
const int skin_stim_time = 25;       // [ms] time of stimulating sensory (based on speed)
const int EES_FREQ = 40;             // [hz] spike frequency of EES
const int LEG_STEPS = 3;             // [step] number of full cycle steps
const float T_SIMULATION = 11 * skin_stim_time * LEG_STEPS;      // [ms] simulation time
const float INH_COEF = 1.0;          // strength coefficient of inhibitory synapses
const float SIM_STEP = 0.025;        // [s] simulation step
const bool is_4pedal = false;        // True if it is a quadrupedal simulation
const bool has_5HT = false;          // True if nuclei has 5-HT receptors

// stuff variables
unsigned int global_id = 0;          // iter to count neurons one by one
const int syn_outdegree = 27;        // synapse number outgoing from one neuron
const int neurons_in_ip = 196;       // number of neurons in interneuronal pool
const int neurons_in_aff_ip = 196;       // number of neurons in interneuronal pool
const int neurons_in_moto = 169;     // motoneurons number
const int neurons_in_group = 20;     // number of neurons in a group
const int neurons_in_afferent = 120; // number of neurons in afferent

// neuron parameters
const float g_Na = 20000.0;          // [nS] Maximal conductance of the Sodium current
const float g_K = 6000.0;            // [nS] Maximal conductance of the Potassium current
const float g_L = 30.0;              // [nS] Conductance of the leak current
const float C_m = 200.0;             // [pF] Capacity of the membrane
const float E_Na = 50.0;             // [mV] Reversal potential for the Sodium current
const float E_K = -100.0;            // [mV] Reversal potential for the Potassium current
const float E_L = -72.0;             // [mV] Reversal potential for the leak current
const float E_ex = 0.0;              // [mV] Reversal potential for excitatory input
const float E_in = -80.0;            // [mV] Reversal potential for inhibitory input
const float tau_syn_exc = 0.2;       // [ms] Decay time of excitatory synaptic current (ms)
const float tau_syn_inh = 2.0;       // [ms] Decay time of inhibitory synaptic current (ms)
const float V_adj = -63.0;           // adjusts threshold to around -50 mV
const float g_bar = 1500;            // [nS] the maximal possible conductivity

unsigned int synapses_number = 0;

// calculate spike frequency in steps [steps]
const unsigned int ees_spike_each_step = (unsigned int)(1000 / EES_FREQ / SIM_STEP);
// calculate steps activation of C0 and C1
const unsigned int steps_activation_C0 = (unsigned int)(5 * skin_stim_time / SIM_STEP);
const unsigned int steps_activation_C1 = (unsigned int)(6 * skin_stim_time / SIM_STEP);
// calculate how much steps in simulation time [steps]
const unsigned int sim_time_in_steps = (unsigned int)(T_SIMULATION / SIM_STEP);

// struct for human-readable initialization of connectomes
struct SynapseMetadata {
	unsigned int post_id;           // post neuron ID
	unsigned int synapse_delay;     // [step] synaptic delay of the synapse (axonal delay is included to this delay)
	float synapse_weight;  // [nS] synaptic weight. Interpreted as changing conductivity of neuron membrane

	SynapseMetadata() = default;
	SynapseMetadata(int post_id, float synapse_delay, float synapse_weight){
		this->post_id = post_id;
		this->synapse_delay = static_cast<int>(synapse_delay * (1 / SIM_STEP) + 0.5);  // round
		this->synapse_weight = synapse_weight;
	}
};

// struct for human-readable initialization of connectomes
struct Metadata {
	Group group;
	float* voltage_array = new float[sim_time_in_steps];  // [mV] array of membrane potential
	float* g_exc = new float[sim_time_in_steps];          // [nS] array of excitatory conductivity
	float* g_inh = new float[sim_time_in_steps];          // [nS] array of inhibition conductivity
	vector<float> spike_vector;                           // [ms] spike times

	Metadata() = default;
	Metadata(Group group){
		this->group = group;
	}
};

// keep all pointers to the groups to compute their mean voltage, g_exc and etc.
vector<Metadata> all_groups;

// form structs of neurons global ID and groups name
Group form_group(string group_name, int nrns_in_group = neurons_in_group) {
	Group group = Group();
	group.group_name = group_name;     // name of a neurons group
	group.id_start = global_id;        // first ID in the group
	group.id_end = global_id + nrns_in_group - 1;  // the latest ID in the group
	group.group_size = nrns_in_group;  // size of the neurons group

	global_id += nrns_in_group;

	printf("Formed %s IDs [%d ... %d] = %d\n", group_name.c_str(), global_id - nrns_in_group, global_id - 1, nrns_in_group);

	all_groups.push_back(Metadata(group));
	return group;
}

Group EES = form_group("EES");
Group E1 = form_group("E1");
Group E2 = form_group("E2");
Group E3 = form_group("E3");
Group E4 = form_group("E4");
Group E5 = form_group("E5");

Group CV1 = form_group("CV1", 1);
Group CV2 = form_group("CV2", 1);
Group CV3 = form_group("CV3", 1);
Group CV4 = form_group("CV4", 1);
Group CV5 = form_group("CV5", 1);
Group CD4 = form_group("CD4", 1);
Group CD5 = form_group("CD5", 1);

Group OM1_0 = form_group("OM1_0");
Group OM1_1 = form_group("OM1_1");
Group OM1_2_E = form_group("OM1_2_E");
Group OM1_2_F = form_group("OM1_2_F");
Group OM1_3 = form_group("OM1_3");

Group OM2_0 = form_group("OM2_0");
Group OM2_1 = form_group("OM2_1");
Group OM2_2_E = form_group("OM2_2_E");
Group OM2_2_F = form_group("OM2_2_F");
Group OM2_3 = form_group("OM2_3");

Group OM3_0 = form_group("OM3_0");
Group OM3_1 = form_group("OM3_1");
Group OM3_2_E = form_group("OM3_2_E");
Group OM3_2_F = form_group("OM3_2_F");
Group OM3_3 = form_group("OM3_3");

Group OM4_0 = form_group("OM4_0");
Group OM4_1 = form_group("OM4_1");
Group OM4_2_E = form_group("OM4_2_E");
Group OM4_2_F = form_group("OM4_2_F");
Group OM4_3 = form_group("OM4_3");

Group OM5_0 = form_group("OM5_0");
Group OM5_1 = form_group("OM5_1");
Group OM5_2_E = form_group("OM5_2_E");
Group OM5_2_F = form_group("OM5_2_F");
Group OM5_3 = form_group("OM5_3");

Group MN_E = form_group("MN_E", neurons_in_moto);
Group MN_F = form_group("MN_F", neurons_in_moto);

Group Ia_E_aff = form_group("Ia_E_aff", neurons_in_afferent);
Group Ia_F_aff = form_group("Ia_F_aff", neurons_in_afferent);

Group R_E = form_group("R_E");
Group R_F = form_group("R_F");

Group Ia_E_pool = form_group("Ia_E_pool", neurons_in_aff_ip);
Group Ia_F_pool = form_group("Ia_F_pool", neurons_in_aff_ip);

Group eIP_E = form_group("eIP_E", neurons_in_ip);
Group eIP_F = form_group("eIP_F", neurons_in_ip);

Group iIP_E = form_group("iIP_E", neurons_in_ip);
Group iIP_F = form_group("iIP_F", neurons_in_ip);

// global vectors of SynapseMetadata of synapses for each neuron
vector<vector<SynapseMetadata>> metadatas(global_id, vector<SynapseMetadata>());

__host__
int ms_to_step(float ms) { return (int)(ms / SIM_STEP); }

__host__
float step_to_ms(int step) { return step * SIM_STEP; }

__global__
void neurons_kernel(float *V_m,
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
                    int early_activated_C_,
                    int sim_iter,
                    int *begin_C_spiking,
                    int *end_C_spiking,
                    int decrease_lvl_Ia_spikes){
	// get ID of the thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Each thread gets same seed, a different sequence number, no offset
	curandState localState;
	curand_init(sim_iter, tid, 0, &localState);

	// Ia aff extensor/flexor IDs [965 ... 1204], control spike number of Ia afferent by resetting neuron current
	if (965 <= tid && tid <= 1204) {
		// rule for the 2nd level
		if (decrease_lvl_Ia_spikes == 1 && tid % 3 == 0) {
			// reset current of 1/3 of neurons
			g_exc[tid] = 0;  // set maximal inhibitory conductivity
		} else {
			// rule for the 3rd level
			if (decrease_lvl_Ia_spikes == 2 && tid % 2 == 0) {
				// reset current of 1/2 of neurons
				g_exc[tid] = 0;  // set maximal inhibitory conductivity
			}
		}
	}

	// generating spikes for EES
	if (0 <= tid && tid <= 19 && (sim_iter % ees_spike_each_step == 0)) {
		g_exc[tid] = g_bar;  // set spike state
	}

	__syncthreads();

	// ignore threads which ID is greater than neurons number
	if (tid < neurons_number) {
		// reset spike flag of the current neuron before calculations
		has_spike[tid] = false;

		if (activated_C_ == 0 && early_activated_C_ == 0 && 2225 <= tid && tid <= 2420 && (sim_iter % 10 == 0)) {
			// normal distribution of choosing which tid's neuron will spike at each 10th step of simulation (0.25ms)
			has_spike[2225 + static_cast<int>(196 * curand_uniform(&localState))] = true;
		}
		// Skin stimulations
		if (activated_C_ == 1) {
			// CV1
			if (tid == 120 && shifted_sim_iter > begin_C_spiking[0] && shifted_sim_iter < end_C_spiking[0] && curand_uniform(&localState) >= 0.5) {
				has_spike[tid] = true;
			}
			// CV2
			if (tid == 121 && shifted_sim_iter > begin_C_spiking[1] && shifted_sim_iter < end_C_spiking[1] && curand_uniform(&localState) >= 0.5) {
				has_spike[tid] = true;
			}
			// CV3
			if (tid == 122 && shifted_sim_iter > begin_C_spiking[2] && shifted_sim_iter < end_C_spiking[2] && curand_uniform(&localState) >= 0.5) {
				has_spike[tid] = true;
			}
			// CV4
			if (tid == 123 && shifted_sim_iter > begin_C_spiking[3] && shifted_sim_iter < end_C_spiking[3] && curand_uniform(&localState) >= 0.5) {
				has_spike[tid] = true;
			}
			// CV5
			if (tid == 124 && shifted_sim_iter > begin_C_spiking[4] && shifted_sim_iter < end_C_spiking[4] && curand_uniform(&localState) >= 0.5) {
				has_spike[tid] = true;
			}
		}

		// the maximal value of input current
		if (g_exc[tid] > g_bar)
			g_exc[tid] = g_bar;
		if (g_inh[tid] > g_bar)
			g_inh[tid] = g_bar;

		if (V_m[tid] > 100)
			V_m[tid] = 100;
		if (V_m[tid] < -100)
			V_m[tid] = -100;

		// use temporary V variable as V_m with adjust
		const float V = V_m[tid] - V_adj;

		// transition rates between open and closed states of the potassium channels
		float alpha_n = 0.032 * (15.0 - V) / (exp((15.0 - V) / 5.0) - 1.0);
		if (isnan(alpha_n)) alpha_n = 0;
		float beta_n = 0.5 * exp((10.0 - V) / 40.0);
		if (isnan(beta_n)) beta_n = 0;

		// transition rates between open and closed states of the activation of sodium channels
		float alpha_m = 0.32 * (13.0 - V) / (exp((13.0 - V) / 4.0) - 1.0);
		if (isnan(alpha_m)) alpha_m = 0;
		float beta_m = 0.28 * (V - 40.0) / (exp((V - 40.0) / 5.0) - 1.0);
		if (isnan(beta_m)) beta_m = 0;

		// transition rates between open and closed states of the inactivation of sodium channels
		float alpha_h = 0.128 * exp((17.0 - V) / 18.0);
		if (isnan(alpha_h)) alpha_h = 0;
		float beta_h = 4.0 / (1.0 + exp((40.0 - V) / 5.0));
		if (isnan(beta_h)) beta_h = 0;

		// re-calculate activation variables
		n[tid] += (alpha_n - (alpha_n + beta_n) * n[tid]) * SIM_STEP;
		m[tid] += (alpha_m - (alpha_m + beta_m) * m[tid]) * SIM_STEP;
		h[tid] += (alpha_h - (alpha_h + beta_h) * h[tid]) * SIM_STEP;

		if (n[tid] > 1) printf("n > 1\n");
		if (n[tid] < 0) printf("n < 0\n");

		if (m[tid] > 1) printf("m > 1\n");
		if (m[tid] < 0) printf("m < 0\n");

		if (h[tid] > 1) printf("h > 1\n");
		if (h[tid] < 0) printf("h < 0\n");

		// ionic currents
		float I_NA = g_Na * pow(m[tid], 3) * h[tid] * (V_m[tid] - E_Na);
		float I_K = g_K * pow(n[tid], 4) * (V_m[tid] - E_K);
		float I_L = g_L * (V_m[tid] - E_L);
		float I_syn_exc = g_exc[tid] * (V_m[tid] - E_ex);
		float I_syn_inh = g_inh[tid] * (V_m[tid] - E_in);

		float dV;
		// if neuron in the refractory state -- ignore synaptic inputs. Re-calculate membrane potential
		if (nrn_ref_time_timer[tid] > 0) {
			dV = -(I_L + I_K + I_NA) / C_m * SIM_STEP;
		} else {
			dV = -(I_L + I_K + I_NA + I_syn_exc + 4 * I_syn_inh) / C_m * SIM_STEP;
		}

		V_m[tid] += dV;

		// re-calculate conductance
		g_exc[tid] += -g_exc[tid] / tau_syn_exc * SIM_STEP;
		g_inh[tid] += -g_inh[tid] / tau_syn_inh * SIM_STEP;

		if (V_m[tid] > 100)
			V_m[tid] = 100;
		if (V_m[tid] < -100)
			V_m[tid] = -100;

		// (threshold && not in refractory period)
		if (V_m[tid] >= ((1637 <= tid && tid <= 1832)? -60 : -50) && nrn_ref_time_timer[tid] == 0) {
			has_spike[tid] = true;  // set spike state. It will be used in the "synapses_kernel"
			nrn_ref_time_timer[tid] = nrn_ref_time[tid];  // set the refractory period
		}

		// update the refractory period timer
		if (nrn_ref_time_timer[tid] > 0)
			nrn_ref_time_timer[tid]--;
	}
}

__global__
void synapses_kernel(bool *neuron_has_spike,     // array of bools -- is neuron has spike or not
                     float *neuron_g_exc,        // array of excitatory conductivity per neuron (changable)
                     float *neuron_g_inh,        // array of inhibitory conductivity per neuron (changable)
                     int *synapses_pre_nrn_id,   // array of pre neurons ID per synapse
                     int *synapses_post_nrn_id,  // array of post neurons ID per synapse
                     int *synapses_delay,        // array of synaptic delay per synapse
                     int *synapses_delay_timer,  // array as above but changable
                     float *synapses_weight,     // array of synaptic weight per synapse
                     int synapses_number){       // number of synapses

	// get ID of the thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// ignore threads which ID is greater than neurons number
	if (tid < synapses_number) {
		// add synaptic delay if neuron has spike
		if (synapses_delay_timer[tid] == -1 && neuron_has_spike[synapses_pre_nrn_id[tid]]) {
			synapses_delay_timer[tid] = synapses_delay[tid];
		}
		// if synaptic delay is zero it means the time when synapse increase I by synaptic weight
		if (synapses_delay_timer[tid] == 0) {
			// post neuron ID = synapses_post_nrn_id[tid][syn_id], thread-safe (!)
			if (synapses_weight[tid] >= 0) {
				atomicAdd(&neuron_g_exc[synapses_post_nrn_id[tid]], synapses_weight[tid]);
			} else {
				// remove negative sign
				atomicAdd(&neuron_g_inh[synapses_post_nrn_id[tid]], -synapses_weight[tid]);
			}
			// make synapse timer a "free" for next spikes
			synapses_delay_timer[tid] = -1;
		}
		// update synapse delay timer
		if (synapses_delay_timer[tid] > 0) {
			synapses_delay_timer[tid]--;
		}
	}
}

void connect_one_to_all(Group pre_neurons, Group post_neurons, float syn_delay, float weight) {
	std::default_random_engine generator;
	generator.seed(time(NULL));
	std::normal_distribution<float> delay_distr(syn_delay, syn_delay / 5);
	std::normal_distribution<float> weight_distr(weight, weight / 10);

	for (int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++) {
		for (int post_id = post_neurons.id_start; post_id <= post_neurons.id_end; post_id++) {
			metadatas.at(pre_id).push_back(SynapseMetadata(post_id, delay_distr(generator), weight_distr(generator)));
		}
	}

	synapses_number += pre_neurons.group_size * post_neurons.group_size;

	printf("Connect %s to %s [one_to_all] (1:%d). Total: %d W=%.2f, D=%.1f\n", pre_neurons.group_name.c_str(),
	       post_neurons.group_name.c_str(), post_neurons.group_size, pre_neurons.group_size * post_neurons.group_size,
	       weight, syn_delay);
}

void connect_fixed_outdegree(Group pre_neurons, Group post_neurons,
                             float syn_delay, float syn_weight, int outdegree= syn_outdegree, bool no_distr= false) {
	// connect neurons with uniform distribution and normal distributon for syn delay and syn_weight
	default_random_engine generator;
	generator.seed(time(NULL));
	uniform_int_distribution<int> id_distr(post_neurons.id_start, post_neurons.id_end);
	normal_distribution<float> delay_distr_gen(syn_delay, syn_delay / 5);
	normal_distribution<float> weight_distr_gen(syn_weight, syn_weight / 10);

	#ifdef DEBUG
	printf("pre group %s (%d, %d) to post %s (%d, %d)\n",
	       pre_neurons.group_name.c_str(),
	       pre_neurons.id_start,
	       pre_neurons.id_end,
	       post_neurons.group_name.c_str(),
	       post_neurons.id_start,
	       post_neurons.id_end);
	#endif

	synapses_number += pre_neurons.group_size * outdegree;

	for (int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++) {
		for (int i = 0; i < outdegree; i++) {
			int rand_post_id = id_distr(generator);
			float syn_delay_distr = delay_distr_gen(generator);
			if (syn_delay_distr <= 0.2) {
				syn_delay_distr = 0.2;
			}
			float syn_weight_distr = weight_distr_gen(generator);
			#ifdef DEBUG
			printf("syn_weight %f (%f), delay %f (%f) \n", syn_weight_distr, syn_weight, syn_delay_distr, syn_delay);
			#endif
			if (no_distr)
				metadatas.at(pre_id).push_back(SynapseMetadata(rand_post_id, syn_delay, syn_weight));
			else
				metadatas.at(pre_id).push_back(SynapseMetadata(rand_post_id, syn_delay_distr, syn_weight_distr));
		}
	}

	printf("Connect %s to %s [fixed_outdegree] (1:%d). Total: %d W=%.2f, D=%.1f\n",
		   pre_neurons.group_name.c_str(), post_neurons.group_name.c_str(), outdegree, pre_neurons.group_size * outdegree, syn_weight, syn_delay);
}

void init_connectomes() {
	float quadru_coef = is_4pedal? 0.5 : 1;
	float sero_coef = has_5HT? 1.5 : 1;

	// input from EES
	connect_fixed_outdegree(EES, E1, 1, 500, syn_outdegree, true);
	connect_fixed_outdegree(E1, E2, 1, 200, syn_outdegree, true);
	connect_fixed_outdegree(E2, E3, 1, 200, syn_outdegree, true);
	connect_fixed_outdegree(E3, E4, 1, 200, syn_outdegree, true);
	connect_fixed_outdegree(E4, E5, 1, 200, syn_outdegree, true);

	connect_one_to_all(CV1, iIP_E, 0.5, 50);
	connect_one_to_all(CV2, iIP_E, 0.5, 50);
	connect_one_to_all(CV3, iIP_E, 0.5, 50);
	connect_one_to_all(CV4, iIP_E, 0.5, 50);
	connect_one_to_all(CV5, iIP_E, 0.5, 50);

	/// OM 1
	// input from EES group 1
	connect_fixed_outdegree(E1, OM1_0, 1, 16, syn_outdegree, true); // 17
	// input from sensory
	connect_one_to_all(CV1, OM1_0, 0.5, 10); // 18
	connect_one_to_all(CV2, OM1_0, 0.5, 10); // 18
	// [inhibition]
	connect_one_to_all(CV3, OM1_3, 1, 80);
	connect_one_to_all(CV4, OM1_3, 1, 80);
	connect_one_to_all(CV5, OM1_3, 1, 80);
	// inner connectomes
	connect_fixed_outdegree(OM1_0, OM1_1, 1, 50);
	connect_fixed_outdegree(OM1_1, OM1_2_E, 1, 27); // 24
	connect_fixed_outdegree(OM1_1, OM1_2_F, 1, 30); // 23
	connect_fixed_outdegree(OM1_1, OM1_3, 1, 3);
	connect_fixed_outdegree(OM1_2_E, OM1_1, 2.5, 27); // 22
	connect_fixed_outdegree(OM1_2_F, OM1_1, 2.5, 30); // 22
	connect_fixed_outdegree(OM1_2_E, OM1_3, 1, 4); // 3
	connect_fixed_outdegree(OM1_2_F, OM1_3, 1, 4); // 3
	connect_fixed_outdegree(OM1_3, OM1_1, 1, -5 * INH_COEF);
	connect_fixed_outdegree(OM1_3, OM1_2_E, 1, -50 * INH_COEF);
	connect_fixed_outdegree(OM1_3, OM1_2_F, 1, -0.1 * INH_COEF);
	// output to OM2
	connect_fixed_outdegree(OM1_2_F, OM2_2_F, 1, 50);
	// output to IP
	connect_fixed_outdegree(OM1_2_E, eIP_E, 1, 16, neurons_in_ip); // 13
	connect_fixed_outdegree(OM1_2_F, eIP_F, 1, 7, neurons_in_ip);

	/// OM 2
	// input from EES group 2
	connect_fixed_outdegree(E2, OM2_0, 3, 8);
	// input from sensory [CV]
	connect_one_to_all(CV2, OM2_0, 0.5, 10.5);
	connect_one_to_all(CV3, OM2_0, 0.5, 10.5);
	// [inhibition]
	connect_one_to_all(CV4, OM2_3, 1, 80);
	connect_one_to_all(CV5, OM2_3, 1, 80);
	// inner connectomes
	connect_fixed_outdegree(OM2_0, OM2_1, 1, 50);
	connect_fixed_outdegree(OM2_1, OM2_2_E, 1, 25);
	connect_fixed_outdegree(OM2_1, OM2_2_F, 1, 23);
	connect_fixed_outdegree(OM2_1, OM2_3, 1, 3);
	connect_fixed_outdegree(OM2_2_E, OM2_1, 2.5, 25);
	connect_fixed_outdegree(OM2_2_F, OM2_1, 2.5, 20);
	connect_fixed_outdegree(OM2_2_E, OM2_3, 1, 3);
	connect_fixed_outdegree(OM2_2_F, OM2_3, 1, 3);
	connect_fixed_outdegree(OM2_3, OM2_1, 1, -70 * INH_COEF);
	connect_fixed_outdegree(OM2_3, OM2_2_E, 1, -70 * INH_COEF);
	connect_fixed_outdegree(OM2_3, OM2_2_F, 1, -70 * INH_COEF);
	// output to OM3
	connect_fixed_outdegree(OM2_2_F, OM3_2_F, 1, 50);
	// output to IP
	connect_fixed_outdegree(OM2_2_E, eIP_E, 3, 10, neurons_in_ip); // 50
	connect_fixed_outdegree(OM2_2_F, eIP_F, 2, 7, neurons_in_ip);

	/// OM 3
	// input from EES group 3
	connect_fixed_outdegree(E3, OM3_0, 1, 8);
	// input from sensory [CV]
	connect_one_to_all(CV3, OM3_0, 0.5, 10.5);
	connect_one_to_all(CV4, OM3_0, 0.5, 10.5);
	// [INH]
	connect_one_to_all(CV5, OM3_3, 1, 80);
	// input from sensory [CD]
	connect_one_to_all(CD4, OM3_0, 1, 11);
	// inner connectomes
	connect_fixed_outdegree(OM3_0, OM3_1, 1, 50);
	connect_fixed_outdegree(OM3_1, OM3_2_E, 1, 25);
	connect_fixed_outdegree(OM3_1, OM3_2_F, 1, 30);
	connect_fixed_outdegree(OM3_1, OM3_3, 1, 3);
	connect_fixed_outdegree(OM3_2_E, OM3_1, 2.5, 25);
	connect_fixed_outdegree(OM3_2_F, OM3_1, 2.5, 20);
	connect_fixed_outdegree(OM3_2_E, OM3_3, 1, 3);
	connect_fixed_outdegree(OM3_2_F, OM3_3, 1, 3);
	connect_fixed_outdegree(OM3_3, OM3_1, 1, -70 * INH_COEF);
	connect_fixed_outdegree(OM3_3, OM3_2_E, 1, -70 * INH_COEF);
	connect_fixed_outdegree(OM3_3, OM3_2_F, 1, -70 * INH_COEF);
	// output to OM3
	connect_fixed_outdegree(OM3_2_F, OM4_2_F, 1, 50);
	connect_fixed_outdegree(OM3_2_E, eIP_E, 2, 9, neurons_in_ip);
	connect_fixed_outdegree(OM3_2_F, eIP_F, 3, 7, neurons_in_ip);

	/// OM 4
	// input from EES group 4
	connect_fixed_outdegree(E4, OM4_0, 2, 8);
	// input from sensory [CV]
	connect_one_to_all(CV4, OM4_0, 0.5, 10.5);
	connect_one_to_all(CV5, OM4_0, 0.5, 10.5);
	// input from sensory [CD]
	connect_one_to_all(CD4, OM4_0, 1, 11);
	connect_one_to_all(CD5, OM4_0, 1, 11);
	// inner connectomes
	connect_fixed_outdegree(OM4_0, OM4_1, 3, 50);
	connect_fixed_outdegree(OM4_1, OM4_2_E, 1, 25);
	connect_fixed_outdegree(OM4_1, OM4_2_F, 1, 23);
	connect_fixed_outdegree(OM4_1, OM4_3, 1, 3);
	connect_fixed_outdegree(OM4_2_E, OM4_1, 2.5, 25);
	connect_fixed_outdegree(OM4_2_F, OM4_1, 2.5, 20);
	connect_fixed_outdegree(OM4_2_E, OM4_3, 1, 3);
	connect_fixed_outdegree(OM4_2_F, OM4_3, 1, 3);
	connect_fixed_outdegree(OM4_3, OM4_1, 1, -70 * INH_COEF);
	connect_fixed_outdegree(OM4_3, OM4_2_E, 1, -70 * INH_COEF);
	connect_fixed_outdegree(OM4_3, OM4_2_F, 1, -70 * INH_COEF);
	// output to OM4
	connect_fixed_outdegree(OM4_2_F, OM5_2_F, 1, 50);
	connect_fixed_outdegree(OM4_2_E, eIP_E, 2, 11, neurons_in_ip); // 10
	connect_fixed_outdegree(OM4_2_F, eIP_F, 1, 7, neurons_in_ip);

	/// OM 5
	// input from EES group 5
	connect_fixed_outdegree(E5, OM5_0, 3, 8);
	// input from sensory [CV]
	connect_one_to_all(CV5, OM5_0, 0.5, 10.5);
	// input from sensory [CD]
	connect_one_to_all(CD5, OM5_0, 1, 11);
	// inner connectomes
	connect_fixed_outdegree(OM5_0, OM5_1, 1, 50);
	connect_fixed_outdegree(OM5_1, OM5_2_E, 1, 25);
	connect_fixed_outdegree(OM5_1, OM5_2_F, 1, 30);
	connect_fixed_outdegree(OM5_1, OM5_3, 1, 3);
	connect_fixed_outdegree(OM5_2_E, OM5_1, 2.5, 25);
	connect_fixed_outdegree(OM5_2_F, OM5_1, 2.5, 30);
	connect_fixed_outdegree(OM5_2_E, OM5_3, 1, 3);
	connect_fixed_outdegree(OM5_2_F, OM5_3, 1, 3);
	connect_fixed_outdegree(OM5_3, OM5_1, 1, -70 * INH_COEF);
	connect_fixed_outdegree(OM5_3, OM5_2_E, 1, -20 * INH_COEF);
	connect_fixed_outdegree(OM5_3, OM5_2_F, 1, -20 * INH_COEF);
	// output to IP
	connect_fixed_outdegree(OM5_2_E, eIP_E, 2, 10, neurons_in_ip); // 15
	connect_fixed_outdegree(OM5_2_F, eIP_F, 3, 7, neurons_in_ip);

	/// reflex arc
	connect_fixed_outdegree(iIP_E, eIP_F, 0.5, -50, neurons_in_ip);
	connect_fixed_outdegree(iIP_F, eIP_E, 0.5, -50, neurons_in_ip);

	connect_fixed_outdegree(iIP_E, OM1_2_F, 0.5, -20, neurons_in_ip);
	connect_fixed_outdegree(iIP_E, OM2_2_F, 0.5, -20, neurons_in_ip);
	connect_fixed_outdegree(iIP_E, OM3_2_F, 0.5, -20, neurons_in_ip);
	connect_fixed_outdegree(iIP_E, OM4_2_F, 0.5, -20, neurons_in_ip);

	connect_fixed_outdegree(EES, Ia_E_aff, 1, 500);
	connect_fixed_outdegree(EES, Ia_F_aff, 1, 500);

	connect_fixed_outdegree(eIP_E, MN_E, 2.5, 1.4, neurons_in_moto); // d1.2 / 1.5 2.0 - 11
	connect_fixed_outdegree(eIP_F, MN_F, 1, 11, neurons_in_moto);

	connect_fixed_outdegree(iIP_E, Ia_E_pool, 1, 10, neurons_in_ip);
	connect_fixed_outdegree(iIP_F, Ia_F_pool, 1, 10, neurons_in_ip);

	connect_fixed_outdegree(Ia_E_pool, MN_F, 1, -20, neurons_in_ip);
	connect_fixed_outdegree(Ia_E_pool, Ia_F_pool, 1, -1, neurons_in_ip);
	connect_fixed_outdegree(Ia_F_pool, MN_E, 1, -4, neurons_in_ip);
	connect_fixed_outdegree(Ia_F_pool, Ia_E_pool, 1, -1, neurons_in_ip);

	connect_fixed_outdegree(Ia_E_aff, MN_E, 2, 8, neurons_in_moto); // 6
	connect_fixed_outdegree(Ia_F_aff, MN_F, 2, 6, neurons_in_moto);

	connect_fixed_outdegree(MN_E, R_E, 2, 1);
	connect_fixed_outdegree(MN_F, R_F, 2, 1);

	connect_fixed_outdegree(R_E, MN_E, 2, -0.5, neurons_in_moto);
	connect_fixed_outdegree(R_E, R_F, 2, -1);

	connect_fixed_outdegree(R_F, MN_F, 2, -0.5, neurons_in_moto);
	connect_fixed_outdegree(R_F, R_E, 2, -1);
}

void save(int test_index, Metadata metadata){
	ofstream file;
	string file_name = "/dat/" + to_string(test_index) + "_" + metadata.group.group_name + ".dat";
	file.open("/home/alex/GitHub/memristive-spinal-cord/GRAS/matrix_solution/" + file_name);
	// save voltage
	for (int sim_iter = 0; sim_iter < sim_time_in_steps; sim_iter++)
		file << metadata.voltage_array[sim_iter] << " ";
	file << endl;

	// save g_exc
	for (int sim_iter = 0; sim_iter < sim_time_in_steps; sim_iter++)
		file << metadata.g_exc[sim_iter] << " ";
	file << endl;

	// save g_inh
	for (int sim_iter = 0; sim_iter < sim_time_in_steps; sim_iter++)
		file << metadata.g_inh[sim_iter] << " ";
	file << endl;

	// save spikes
	for (float &value: metadata.spike_vector) {
		file << value << " ";
	}
	file.close();

	printf("saved %s \n", metadata.group.group_name.c_str());
}

void save_result(int test_index, int full_save) {
	char cwd[256];
	getcwd(cwd, sizeof(cwd));
	printf("[Test #%d] Save %s results to: %s \n", test_index, (full_save == 0)? "only MOTO" : "ALL", cwd);

	for(Metadata metadata : all_groups) {
		if (full_save == 0) {
			if(metadata.group.group_name == "MN_E")
				save(test_index, metadata);
			if(metadata.group.group_name == "MN_F")
				save(test_index, metadata);
		} else {
			save(test_index, metadata);
		}
	}
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
void init_array(type *array, int size, type value) {
	for(int i = 0; i < size; i++)
		array[i] = value;
}

__host__
void simulate(int test_index, int full_save) {
	chrono::time_point<chrono::system_clock> simulation_t_start, simulation_t_end;
	const int neurons_number = static_cast<int>(metadatas.size());

	// init connectomes
	init_connectomes();

	/// CPU variables
	// neuron variables
	float nrn_n[neurons_number];             // dimensionless quantity [0 .. 1] of potassium channel activation
	float nrn_h[neurons_number];             // dimensionless quantity [0 .. 1] of sodium channel activation
	float nrn_m[neurons_number];             // dimensionless quantity [0 .. 1] of sodium channel inactivation
	float nrn_v_m[neurons_number];           // [mV] neuron membrane potential
	float nrn_g_exc[neurons_number];         // [nS] excitatory synapse exponential conductance
	float nrn_g_inh[neurons_number];         // [nS] inhibitory synapse exponential conductance
	bool nrn_has_spike[neurons_number];      // neuron state - has spike or not
	int nrn_ref_time[neurons_number];        // [step] neuron refractory time
	int nrn_ref_time_timer[neurons_number];  // [step] neuron refractory time timer

	int begin_C_spiking[5] = {ms_to_step(0),
	                          ms_to_step(skin_stim_time),
	                          ms_to_step(2 * skin_stim_time),
	                          ms_to_step(3 * skin_stim_time),
	                          ms_to_step(5 * skin_stim_time)};
	int end_C_spiking[5] = {ms_to_step(skin_stim_time - 0.1),
	                        ms_to_step(2 * skin_stim_time - 0.1),
	                        ms_to_step(3 * skin_stim_time - 0.1),
	                        ms_to_step(5 * skin_stim_time - 0.1),
	                        ms_to_step(6 * skin_stim_time - 0.1)};

	// fill arrays by initial data
	init_array<float>(nrn_n, neurons_number, 0);      // by default neurons have closed potassium channel
	init_array<float>(nrn_h, neurons_number, 1);      // by default neurons have opened sodium channel activation
	init_array<float>(nrn_m, neurons_number, 0);      // by default neurons have closed sodium channel inactivation
	init_array<float>(nrn_v_m, neurons_number, E_L);  // by default neurons have E_L membrane state at start
	init_array<float>(nrn_g_exc, neurons_number, 0);  // by default neurons have zero excitatory synaptic conductivity
	init_array<float>(nrn_g_inh, neurons_number, 0);  // by default neurons have zero inhibitory synaptic conductivity
	init_array<bool>(nrn_has_spike, neurons_number, false);         // by default neurons haven't spikes at start
	init_array<int>(nrn_ref_time, neurons_number, ms_to_step(3.0)); // by default neurons have 3ms refractory period
	init_array<int>(nrn_ref_time_timer, neurons_number, 0);         // by default neurons have ref_t timers as 0
	// rand_normal_init_array<float>(nrn_c_m, neurons_number, 200, 5)

	// synapse variables
	unsigned int syn_id = 0;
	int *synapses_pre_nrn_id = (int *) malloc(datasize<int>(synapses_number));
	int *synapses_post_nrn_id = (int *) malloc(datasize<int>(synapses_number));
	float *synapses_weight = (float *) malloc(datasize<float>(synapses_number));
	int *synapses_delay = (int *) malloc(datasize<int>(synapses_number));
	int *synapses_delay_timer = (int *) malloc(datasize<int>(synapses_number));

	// fill arrays of synapses
	for(int neuron_id = 0; neuron_id < neurons_number; neuron_id++) {
		// for each metadata of current neuron fill data in array to copy in future pointers to the GPU
		for(SynapseMetadata metadata : metadatas.at(neuron_id)) {
			synapses_pre_nrn_id[syn_id] = neuron_id;
			synapses_post_nrn_id[syn_id] = metadata.post_id;
			synapses_delay[syn_id] = metadata.synapse_delay;
			synapses_weight[syn_id] = metadata.synapse_weight;
			syn_id++;
		}
	}
	init_array<int>(synapses_delay_timer, synapses_number, -1);

	// neuron variables
	float* gpu_nrn_n;
	float* gpu_nrn_h;
	float* gpu_nrn_m;
	float* gpu_nrn_v_m;
	float* gpu_nrn_g_exc;
	float* gpu_nrn_g_inh;
	bool* gpu_nrn_has_spike;
	int* gpu_nrn_ref_time;
	int* gpu_nrn_ref_time_timer;

	// synapse variables
	int* gpu_syn_pre_nrn_id;
	int* gpu_syn_post_nrn_id;
	float* gpu_syn_weight;
	int* gpu_syn_delay;
	int* gpu_syn_delay_timer;

	int *gpu_begin_C_spiking;
	int *gpu_end_C_spiking;

	// allocate memory in the GPU
	cudaMalloc(&gpu_nrn_n, datasize<float>(neurons_number));
	cudaMalloc(&gpu_nrn_h, datasize<float>(neurons_number));
	cudaMalloc(&gpu_nrn_m, datasize<float>(neurons_number));
	cudaMalloc(&gpu_nrn_v_m, datasize<float>(neurons_number));
	cudaMalloc(&gpu_nrn_g_exc, datasize<float>(neurons_number));
	cudaMalloc(&gpu_nrn_g_inh, datasize<float>(neurons_number));
	cudaMalloc(&gpu_nrn_has_spike, datasize<bool>(neurons_number));
	cudaMalloc(&gpu_nrn_ref_time, datasize<int>(neurons_number));
	cudaMalloc(&gpu_nrn_ref_time_timer, datasize<int>(neurons_number));

	cudaMalloc(&gpu_syn_pre_nrn_id, datasize<int>(synapses_number));
	cudaMalloc(&gpu_syn_post_nrn_id, datasize<int>(synapses_number));
	cudaMalloc(&gpu_syn_weight, datasize<float>(synapses_number));
	cudaMalloc(&gpu_syn_delay, datasize<int>(synapses_number));
	cudaMalloc(&gpu_syn_delay_timer, datasize<int>(synapses_number));

	cudaMalloc(&gpu_begin_C_spiking, datasize<int>(5));
	cudaMalloc(&gpu_end_C_spiking, datasize<int>(5));

	// copy data from CPU to GPU
	memcpyHtD<float>(gpu_nrn_n, nrn_n, neurons_number);
	memcpyHtD<float>(gpu_nrn_h, nrn_h, neurons_number);
	memcpyHtD<float>(gpu_nrn_m, nrn_m, neurons_number);
	memcpyHtD<float>(gpu_nrn_v_m, nrn_v_m, neurons_number);
	memcpyHtD<float>(gpu_nrn_g_exc, nrn_g_exc, neurons_number);
	memcpyHtD<float>(gpu_nrn_g_inh, nrn_g_inh, neurons_number);
	memcpyHtD<bool>(gpu_nrn_has_spike, nrn_has_spike, neurons_number);
	memcpyHtD<int>(gpu_nrn_ref_time, nrn_ref_time, neurons_number);
	memcpyHtD<int>(gpu_nrn_ref_time_timer, nrn_ref_time_timer, neurons_number);

	memcpyHtD<int>(gpu_syn_pre_nrn_id, synapses_pre_nrn_id, synapses_number);
	memcpyHtD<int>(gpu_syn_post_nrn_id, synapses_post_nrn_id, synapses_number);
	memcpyHtD<float>(gpu_syn_weight, synapses_weight, synapses_number);
	memcpyHtD<int>(gpu_syn_delay, synapses_delay, synapses_number);
	memcpyHtD<int>(gpu_syn_delay_timer, synapses_delay_timer, synapses_number);

	memcpyHtD<int>(gpu_begin_C_spiking, begin_C_spiking, 5);
	memcpyHtD<int>(gpu_end_C_spiking, end_C_spiking, 5);

	// preparations for simulation
	int threads_per_block = 32;
	int nrn_num_blocks = neurons_number / threads_per_block + 1;
	int syn_num_blocks = synapses_number / threads_per_block + 1;

	int total_nrn_threads = threads_per_block * nrn_num_blocks;
	int total_syn_threads = threads_per_block * syn_num_blocks;

	printf("* * * Start GPU * * *\n");
	printf("Neurons kernel: %d threads (%.3f%% threads idle) [%d blocks x %d threads per block] mapped on %d neurons \n",
	       total_nrn_threads, (double)(total_nrn_threads - neurons_number) / total_nrn_threads * 100,
	       nrn_num_blocks, threads_per_block, neurons_number);
	printf("Synapses kernel: %d threads (%.3f%% threads idle) [%d blocks x %d threads per block] mapped on %d synapses \n",
	       total_syn_threads, (double)(total_syn_threads - synapses_number) / total_syn_threads * 100,
	       syn_num_blocks, threads_per_block, synapses_number);

	// stuff variables for controlling C0/C1 activation
	int activated_C_ = 0;
	int shift_time_by_step = 0;
	int local_iter = 0;
	int decrease_lvl_Ia_spikes;
	int early_activated_C_ = 0;

	simulation_t_start = chrono::system_clock::now();

	// the main simulation loop
	for (int sim_iter = 0; sim_iter < sim_time_in_steps; sim_iter++) {
		decrease_lvl_Ia_spikes = 0;

		// if flexor C0 activated, find the end of it and change to C1
		if (activated_C_ == 0) {
			if (local_iter != 0 && local_iter % steps_activation_C0 == 0) {
				activated_C_ = 1; // change to C1
				local_iter = 0;   // reset local time iterator
				shift_time_by_step += steps_activation_C0;  // add constant 125 ms
			}
			if (local_iter != 0 && (local_iter + 400) % steps_activation_C0 == 0) {
				early_activated_C_ = 1;
			}
			// if extensor C1 activated, find the end of it and change to C0
		} else {
			if (local_iter != 0 && local_iter % steps_activation_C1 == 0) {
				activated_C_ = 0; // change to C0
				local_iter = 0;   // reset local time iterator
				shift_time_by_step += steps_activation_C1;  // add time equal to n_layers * 25 ms
			}
			if (local_iter != 0 && (local_iter + 400) % steps_activation_C1 == 0) {
				early_activated_C_ = 0;
			}
		}

		int shifted_iter_time = sim_iter - shift_time_by_step;

		// CV1
		if ((begin_C_spiking[0] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[0])) {
			decrease_lvl_Ia_spikes = 2;
		} else {
			// CV2
			if ((begin_C_spiking[1] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[1])) {
				decrease_lvl_Ia_spikes = 1;
			} else {
				// CV3
				if ((begin_C_spiking[2] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[2])) {
					decrease_lvl_Ia_spikes = 0;
				} else {
					// CV4
					if ((begin_C_spiking[3] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[3])) {
						decrease_lvl_Ia_spikes = 1;
					} else {
						// CV5
						if ((begin_C_spiking[4] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[4])) {
							decrease_lvl_Ia_spikes = 2;
						}
					}
				}
			}
		}

		// update local iter (warning: can be resetted at C0/C1 activation)
		local_iter++;

		// invoke GPU kernel for neurons
		neurons_kernel<<<nrn_num_blocks, threads_per_block>>>(gpu_nrn_v_m,
		                                                      gpu_nrn_h,
		                                                      gpu_nrn_m,
		                                                      gpu_nrn_n,
		                                                      gpu_nrn_g_exc,
		                                                      gpu_nrn_g_inh,
		                                                      gpu_nrn_has_spike,
		                                                      gpu_nrn_ref_time,
		                                                      gpu_nrn_ref_time_timer,
		                                                      neurons_number,
		                                                      sim_iter - shift_time_by_step,
		                                                      activated_C_,
		                                                      early_activated_C_,
		                                                      sim_iter,
		                                                      gpu_begin_C_spiking,
		                                                      gpu_end_C_spiking,
		                                                      decrease_lvl_Ia_spikes);

		// copy data from GPU
		memcpyDtH<float>(nrn_v_m, gpu_nrn_v_m, neurons_number);
		memcpyDtH<float>(nrn_g_exc, gpu_nrn_g_exc, neurons_number);
		memcpyDtH<float>(nrn_g_inh, gpu_nrn_g_inh, neurons_number);
		memcpyDtH<bool>(nrn_has_spike, gpu_nrn_has_spike, neurons_number);

		// fill records arrays
		for(Metadata &metadata : all_groups) {
			float nrn_mean_volt = 0;
			float nrn_mean_g_exc = 0;
			float nrn_mean_g_inh = 0;

			for(int tid = metadata.group.id_start; tid <= metadata.group.id_end; tid++) {
				nrn_mean_volt += nrn_v_m[tid];
				nrn_mean_g_exc += nrn_g_exc[tid];
				nrn_mean_g_inh += nrn_g_inh[tid];
				if (nrn_has_spike[tid]) {
					metadata.spike_vector.push_back(step_to_ms(sim_iter) + 0.25);
				}
			}
			metadata.voltage_array[sim_iter] = nrn_mean_volt / metadata.group.group_size;
			metadata.g_exc[sim_iter] = nrn_mean_g_exc / metadata.group.group_size;
			metadata.g_inh[sim_iter] = nrn_mean_g_inh / metadata.group.group_size;
		}

		// invoke GPU kernel for synapses
		synapses_kernel<<<syn_num_blocks, threads_per_block>>>(gpu_nrn_has_spike,
		                                                       gpu_nrn_g_exc,
		                                                       gpu_nrn_g_inh,
		                                                       gpu_syn_pre_nrn_id,
		                                                       gpu_syn_post_nrn_id,
		                                                       gpu_syn_delay,
		                                                       gpu_syn_delay_timer,
		                                                       gpu_syn_weight,
		                                                       synapses_number);
	} // end of the simulation iteration loop

	simulation_t_end = chrono::system_clock::now();

	cudaDeviceSynchronize();  // tell the CPU to halt further processing until the CUDA has finished doing its business
	cudaDeviceReset();  // remove all all device allocations (destroy a CUDA context)

	// save recorded data
	save_result(test_index, full_save);

	auto sim_time_diff = chrono::duration_cast<chrono::milliseconds>(simulation_t_end - simulation_t_start).count();
	printf("Elapsed %li ms (measured) | T_sim = %.2f ms\n", sim_time_diff, T_SIMULATION);
	printf("%s x%f\n", (double)(T_SIMULATION / sim_time_diff) > 1?
	                   COLOR_GREEN "faster" COLOR_RESET: COLOR_RED "slower" COLOR_RESET, T_SIMULATION / sim_time_diff);
}

// runner
int main(int argc, char* argv[]) {
	// arg 1: test index (0 equal to test with i=0, 1 equal to tests with i=0,1)
	// arg 2: 0 - save only motoneurons V_m, 1 - save all results
	simulate(atoi(argv[1]), atoi(argv[2]));

	return 0;
}