#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <ctime>
#include <stdexcept>

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unistd.h>

#include "Group.cpp"

#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __global__
#define __shared__
#endif

using namespace std;

const unsigned int neurons_in_group = 20;
const unsigned int syn_outdegree = 27;
const unsigned int neurons_in_moto = 169;
const unsigned int neurons_in_ip = 196;
const unsigned int neurons_in_afferent = 196;

const float INH_COEF = 1.0;
const int EES_FREQ = 40;
const float speed_to_time = 25;

// 6 cms = 125
// 15 cms = 50
// 21 cms = 25

// stuff variable
unsigned int global_id = 0;

const float T_sim = 1000;
const float sim_step = 0.25;
const unsigned int sim_step_time = (unsigned int)(T_sim / sim_step);

__host__
int ms_to_step(float ms) { return (int)(ms / sim_step); }

struct Metadata{
	Metadata() = default;
	Metadata(int post_id, float synapse_ref_t, float synapse_weight){
		this->post_id = post_id;
		this->synapse_ref_t = static_cast<int>(synapse_ref_t * (1 / sim_step));
		this->synapse_weight = synapse_weight;
	}
	int post_id;
	int synapse_ref_t;
	float synapse_weight;
};

Group form_group(string group_name, int nrns_in_group = neurons_in_group, float ref_t = 3.0) {
	Group group = Group();

	group.group_name = group_name;
	group.id_start = global_id;
	group.id_end = global_id + nrns_in_group - 1;
	group.group_size = nrns_in_group;

	printf("Formed %s IDs [%d ... %d] = %d\n",
		   group_name.c_str(), global_id, global_id + nrns_in_group - 1, nrns_in_group);

	global_id += nrns_in_group;

	return group;
}

// segment start
Group C1 = form_group("C1");
Group C2 = form_group("C2");

Group C3 = form_group("C3");
Group C4 = form_group("C4");
Group C5 = form_group("C5");

Group D1_1 = form_group("D1_1");
Group D1_2 = form_group("D1_2");
Group D1_3 = form_group("D1_3");
Group D1_4 = form_group("D1_4");

Group D2_1 = form_group("D2_1");
Group D2_2 = form_group("D2_2");
Group D2_3 = form_group("D2_3");
Group D2_4 = form_group("D2_4");

Group D3_1 = form_group("D3_1");
Group D3_2 = form_group("D3_2");
Group D3_3 = form_group("D3_3");
Group D3_4 = form_group("D3_4");

Group D4_1 = form_group("D4_1");
Group D4_2 = form_group("D4_2");
Group D4_3 = form_group("D4_3");
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

Group G3_1 = form_group("G3_1");
Group G3_2 = form_group("G3_2");
Group G3_3 = form_group("G3_3");

Group G4_1 = form_group("G4_1");
Group G4_2 = form_group("G4_2");
Group G4_3 = form_group("G4_3");

Group G5_1 = form_group("G5_1");
Group G5_2 = form_group("G5_2");
Group G5_3 = form_group("G5_3");

Group IP_E = form_group("IP_E", neurons_in_ip);
Group MP_E = form_group("MP_E", neurons_in_moto);
Group EES = form_group("EES");

Group inh_group3 = form_group("inh_group3");
Group inh_group4 = form_group("inh_group4");
Group inh_group5 = form_group("inh_group5");

Group ees_group1 = form_group("ees_group1");
Group ees_group2 = form_group("ees_group2");
Group ees_group3 = form_group("ees_group3");
Group ees_group4 = form_group("ees_group4");
Group Ia = form_group("Ia", neurons_in_afferent);

vector<vector<Metadata>> metadatas(global_id, vector<Metadata>());

bool* has_multimeter;
bool* has_generator;
int* begin_spiking;
int* end_spiking;
int* spiking_per_step;

__global__
void sim_GPU(float* old_v,
             float* old_u,
             float* nrn_current,
             float step_dt,
             int* refractory_time,
             int* refractory_time_timer,
             int* synapses_number,
             bool* has_spike,
             int** post_ids,
             int** synapses_delay,
             int** synapses_delay_timer,
             float** synapses_weight,
             unsigned int nrn_size,
             bool* has_generator,
             bool* has_multimeter,
             float* multimeter_result,
             int* begin_spiking,
             int* end_spiking,
             int* spiking_per_step) {

	__shared__ float mm_result;

	// Parameters (const)
	const float C = 100.0f;        // [pF] membrane capacitance
	const float V_rest = -72.0f;   // [mV] resting membrane potential
	const float V_thld = -55.0f;   // [mV] spike threshold
	const float k = 0.7f;          // [pA * mV-1] constant ("1/R")
	const float b = 0.2f;          // [pA * mV-1] sensitivity of U_m to the sub-threshold fluctuations of the V_m
	const float a = 0.02f;         // [ms-1] time scale of the recovery variable U_m. Higher a, the quicker recovery
	const float c = -80.0f;        // [mV] after-spike reset value of V_m
	const float d = 6.0f;          // [pA] after-spike reset value of U_m
	const float V_peak = 35.0f;    // [mV] spike cutoff value
	const float sim_step = 0.25;

	for (int sim_iter = 0; sim_iter < sim_step_time; sim_iter++) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		if(tid == 0)
			mm_result = 0;
		__syncthreads();

		for (; tid < nrn_size; tid += blockDim.x * gridDim.x) {
			if (has_generator[tid] &&
				sim_iter >= begin_spiking[tid] &&
				sim_iter < end_spiking[tid] &&
				(sim_iter % spiking_per_step[tid] == 0)) {
				nrn_current[tid] = 10000;
			}

			float V_old = old_v[tid];
			float U_old = old_u[tid];
			float I_syn = (refractory_time_timer[tid] > 0) ? 0 : nrn_current[tid];

			// absolute refractory period : calculate V_m and U_m WITHOUT synaptic weight
			// action potential : calculate V_m and U_m WITH synaptic weight
			float V_m = V_old + step_dt * (k * (V_old - V_rest) * (V_old - V_thld) - U_old + I_syn) / C;
			float U_m = U_old + step_dt * a * (b * (V_old - V_rest) - U_old);

			if (V_m < c)
				V_m = c;

			// save the V_m and I value every iter step if has multimeter
			if (has_multimeter[tid]) {
				atomicAdd(&mm_result, V_m);
			}

			// threshold crossing (spike)
			if (V_m >= V_peak) {
				// set spike status
				has_spike[tid] = true;
				// redefine V_old and U_old
				old_v[tid] = c;
				old_u[tid] += d;
				// set the refractory period
				refractory_time_timer[tid] = refractory_time[tid];
			} else {
				// redefine V_old and U_old
				old_v[tid] = V_m;
				old_u[tid] = U_m;
			}

			int *ptr_delay_timers = synapses_delay_timer[tid];

			for (int i = 0; i < synapses_number[tid]; i++) {
				if (has_spike[tid] && ptr_delay_timers[i] == -1) {
					ptr_delay_timers[i] = synapses_delay[tid][i];
				}
				if (ptr_delay_timers[i] == 0) {
					int post_id = post_ids[tid][i];
					if (has_generator[tid] &&
					nrn_current[post_id] <= 300 &&
					nrn_current[post_id] >= -300) {
						nrn_current[post_id] += synapses_weight[tid][i];
					}
					ptr_delay_timers[i] = -1;
				}
				if (ptr_delay_timers[i] > 0) {
					ptr_delay_timers[i]--;
				}
			}

			has_spike[tid] = false;

			// update currents of the neuron
			if (I_syn != 0) {
				// decrease current potential
				if (I_syn > 0) nrn_current[tid] /= 2;   // for positive current
				if (I_syn < 0) nrn_current[tid] /= 1.1;   // for negative current
				// avoid the near value to 0
				if (I_syn > 0 && I_syn <= 1) nrn_current[tid] = 0;
				if (I_syn <= 0 && I_syn >= -1) nrn_current[tid] = 0;
			}

			// update the refractory period timer
			if (refractory_time_timer[tid] > 0)
				refractory_time_timer[tid]--;
		}

		__syncthreads();
		multimeter_result[sim_iter] = mm_result;
	}
}

float rand_dist(float data, float delta) {
	return float(rand()) / float(RAND_MAX) * 2 * delta + data - delta;
}

int get_random_neighbor(Group &group) {
	return group.id_start + rand() % group.group_size;
}

void connect_fixed_outdegree(Group pre_neurons, Group post_neurons,
		float syn_delay, float weight, int outdegree = syn_outdegree) {
	//weight *= 0.4;
	float time_delta = syn_delay * 0.4f;   //0.4
	float weight_delta = weight * 0.3f;    //0.3
	printf("Connect %s with %s (1:%d). W=%.2f (±%.2f), D=%.1f (±%.1f)\n",
		   pre_neurons.group_name.c_str(),
		   post_neurons.group_name.c_str(),
		   post_neurons.group_size,
		   weight, weight_delta,
		   syn_delay, time_delta);

	for (int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++) {
		for (int i = 0; i < outdegree; i++) {
			int rand_post_id = get_random_neighbor(post_neurons);
			float syn_delay_dist = rand_dist(syn_delay, time_delta);
			float syn_weight_dist = rand_dist(weight, weight_delta);

			metadatas.at(pre_id).push_back(Metadata(rand_post_id, syn_delay_dist, syn_weight_dist));
		}
	}
}

void group_add_multimeter(Group &nrn_group) {
	printf("Added multmeter to %s \n", nrn_group.group_name.c_str());

	for (int nrn_id = nrn_group.id_start; nrn_id <= nrn_group.id_end; nrn_id++) {
		has_multimeter[nrn_id] = true;
	}
}

void group_add_spike_generator(Group &nrn_group, float start, float end, int hz){
	printf("Added generator to %s \n", nrn_group.group_name.c_str());

	for (int nrn_id = nrn_group.id_start; nrn_id <= nrn_group.id_end; nrn_id++) {
		has_generator[nrn_id] = true;
		begin_spiking[nrn_id] = ms_to_step(start);
		end_spiking[nrn_id] = ms_to_step(end);
		spiking_per_step[nrn_id] = ms_to_step(1.0f / hz * 1000);
	}
}

void init_extensor() {
	group_add_multimeter(EES);
	group_add_spike_generator(C1, 0, speed_to_time, 200);
	group_add_spike_generator(C2, speed_to_time, 2*speed_to_time, 200);
	group_add_spike_generator(C3, 2*speed_to_time, 3*speed_to_time, 200);
	group_add_spike_generator(C4, 3*speed_to_time, 5*speed_to_time, 200);
	group_add_spike_generator(C5, 5*speed_to_time, 6*speed_to_time, 200);
	group_add_spike_generator(EES, 0, T_sim, EES_FREQ);

	connect_fixed_outdegree(C3, inh_group3, 0.5, 20.0);
	connect_fixed_outdegree(C4, inh_group4, 0.5, 20.0);
	connect_fixed_outdegree(C5, inh_group5, 0.5, 20.0);

	connect_fixed_outdegree(inh_group3, G1_3, 2.8, 20.0);

	connect_fixed_outdegree(inh_group4, G1_3, 1.0, 20.0);
	connect_fixed_outdegree(inh_group4, G2_3, 1.0, 20.0);

	connect_fixed_outdegree(inh_group5, G1_3, 2.0, 20.0);
	connect_fixed_outdegree(inh_group5, G2_3, 1.0, 20.0);
	connect_fixed_outdegree(inh_group5, G3_3, 1.0, 20.0);
	connect_fixed_outdegree(inh_group5, G4_3, 1.0, 20.0);

	/// D1
	// input from sensory
	connect_fixed_outdegree(C1, D1_1, 1.0, 1.0); // 3.0
	connect_fixed_outdegree(C1, D1_4, 1.0, 1.0); // 3.0
	connect_fixed_outdegree(C2, D1_1, 1.0, 1.0); // 3.0
	connect_fixed_outdegree(C2, D1_4, 1.0, 1.0); // 3.0
	// input from EES
	connect_fixed_outdegree(EES, D1_1, 1.0, 17); // was 27 // 17 Threshold / 7 ST
	connect_fixed_outdegree(EES, D1_4, 1.0, 17); // was 27 // 17 Threshold / 7 ST
	// inner connectomes
	connect_fixed_outdegree(D1_1, D1_2, 1, 7.0);
	connect_fixed_outdegree(D1_1, D1_3, 1, 16.0);
	connect_fixed_outdegree(D1_2, D1_1, 1, 7.0);
	connect_fixed_outdegree(D1_2, D1_3, 1, 20.0);
	connect_fixed_outdegree(D1_3, D1_1, 1, -10.0 * INH_COEF);    // 10
	connect_fixed_outdegree(D1_3, D1_2, 1, -10.0 * INH_COEF);    // 10
	connect_fixed_outdegree(D1_4, D1_3, 2, -10.0 * INH_COEF);    // 10
	// output to
	connect_fixed_outdegree(D1_3, G1_1, 3, 12.5);
	connect_fixed_outdegree(D1_3, ees_group1, 1.0, 60); // 30

	// EES group connectomes
	connect_fixed_outdegree(ees_group1, ees_group2, 1.0, 20.0);

	/// D2 ///
	// input from Sensory
	connect_fixed_outdegree(C2, D2_1, 0.7, 2.0); // 4
	connect_fixed_outdegree(C2, D2_4, 0.7, 2.0); // 4
	connect_fixed_outdegree(C3, D2_1, 0.7, 2.0); // 4
	connect_fixed_outdegree(C3, D2_4, 0.7, 2.0); // 4
	// input from Group (1)
	connect_fixed_outdegree(ees_group1, D2_1, 2.0, 1.7); // 5.0
	connect_fixed_outdegree(ees_group1, D2_4, 2.0, 1.7); // 5.0
	// inner connectomes
	connect_fixed_outdegree(D2_1, D2_2, 1.0, 7.0);
	connect_fixed_outdegree(D2_1, D2_3, 1.0, 20.0);
	connect_fixed_outdegree(D2_2, D2_1, 1.0, 7.0);
	connect_fixed_outdegree(D2_2, D2_3, 1.0, 20.0);
	connect_fixed_outdegree(D2_3, D2_1, 1.0, -10.0 * INH_COEF);    // 10
	connect_fixed_outdegree(D2_3, D2_2, 1.0, -10.0 * INH_COEF);    // 10
	connect_fixed_outdegree(D2_4, D2_3, 2.0, -10.0 * INH_COEF);    // 10
	// output to generator
	connect_fixed_outdegree(D2_3, G2_1, 1.0, 30.5);

	// EES group connectomes
	connect_fixed_outdegree(ees_group2, ees_group3, 1.0, 20.0);

	/// D3
	// input from Sensory
	connect_fixed_outdegree(C3, D3_1, 0.7, 1.9); // 2.5 - 2.2 - 1.3
	connect_fixed_outdegree(C3, D3_4, 0.7, 1.9); // 2.5 - 2.2 - 1.3
	connect_fixed_outdegree(C4, D3_1, 0.7, 1.9); // 2.5 - 2.2 - 1.3
	connect_fixed_outdegree(C4, D3_4, 0.7, 1.9); // 2.5 - 2.2 - 1.3
	// input from Group (2)
	connect_fixed_outdegree(ees_group2, D3_1, 1.1, 1.7); // 6
	connect_fixed_outdegree(ees_group2, D3_4, 1.1, 1.7); // 6
	// inner connectomes
	connect_fixed_outdegree(D3_1, D3_2, 1.0, 7.0);
	connect_fixed_outdegree(D3_1, D3_3, 1.0, 20.0);
	connect_fixed_outdegree(D3_2, D3_1, 1.0, 7.0);
	connect_fixed_outdegree(D3_2, D3_3, 1.0, 20.0);
	connect_fixed_outdegree(D3_3, D3_1, 1.0, -10 * INH_COEF);    // 10
	connect_fixed_outdegree(D3_3, D3_2, 1.0, -10 * INH_COEF);    // 10
	connect_fixed_outdegree(D3_4, D3_3, 2.0, -10 * INH_COEF);    // 10
	// output to generator
	connect_fixed_outdegree(D3_3, G3_1, 1, 25.0);
	// suppression of the generator
	connect_fixed_outdegree(D3_3, G1_3, 1.5, 30.0);

	// EES group connectomes
	connect_fixed_outdegree(ees_group3, ees_group4, 2.0, 20.0);

	/// D4
	// input from Sensory
	connect_fixed_outdegree(C4, D4_1, 1.0, 2.0); // 2.5
	connect_fixed_outdegree(C4, D4_4, 1.0, 2.0); // 2.5
	connect_fixed_outdegree(C5, D4_1, 1.0, 2.0); // 2.5
	connect_fixed_outdegree(C5, D4_4, 1.0, 2.0); // 2.5
	// input from Group (3)
	connect_fixed_outdegree(ees_group3, D4_1, 1.0, 1.7); // 6.0
	connect_fixed_outdegree(ees_group3, D4_4, 1.0, 1.7); // 6.0
	// inner connectomes
	connect_fixed_outdegree(D4_1, D4_2, 1.0, 7.0);
	connect_fixed_outdegree(D4_1, D4_3, 1.0, 20.0);
	connect_fixed_outdegree(D4_2, D4_1, 1.0, 7.0);
	connect_fixed_outdegree(D4_2, D4_3, 1.0, 20.0);
	connect_fixed_outdegree(D4_3, D4_1, 1.0, -10.0 * INH_COEF); // 10
	connect_fixed_outdegree(D4_3, D4_2, 1.0, -10.0 * INH_COEF); // 10
	connect_fixed_outdegree(D4_4, D4_3, 2.0, -10.0 * INH_COEF); // 10
	// output to the generator
	connect_fixed_outdegree(D4_3, G4_1, 3.0, 20.0);
	// suppression of the generator
	connect_fixed_outdegree(D4_3, G2_3, 1.0, 30.0);


	/// D5
	// input from Sensory
	connect_fixed_outdegree(C5, D5_1, 1.0, 2.0);
	connect_fixed_outdegree(C5, D5_4, 1.0, 2.0);
	// input from Group (4)
	connect_fixed_outdegree(ees_group4, D5_1, 1.0, 1.7); // 5.0
	connect_fixed_outdegree(ees_group4, D5_4, 1.0, 1.7); // 5.0
	// inner connectomes
	connect_fixed_outdegree(D5_1, D5_2, 1.0, 7.0);
	connect_fixed_outdegree(D5_1, D5_3, 1.0, 20.0);
	connect_fixed_outdegree(D5_2, D5_1, 1.0, 7.0);
	connect_fixed_outdegree(D5_2, D5_3, 1.0, 20.0);
	connect_fixed_outdegree(D5_3, D5_1, 1.0, -10.0 * INH_COEF);
	connect_fixed_outdegree(D5_3, D5_2, 1.0, -10.0 * INH_COEF);
	connect_fixed_outdegree(D5_4, D5_3, 2.0, -10.0 * INH_COEF);
	// output to the generator
	connect_fixed_outdegree(D5_3, G5_1, 3, 30.0);
	// suppression of the genearator
	connect_fixed_outdegree(D5_3, G1_3, 1.0, 30.0);
	connect_fixed_outdegree(D5_3, G2_3, 1.0, 30.0);
	connect_fixed_outdegree(D5_3, G3_3, 1.0, 30.0);
	connect_fixed_outdegree(D5_3, G4_3, 1.0, 30.0);

	/// G1 ///
	// inner connectomes
	connect_fixed_outdegree(G1_1, G1_2, 1.0, 10.0);
	connect_fixed_outdegree(G1_1, G1_3, 1.0, 10.0);
	connect_fixed_outdegree(G1_2, G1_1, 1.0, 10.0);
	connect_fixed_outdegree(G1_2, G1_3, 1.0, 10.0);
	connect_fixed_outdegree(G1_3, G1_1, 0.7, -50.0 * INH_COEF);
	connect_fixed_outdegree(G1_3, G1_2, 0.7, -50.0 * INH_COEF);
	// output to IP_E
	connect_fixed_outdegree(G1_1, IP_E, 3, 25.0); // 18 normal
	connect_fixed_outdegree(G1_1, IP_E, 3, 25.0); // 18 normal

	/// G2 ///
	// inner connectomes
	connect_fixed_outdegree(G2_1, G2_2, 1.0, 10.0);
	connect_fixed_outdegree(G2_1, G2_3, 1.0, 20.0);
	connect_fixed_outdegree(G2_2, G2_1, 1.0, 10.0);
	connect_fixed_outdegree(G2_2, G2_3, 1.0, 20.0);
	connect_fixed_outdegree(G2_3, G2_1, 0.5, -30.0 * INH_COEF);
	connect_fixed_outdegree(G2_3, G2_2, 0.5, -30.0 * INH_COEF);
	// output to IP_E
	connect_fixed_outdegree(G2_1, IP_E, 1.0, 65.0); // 35 normal
	connect_fixed_outdegree(G2_2, IP_E, 1.0, 65.0); // 35 normal

	/// G3 ///
	// inner connectomes
	connect_fixed_outdegree(G3_1, G3_2, 1.0, 14.0); //12
	connect_fixed_outdegree(G3_1, G3_3, 1.0, 20.0);
	connect_fixed_outdegree(G3_2, G3_1, 1.0, 12.0);
	connect_fixed_outdegree(G3_2, G3_3, 1.0, 20.0);
	connect_fixed_outdegree(G3_3, G3_1, 0.5, -30.0 * INH_COEF);
	connect_fixed_outdegree(G3_3, G3_2, 0.5, -30.0 * INH_COEF);
	// output to IP_E
	connect_fixed_outdegree(G3_1, IP_E, 2, 25.0);   // 20 normal
	connect_fixed_outdegree(G3_1, IP_E, 2, 25.0);   // 20 normal

	/// G4 ///
	// inner connectomes
	connect_fixed_outdegree(G4_1, G4_2, 1.0, 10.0);
	connect_fixed_outdegree(G4_1, G4_3, 1.0, 10.0);
	connect_fixed_outdegree(G4_2, G4_1, 1.0, 5.0);
	connect_fixed_outdegree(G4_2, G4_3, 1.0, 10.0);
	connect_fixed_outdegree(G4_3, G4_1, 0.5, -30.0 * INH_COEF);
	connect_fixed_outdegree(G4_3, G4_2, 0.5, -30.0 * INH_COEF);
	// output to IP_E
	connect_fixed_outdegree(G4_1, IP_E, 1.0, 17.0);
	connect_fixed_outdegree(G4_1, IP_E, 1.0, 17.0);

	/// G5 ///
	// inner connectomes
	connect_fixed_outdegree(G5_1, G5_2, 1.0, 7.0);
	connect_fixed_outdegree(G5_1, G5_3, 1.0, 10.0);
	connect_fixed_outdegree(G5_2, G5_1, 1.0, 7.0);
	connect_fixed_outdegree(G5_2, G5_3, 1.0, 10.0);
	connect_fixed_outdegree(G5_3, G5_1, 0.5, -30.0 * INH_COEF);
	connect_fixed_outdegree(G5_3, G5_2, 0.5, -30.0 * INH_COEF);
	// output to IP_E
	connect_fixed_outdegree(G5_1, IP_E, 2, 20.0); // normal 18
	connect_fixed_outdegree(G5_1, IP_E, 2, 20.0); // normal 18

	connect_fixed_outdegree(IP_E, MP_E, 1, 11); // 14
	connect_fixed_outdegree(EES, MP_E, 2, 50); // 50
	//connect_fixed_outdegree(Ia, MP_E, 1, 50)
}

void save_result(int test_index, float* multimeter_results, int nrn_size) {
	// Printing results function
	char cwd[256];
	getcwd(cwd, sizeof(cwd));
	printf("Save results to: %s \n", cwd);

	string new_name = "/test.dat";

	ofstream myfile;
	myfile.open(cwd + new_name);

	int time = 0;
	while (time < sim_step_time) {
		myfile << 0 << " " << time / 10.0f << " " << multimeter_results[time] / nrn_size << "\n";
		time += 1;
	}
	myfile.close();
}


template <typename type>
void memcpyHtD(type* gpu, type* host, int size) {
	//cudaMalloc(&gpu, sizeof(type) * size);
	cudaMemcpy(gpu, host, sizeof(type) * size, cudaMemcpyHostToDevice);
}

__host__
void simulate() {
	// get synapse number

	int neurons_number = static_cast<int>(metadatas.size());
	printf("num %d \n", neurons_number);

	const float V_rest = -72.0f;

	float* gpu_old_v;
	float* gpu_old_u;
	int* gpu_nrn_refractory_time;
	int* gpu_nrn_refractory_timer;
	bool* gpu_has_spike;
	bool* gpu_has_generator;
	bool* gpu_has_multimeter;
	float* gpu_nrn_current;
	int* gpu_synapses_number;
	int* gpu_begin_spiking;
	int* gpu_end_spiking;
	int* gpu_spiking_per_step;

	float* gpu_multimeter_result;

	float old_v[neurons_number] = {V_rest};
	float old_u[neurons_number] = {0.0f};
	int nrn_refractory_time[neurons_number] = {ms_to_step(3.0)};
	int nrn_refractory_timer[neurons_number] = {-1};
	bool has_spike[neurons_number] = {false};
	float nrn_current[neurons_number] = {0.0f};
	int synapses_number[neurons_number];

	float multimeter_result[sim_step_time] = {0.0f};

	has_multimeter = (bool *)malloc(neurons_number * sizeof(bool *));
	has_generator = (bool *)malloc(neurons_number * sizeof(bool *));
	begin_spiking = (int *)malloc(neurons_number * sizeof(int *));
	end_spiking = (int *)malloc(neurons_number * sizeof(int *));
	spiking_per_step = (int *)malloc(neurons_number * sizeof(int *));

	init_extensor();

	int **post_ids, **gpu_post_ids;
	int **synapses_delay, **gpu_synapses_delay;
	int **synapses_delay_timer, **gpu_synapses_delay_timer;
	float **synapses_weight, **gpu_synapses_weight;

	post_ids = (int **)malloc(neurons_number * sizeof(int **));
	cudaMalloc((void**)&gpu_post_ids, neurons_number * sizeof(int *));

	synapses_delay = (int **)malloc(neurons_number * sizeof(int *));
	cudaMalloc((void**)&gpu_synapses_delay, neurons_number * sizeof(int *));

	synapses_delay_timer = (int **)malloc(neurons_number * sizeof(int *));
	cudaMalloc((void**)&gpu_synapses_delay_timer, neurons_number * sizeof(int *));

	synapses_weight = (float **)malloc(neurons_number * sizeof(float *));
	cudaMalloc((void**)&gpu_synapses_weight, neurons_number * sizeof(float *));

	// synapse info
	for(unsigned long neuron_id = 0; neuron_id < neurons_number; neuron_id++) {
		int data_size = static_cast<int>(metadatas.at(neuron_id).size());
		synapses_number[neuron_id] = data_size;

		int post_ids_TMP[data_size];
		int synapses_delay_TMP[data_size];
		int synapses_delay_timer_TMP[data_size] = {-1};
		float synapses_weight_TMP[data_size];

		int data_iter = 0;
		for(Metadata m : metadatas.at(neuron_id)) {
			post_ids_TMP[data_iter] = m.post_id;
			synapses_delay_TMP[data_iter] = m.synapse_ref_t;
			synapses_weight_TMP[data_iter] = m.synapse_weight;
			data_iter++;
		}

		post_ids[neuron_id] = post_ids_TMP;
		synapses_delay[neuron_id] = synapses_delay_TMP;
		synapses_delay_timer[neuron_id] = synapses_delay_timer_TMP;
		synapses_weight[neuron_id] = synapses_weight_TMP;

		cudaMalloc((void**)&post_ids[neuron_id], data_size * sizeof(int));
		cudaMalloc((void**)&synapses_delay[neuron_id], data_size * sizeof(int));
		cudaMalloc((void**)&synapses_delay_timer[neuron_id], data_size * sizeof(int));
		cudaMalloc((void**)&synapses_weight[neuron_id], data_size * sizeof(float));
	}

	memcpyHtD<int *>(gpu_post_ids, post_ids, neurons_number);
	cudaMemcpy(gpu_synapses_delay, synapses_delay, neurons_number * sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_synapses_delay_timer, synapses_delay_timer, neurons_number * sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_synapses_weight, synapses_weight, neurons_number * sizeof(float *), cudaMemcpyHostToDevice);


	cudaMalloc(&gpu_old_v, sizeof(float) * neurons_number);
	memcpyHtD<float>(gpu_old_v, old_v, neurons_number);

	cudaMalloc(&gpu_old_u, sizeof(float) * neurons_number);
	memcpyHtD<float>(gpu_old_u, old_u, neurons_number);

	cudaMalloc(&gpu_has_spike, sizeof(bool) * neurons_number);
	cudaMemcpy(gpu_has_spike, has_spike, sizeof(bool) * neurons_number, cudaMemcpyHostToDevice);

	cudaMalloc(&gpu_nrn_refractory_time, sizeof(int) * neurons_number);
	cudaMemcpy(gpu_nrn_refractory_time, nrn_refractory_time, sizeof(int) * neurons_number, cudaMemcpyHostToDevice);

	cudaMalloc(&gpu_nrn_refractory_timer, sizeof(int) * neurons_number);
	cudaMemcpy(gpu_nrn_refractory_timer, nrn_refractory_timer, sizeof(int) * neurons_number, cudaMemcpyHostToDevice);

	cudaMalloc(&gpu_nrn_current, sizeof(float) * neurons_number);
	cudaMemcpy(gpu_nrn_current, nrn_current, sizeof(float) * neurons_number, cudaMemcpyHostToDevice);

	cudaMalloc(&gpu_has_generator, sizeof(bool) * neurons_number);
	cudaMemcpy(gpu_has_generator, has_generator, sizeof(bool) * neurons_number, cudaMemcpyHostToDevice);

	cudaMalloc(&gpu_begin_spiking, sizeof(int) * neurons_number);
	cudaMemcpy(gpu_begin_spiking, begin_spiking, sizeof(int) * neurons_number, cudaMemcpyHostToDevice);

	cudaMalloc(&gpu_end_spiking, sizeof(int) * neurons_number);
	cudaMemcpy(gpu_end_spiking, end_spiking, sizeof(int) * neurons_number, cudaMemcpyHostToDevice);

	cudaMalloc(&gpu_has_multimeter, sizeof(bool) * neurons_number);
	cudaMemcpy(gpu_has_multimeter, has_multimeter, sizeof(bool) * neurons_number, cudaMemcpyHostToDevice);

	cudaMalloc(&gpu_synapses_number, sizeof(int) * neurons_number);
	cudaMemcpy(gpu_synapses_number, synapses_number, sizeof(int) * neurons_number, cudaMemcpyHostToDevice);

	cudaMalloc(&gpu_spiking_per_step, sizeof(int) * neurons_number);
	cudaMemcpy(gpu_spiking_per_step, spiking_per_step, sizeof(int) * neurons_number, cudaMemcpyHostToDevice);

	cudaMalloc(&gpu_multimeter_result, sizeof(float) * sim_step_time);
	cudaMemcpy(gpu_multimeter_result, multimeter_result, sizeof(float) * sim_step_time, cudaMemcpyHostToDevice);

	int threads_per_block = 1024;
	int num_blocks = 1;//neurons_number / threads_per_block + 1;

	printf("Size of NRN %i \n", neurons_number);
	printf("Start GPU with %d threads x %d blocks (Total: %d th). With useless %d threads\n\n",
		   threads_per_block, num_blocks,
		   threads_per_block * num_blocks, threads_per_block * num_blocks - neurons_number);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	int shared_mem_size = sim_step_time * sizeof(float);
	// the main loop
		sim_GPU<<<num_blocks, threads_per_block, shared_mem_size>>>(
				gpu_old_v, gpu_old_u,
				gpu_nrn_current,
				sim_step,
				gpu_nrn_refractory_time,
				gpu_nrn_refractory_timer,
				gpu_synapses_number,
				gpu_has_spike,
				gpu_post_ids,
				gpu_synapses_delay,
				gpu_synapses_delay_timer,
				gpu_synapses_weight,
				neurons_number,
				gpu_has_generator,
				gpu_has_multimeter,
				gpu_multimeter_result,
				gpu_begin_spiking,
				gpu_end_spiking,
				gpu_spiking_per_step
		);

	cudaEventRecord(stop);

	cudaDeviceSynchronize();

	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	double t = milliseconds / 1e3;
	printf("Ellapsed time: %fs (%s) \n", t, t < 1? "YES!": "fuck...");

	// copy neurons/synapses array to the HOST
	cudaMemcpy(multimeter_result, gpu_multimeter_result, sizeof(float) * sim_step_time, cudaMemcpyDeviceToHost);

	// tell the CPU to halt further processing until the CUDA kernel has finished doing its business
	cudaDeviceSynchronize();

	// before cudaFree (!)
	save_result(0, multimeter_result, neurons_in_group);

	// remove data from HOST

	//practice good housekeeping by resetting the device when you are done
	cudaDeviceReset();
}


int main() {
	// set randon seed
	srand(time(NULL));
	simulate();

	return 0;
}