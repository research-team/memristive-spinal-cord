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

const unsigned int syn_outdegree = 27;
const unsigned int neurons_in_ip = 196;
const unsigned int neurons_in_moto = 169;
const unsigned int neurons_in_group = 20;
const unsigned int neurons_in_afferent = 196;

const int speed = 25;
const int EES_FREQ = 40;
const float INH_COEF = 1.0;

// 6 cms = 125
// 15 cms = 50
// 21 cms = 25

// stuff variable
unsigned int global_id = 0;

const float T_sim = 150;
const float sim_step = 0.25;
const unsigned int sim_time_in_step = (unsigned int)(T_sim / sim_step);

__host__
int ms_to_step(float ms) { return (int)(ms / sim_step); }

struct Metadata{
	/*
	 *
	 */
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

Group form_group(string group_name, int nrns_in_group = neurons_in_group) {
	/*
	 *
	 */
	Group group = Group();

	group.group_name = group_name;
	group.id_start = global_id;
	group.id_end = global_id + nrns_in_group - 1;
	group.group_size = nrns_in_group;

	global_id += nrns_in_group;

	printf("Formed %s IDs [%d ... %d] = %d\n",
		   group_name.c_str(), global_id - nrns_in_group, global_id - 1, nrns_in_group);

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

__device__
float calc_afferent(float moto){
	return 0;
}

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

__global__
void sim_kernel(float* old_v,
                float* old_u,
                float* nrn_current,
                int* refractory_time,
                int* refractory_time_timer,
                int* synapses_number,
                bool* has_spike,
                int** synapses_post_nrn_id,
                int** synapses_delay,
                int** synapses_delay_timer,
                float** synapses_weight,
                unsigned int nrn_size,
                bool* has_generator,
                bool* has_multimeter,
                float* multimeter_result,
                float* current_result,
                int* begin_spiking,
                int* end_spiking,
                int* spiking_per_step,
                // ToDo remove after debugging
                float* global_multimeter,
                float* global_currents) {

	__shared__ float moto_Vm_per_step;

	// the main simulation loop
	for (int sim_iter = 0; sim_iter < sim_time_in_step; sim_iter++) {
		// get id of the thread
		int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

		// init shared values
		if(thread_id == 0) {
			moto_Vm_per_step = 0;
		}

		// wait all threads
		__syncthreads();

		// thread stride loop (0, 1024, 1, 1025, 2, 1026 ...)
		for (int tid = thread_id; tid < nrn_size; tid += blockDim.x * gridDim.x) {
			// spike-generator invoking
			if (has_generator[tid] &&
				sim_iter >= begin_spiking[tid] &&
				sim_iter < end_spiking[tid] &&
				(sim_iter % spiking_per_step[tid] == 0)) {
				nrn_current[tid] = 15000;
			}

			// todo check with the real neurobiology mechanism
			// absolute refractory period : calculate V_m and U_m WITHOUT synaptic weight (nrn_current)
			// action potential : calculate V_m and U_m WITH synaptic weight (nrn_current)
			nrn_current[tid] = (refractory_time_timer[tid] > 0) ? 0 : nrn_current[tid];

			float V_old = old_v[tid];
			float U_old = old_u[tid];
			float I_current = nrn_current[tid];

//			if(tid == 100)
//				current_result[sim_iter] = I_current;

			float V_m = V_old + sim_step * (k * (V_old - V_rest) * (V_old - V_thld) - U_old + I_current) / C;
			float U_m = U_old + sim_step * a * (b * (V_old - V_rest) - U_old);

			// set bottom border of membrane potential
			if (V_m < c)
				V_m = c;

//			if (tid >= 996 && tid <= 1164) {
//				atomicAdd(&moto_Vm_per_step, V_m);
//			}

			// save the V_m value every iter step if has multimeter
//			if (has_multimeter[tid]) {
//				if(tid == 100)
//					atomicAdd(&multimeter_result[sim_iter], (V_m >= V_thld)? V_peak : V_m);
////					multimeter_result[sim_iter] = (V_m >= V_thld)? V_peak : V_m;
//			}

			// ToDo remove after debugging
			int index = sim_iter + tid * sim_time_in_step;
			global_multimeter[index] = (V_m >= V_thld)? V_peak : V_m;
			global_currents[index] = I_current;

			// threshold crossing (spike)
			if (V_m >= V_thld) {
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

			// pointers to current neuronID synapses_delay_timer (decrease array calls)
			int *ptr_delay_timers = synapses_delay_timer[tid];

			for (int syn_id = 0; syn_id < synapses_number[tid]; syn_id++) {
				if (has_spike[tid] && ptr_delay_timers[syn_id] == -1) {
					ptr_delay_timers[syn_id] = synapses_delay[tid][syn_id];
				}
				if (ptr_delay_timers[syn_id] == 0) {
					int post_nrn_id = synapses_post_nrn_id[tid][syn_id];
					if (nrn_current[post_nrn_id] <= 120000 && nrn_current[post_nrn_id] >= -120000) {
						nrn_current[post_nrn_id] += synapses_weight[tid][syn_id];
					}
					ptr_delay_timers[syn_id] = -1;
				}
				if (ptr_delay_timers[syn_id] > 0) {
					ptr_delay_timers[syn_id]--;
				}
			} // end synapse updating loop

			has_spike[tid] = false;

			// update currents of the neuron
			if (I_current != 0) {
				// decrease current potential
				if (I_current > 0) nrn_current[tid] /= 2;   // for positive current
				if (I_current < 0) nrn_current[tid] /= 1.1;   // for negative current
				// avoid the near value to 0
				if (I_current > 0 && I_current <= 1) nrn_current[tid] = 0;
				if (I_current <= 0 && I_current >= -1) nrn_current[tid] = 0;
			}

			// update the refractory period timer
			if (refractory_time_timer[tid] > 0)
				refractory_time_timer[tid]--;
		} // end of stride loop

		if (thread_id == 0) {
			// mean time between spikes (ms)
			float Ia_interval = (1000 / (6.2 * pow(speed, 0.6) + 0.17 + 0.06 * (moto_Vm_per_step / neurons_in_moto)));
			// mean number of spikes
			// 1000 / ((6.2*pow(speed, 0.6) + 0.17 + 0.06*(vMN / neurons_in_moto + 65))*8)
			// calc_afferent(moto_Vm_per_step / 169);
		}

		// wait all threads
		__syncthreads();
	} // end of sim iteration
}

float rand_dist(float data, float delta) {
	return float(rand()) / float(RAND_MAX) * 2 * delta + data - delta;
}

int get_random_neighbor(Group &group) {
	return group.id_start + rand() % group.group_size;
}

void connect_fixed_outdegree(Group pre_neurons, Group post_neurons,
							 float syn_delay, float weight, int outdegree = syn_outdegree) {
	weight *= 0.8;
	float time_delta = syn_delay * 0.2f;
	float weight_delta = weight * 0.1f;

	for (int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++) {
		for (int i = 0; i < outdegree; i++) {
			int rand_post_id = get_random_neighbor(post_neurons);
			float syn_delay_dist = rand_dist(syn_delay, time_delta);
			float syn_weight_dist = rand_dist(weight, weight_delta);

			metadatas.at(pre_id).push_back(Metadata(rand_post_id, syn_delay_dist, syn_weight_dist));
		}
	}

	printf("Connect %s with %s (1:%d). W=%.2f (±%.2f), D=%.1f (±%.1f)\n",
		   pre_neurons.group_name.c_str(),
		   post_neurons.group_name.c_str(),
		   post_neurons.group_size,
		   weight, weight_delta,
		   syn_delay, time_delta);
}

void group_add_multimeter(Group &nrn_group) {
	/*
	 *
	 */
	for (int nrn_id = nrn_group.id_start; nrn_id <= nrn_group.id_end; nrn_id++) {
		has_multimeter[nrn_id] = true;
	}

	printf("Added multmeter to %s \n", nrn_group.group_name.c_str());
}

void group_add_spike_generator(Group &nrn_group, float start, float end, int hz){
	/*
	 *
	 */
	for (int nrn_id = nrn_group.id_start; nrn_id <= nrn_group.id_end; nrn_id++) {
		has_generator[nrn_id] = true;
		begin_spiking[nrn_id] = ms_to_step(start + 0.1);
		end_spiking[nrn_id] = ms_to_step(end);
		spiking_per_step[nrn_id] = ms_to_step(1000.0f / hz);
	}

	printf("Added generator to %s \n", nrn_group.group_name.c_str());
}

void init_extensor() {
	/*
	 *
	 */
	group_add_multimeter(D1_1);
	group_add_spike_generator(C1, 0, speed, 200);
	group_add_spike_generator(C2, speed, 2*speed, 200);
	group_add_spike_generator(C3, 2*speed, 3*speed, 200);
	group_add_spike_generator(C4, 3*speed, 5*speed, 200);
	group_add_spike_generator(C5, 5*speed, 6*speed, 200);
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
	connect_fixed_outdegree(Ia, MP_E, 1, 1);
}

void save_result(int test_index, float* global_multimeter, float* global_currents, int neurons_number) {
	/* Printing results function
	 *
	 */
	char cwd[256];
	getcwd(cwd, sizeof(cwd));
	printf("Save results to: %s \n", cwd);

	string new_name = "/volt.dat";

	ofstream myfile;
	myfile.open(cwd + new_name);

	for(int nrn_id = 0; nrn_id < neurons_number; nrn_id++){
		myfile << nrn_id << " ";
		for(int sim_iter = 0; sim_iter < sim_time_in_step; sim_iter++)
			myfile << global_multimeter[sim_iter + nrn_id * sim_time_in_step] << " ";
		myfile << "\n";
	}

	myfile.close();

	new_name = "/curr.dat";

	myfile.open(cwd + new_name);

	for(int nrn_id = 0; nrn_id < neurons_number; nrn_id++){
		myfile << nrn_id << " ";
		for(int sim_iter = 0; sim_iter < sim_time_in_step; sim_iter++)
			myfile << global_currents[sim_iter + nrn_id * sim_time_in_step] << " ";
		myfile << "\n";
	}

	myfile.close();
}

template <typename type>
void memcpyHtD(type* gpu, type* host, int size) {
	cudaMemcpy(gpu, host, sizeof(type) * size, cudaMemcpyHostToDevice);
}

template <typename type>
void memcpyDtH(type* host, type* gpu, int size) {
	cudaMemcpy(host, gpu, sizeof(type) * size, cudaMemcpyDeviceToHost);
}

template <typename type>
unsigned int datasize(int size) {
	return sizeof(type) * size;
}

template <typename type>
void init_array(type *array, int size, type value){
	for(int i = 0; i < size; i++)
		array[i] = value;
}

__host__
void simulate() {
	/*
	 *
	 */
	int neurons_number = static_cast<int>(metadatas.size());

	float* gpu_old_v;
	float* gpu_old_u;
	int* gpu_nrn_ref_time;
	int* gpu_nrn_ref_timer;
	bool* gpu_has_spike;
	bool* gpu_has_generator;
	bool* gpu_has_multimeter;
	float* gpu_nrn_current;
	int* gpu_synapses_number;
	int* gpu_begin_spiking;
	int* gpu_end_spiking;
	int* gpu_spiking_per_step;
	float* gpu_multimeter_result;
	float* gpu_current_result;

	// ToDo remove after debugging
	float* gpu_global_multimeter;
	float* gpu_global_currents;

	int synapses_number[neurons_number];

	float old_v[neurons_number];
	init_array<float>(old_v, neurons_number, -72);

	float old_u[neurons_number];
	init_array<float>(old_u, neurons_number, 0);

	int nrn_ref_time[neurons_number];
	init_array<int>(nrn_ref_time, neurons_number, ms_to_step(3.0));

	int nrn_ref_timer[neurons_number];
	init_array<int>(nrn_ref_timer, neurons_number, -1);

	bool has_spike[neurons_number];
	init_array<bool>(has_spike, neurons_number, false);

	float nrn_current[neurons_number];
	init_array<float>(nrn_current, neurons_number, 0);

	float multimeter_result[sim_time_in_step];
	init_array<float>(multimeter_result, sim_time_in_step, 0);

	float current_result[sim_time_in_step];
	init_array<float>(current_result, sim_time_in_step, 0);

	// ToDo remove after debugging
	float global_multimeter[neurons_number * sim_time_in_step];
	init_array<float>(global_multimeter, neurons_number * sim_time_in_step, 0);
	// ToDo remove after debugging
	float global_currents[neurons_number * sim_time_in_step];
	init_array<float>(global_currents, neurons_number * sim_time_in_step, 0);

	has_multimeter = (bool *)malloc(datasize<bool *>(neurons_number));
	has_generator = (bool *)malloc(datasize<bool *>(neurons_number));
	begin_spiking = (int *)malloc(datasize<int *>(neurons_number));
	end_spiking = (int *)malloc(datasize<int *>(neurons_number));
	spiking_per_step = (int *)malloc(datasize<int *>(neurons_number));

	init_extensor();

	int **gpu_synapses_post_nrn_id, **synapses_post_nrn_id = (int **)malloc(datasize<int* >(neurons_number));
	int **gpu_synapses_delay, **synapses_delay = (int **)malloc(datasize<int* >(neurons_number));
	int **gpu_synapses_delay_timer, **synapses_delay_timer = (int **)malloc(datasize<int* >(neurons_number));
	float **gpu_synapses_weight, **synapses_weight = (float **)malloc(datasize<float* >(neurons_number));


	// fill arrays of synapses
	for(int neuron_id = 0; neuron_id < neurons_number; neuron_id++) {
		int syn_count = static_cast<int>(metadatas.at(neuron_id).size());

		int tmp_synapses_post_nrn_id[syn_count];
		int tmp_synapses_delay[syn_count];
		int tmp_synapses_delay_timer[syn_count];
		float tmp_synapses_weight[syn_count];

		int syn_id = 0;
		for(Metadata m : metadatas.at(neuron_id)) {
			tmp_synapses_post_nrn_id[syn_id] = m.post_id;
			tmp_synapses_delay[syn_id] = m.synapse_ref_t;
			tmp_synapses_delay_timer[syn_id] = -1;
			tmp_synapses_weight[syn_id] = m.synapse_weight * 200;
			syn_id++;
		}

		synapses_number[neuron_id] = syn_count;

		cudaMalloc((void**)&synapses_post_nrn_id[neuron_id], datasize<int>(syn_count));
		cudaMalloc((void**)&synapses_delay[neuron_id], datasize<int>(syn_count));
		cudaMalloc((void**)&synapses_delay_timer[neuron_id], datasize<int>(syn_count));
		cudaMalloc((void**)&synapses_weight[neuron_id], datasize<float>(syn_count));

		cudaMemcpy(synapses_post_nrn_id[neuron_id], &tmp_synapses_post_nrn_id, datasize<int>(syn_count), cudaMemcpyHostToDevice);
		cudaMemcpy(synapses_delay[neuron_id], &tmp_synapses_delay, datasize<int>(syn_count), cudaMemcpyHostToDevice);
		cudaMemcpy(synapses_delay_timer[neuron_id], &tmp_synapses_delay_timer, datasize<int>(syn_count), cudaMemcpyHostToDevice);
		cudaMemcpy(synapses_weight[neuron_id], &tmp_synapses_weight, datasize<float>(syn_count), cudaMemcpyHostToDevice);
	}

	cudaMalloc((void ***)&gpu_synapses_post_nrn_id, datasize<int *>(neurons_number));
	memcpyHtD<int *>(gpu_synapses_post_nrn_id, synapses_post_nrn_id, neurons_number);

	cudaMalloc((void ***)&gpu_synapses_delay, datasize<int *>(neurons_number));
	memcpyHtD<int *>(gpu_synapses_delay, synapses_delay, neurons_number);

	cudaMalloc((void ***)&gpu_synapses_delay_timer, datasize<int *>(neurons_number));
	memcpyHtD<int *>(gpu_synapses_delay_timer, synapses_delay_timer, neurons_number);

	cudaMalloc((void ***)&gpu_synapses_weight, datasize<float *>(neurons_number));
	memcpyHtD<float *>(gpu_synapses_weight, synapses_weight, neurons_number);

	cudaMalloc(&gpu_old_v, datasize<float>(neurons_number));
	memcpyHtD<float>(gpu_old_v, old_v, neurons_number);

	cudaMalloc(&gpu_old_u, datasize<float>(neurons_number));
	memcpyHtD<float>(gpu_old_u, old_u, neurons_number);

	cudaMalloc(&gpu_has_spike, datasize<bool>(neurons_number));
	memcpyHtD<bool>(gpu_has_spike, has_spike, neurons_number);

	cudaMalloc(&gpu_nrn_ref_time, datasize<int>(neurons_number));
	memcpyHtD<int>(gpu_nrn_ref_time, nrn_ref_time, neurons_number);

	cudaMalloc(&gpu_nrn_ref_timer, datasize<int>(neurons_number));
	memcpyHtD<int>(gpu_nrn_ref_timer, nrn_ref_timer, neurons_number);

	cudaMalloc(&gpu_nrn_current, datasize<float>(neurons_number));
	memcpyHtD<float>(gpu_nrn_current, nrn_current, neurons_number);

	cudaMalloc(&gpu_has_generator, datasize<bool>(neurons_number));
	memcpyHtD<bool>(gpu_has_generator, has_generator, neurons_number);

	cudaMalloc(&gpu_begin_spiking, datasize<int>(neurons_number));
	memcpyHtD<int>(gpu_begin_spiking, begin_spiking, neurons_number);

	cudaMalloc(&gpu_end_spiking, datasize<int>(neurons_number));
	memcpyHtD<int>(gpu_end_spiking, end_spiking, neurons_number);

	cudaMalloc(&gpu_has_multimeter, datasize<bool>(neurons_number));
	memcpyHtD<bool>(gpu_has_multimeter, has_multimeter, neurons_number);

	cudaMalloc(&gpu_synapses_number, datasize<int>(neurons_number));
	memcpyHtD<int>(gpu_synapses_number, synapses_number, neurons_number);

	cudaMalloc(&gpu_spiking_per_step, datasize<int>(neurons_number));
	memcpyHtD<int>(gpu_spiking_per_step, spiking_per_step, neurons_number);

	cudaMalloc(&gpu_multimeter_result, datasize<float>(sim_time_in_step));
	memcpyHtD<float>(gpu_multimeter_result, multimeter_result, sim_time_in_step);

	cudaMalloc(&gpu_current_result, datasize<float>(sim_time_in_step));
	memcpyHtD<float>(gpu_current_result, current_result, sim_time_in_step);

	// FixMe debugging functionality
	cudaMalloc(&gpu_global_multimeter, datasize<float>(neurons_number * sim_time_in_step));
	memcpyHtD<float>(gpu_global_multimeter, global_multimeter, neurons_number * sim_time_in_step);
	// FixMe debugging functionality
	cudaMalloc(&gpu_global_currents, datasize<float>(neurons_number * sim_time_in_step));
	memcpyHtD<float>(gpu_global_currents, global_currents, neurons_number * sim_time_in_step);

	int threads_per_block = 1024;
	int num_blocks = 1; //neurons_number / threads_per_block + 1;

	printf("Size of network: %i \n", neurons_number);
	printf("Start GPU with %d threads x %d blocks (Total: %d th) \n",
		   threads_per_block, num_blocks, threads_per_block * num_blocks);

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	sim_kernel<<<num_blocks, threads_per_block>>>(
	    gpu_old_v,
	    gpu_old_u,
	    gpu_nrn_current,
	    gpu_nrn_ref_time,
	    gpu_nrn_ref_timer,
	    gpu_synapses_number,
	    gpu_has_spike,
	    gpu_synapses_post_nrn_id,
	    gpu_synapses_delay,
	    gpu_synapses_delay_timer,
	    gpu_synapses_weight,
	    neurons_number,
	    gpu_has_generator,
	    gpu_has_multimeter,
	    gpu_multimeter_result,
	    gpu_current_result,
	    gpu_begin_spiking,
	    gpu_end_spiking,
	    gpu_spiking_per_step,
	    // ToDo remove after debugging
	    gpu_global_multimeter,
	    gpu_global_currents
	);

	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	double t = milliseconds / 1e3;
	double realtime_factor = T_sim / t / 1e3;
	printf("Ellapsed time: %fs. Realtime factor: x%f (%s than realtime)\n",
		   t, realtime_factor, realtime_factor > 1? "faster":"slower");

	// copy neurons/synapses array to the HOST
	memcpyDtH<float>(multimeter_result, gpu_multimeter_result, sim_time_in_step);
	memcpyDtH<float>(current_result, gpu_current_result, sim_time_in_step);
	// ToDo remove after debugging
	memcpyDtH<float>(global_multimeter, gpu_global_multimeter, neurons_number * sim_time_in_step);
	memcpyDtH<float>(global_currents, gpu_global_currents, neurons_number * sim_time_in_step);

	// tell the CPU to halt further processing until the CUDA kernel has finished doing its business
	cudaDeviceSynchronize();

	// before cudaFree (!)
	save_result(0, global_multimeter, global_currents, neurons_number);

	//practice good housekeeping by resetting the device when you are done
	cudaDeviceReset();
}


int main() {
	srand(time(NULL));	// set randon seed
	simulate();

	return 0;
}