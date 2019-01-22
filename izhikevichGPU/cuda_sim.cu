#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <ctime>

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unistd.h>

#include "Synapse.cpp"
#include "Group.cpp"

using namespace std;

const unsigned int neurons_in_group = 5;
const unsigned int neurons_in_moto = 5;
const float INH_COEF = 1.0;
const int EES_FREQ = 40;
const float speed_to_time = 25;

// 6 cms = 125
// 15 cms = 50
// 21 cms = 25

Neuron *gpu_neurons;
Synapse *gpu_synapses;

vector<Neuron> host_neurons_vector;
vector<Synapse> host_synapses_vector;
vector<Group> with_multimeter;

int global_id = 0;
// simulation properties
float T_sim = speed_to_time * 6;
float step = 0.1;
int sim_step_time = (int)(T_sim / step);

void save_result(int test_index, Neuron* host_neurons);


Group form_group(string group_name, int nrns_in_group = neurons_in_group) {
	Group group = Group();

	group.group_name = group_name;
	group.id_start = global_id;
	group.id_end = global_id + nrns_in_group - 1;
	group.group_size = nrns_in_group;

	for (int local_id = 0; local_id < nrns_in_group; ++local_id) {
		// append the anonymous Neuron object
		host_neurons_vector.push_back(Neuron(global_id, group_name, 3.0));
		global_id++;
	}

	printf("Formed %s IDs [%d ... %d] = %d\n",
		   group_name.c_str(), global_id, global_id + nrns_in_group - 1, nrns_in_group);

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

Group IP_E = form_group("IP_E", neurons_in_moto);
Group MP_E = form_group("MP_E", neurons_in_moto);
Group EES = form_group("EES");

Group inh_group3 = form_group("inh_group3");
Group inh_group4 = form_group("inh_group4");
Group inh_group5 = form_group("inh_group5");

Group ees_group1 = form_group("ees_group1");
Group ees_group2 = form_group("ees_group2");
Group ees_group3 = form_group("ees_group3");
Group ees_group4 = form_group("ees_group4");


__global__
void sim_GPU(Neuron *neurons, Synapse *synapses, int nrn_size, int syn_size, int sim_iter) {
	int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

	// update neurons state
	if (thread_id < syn_size) {
		if (thread_id < nrn_size) {
			neurons[thread_id].update(sim_iter, thread_id);
		}
		// wait until all threads in block have finished to this point and calculated neurons state
		__syncthreads();

		// update synapses state
		if (thread_id < syn_size) {
			synapses[thread_id].update(sim_iter, thread_id);
		}
		// wait until all threads in block have finished to this point and calculated synapses state
		__syncthreads();

		// unset spike flag (this realization frees up neuron <-> synapses relation for easiest GPU implementation)
		if (thread_id < nrn_size) {
			neurons[thread_id].set_has_spike(false);
		}
		__syncthreads();
	}

}

float rand_dist(float data, float delta) {
	return float(rand()) / float(RAND_MAX) * 2 * delta + data - delta;
}

int get_random_neighbor(int pre_id, int post_neurons_number) {
	int post_neuron_id = rand() % post_neurons_number;
	while (post_neuron_id == pre_id)
		post_neuron_id = rand() % post_neurons_number;
	return post_neuron_id;
}

void connect_fixed_outdegree(Group pre_neurons, Group post_neurons, float syn_delay, float weight) {
	//weight *= 0.4;
	float time_delta = syn_delay * 0.4f; //0.4
	float weight_delta = weight * 0.3f; //0.3
	printf("Connect %s with %s (1:%d). W=%.2f (±%.2f), D=%.1f (±%.1f)\n",
		   pre_neurons.group_name.c_str(),
		   post_neurons.group_name.c_str(),
		   post_neurons.group_size,
		   weight, weight_delta,
		   syn_delay, time_delta);


	for (int pre_id = 0; pre_id < pre_neurons.group_size; ++pre_id) {
		for (int post_id = 0; post_id < post_neurons.group_size; ++post_id) {
			int post_neuron_index = get_random_neighbor(pre_id, post_neurons.group_size);
			float syn_delay_dist = rand_dist(syn_delay, time_delta);
			float syn_weight_dist = rand_dist(weight, weight_delta);

			// append the anonymous Synapse object
			host_synapses_vector.push_back(
					Synapse(&gpu_neurons[pre_id],  // (!) pointers should point to the memory in GPU!
							&gpu_neurons[post_neuron_index],
							syn_delay_dist,
							syn_weight_dist));
		}
	}
}



void group_add_multimeter(Group &nrn_group) {
	printf("Added multmeter to %s \n", nrn_group.group_name.c_str());

	with_multimeter.push_back(nrn_group);

	for (int nrn_id = nrn_group.id_start; nrn_id <= nrn_group.id_end; ++nrn_id) {
		float *mm_data;
		float *curr_data;

		cudaMalloc(&mm_data, sizeof(float) * sim_step_time);
		cudaMalloc(&curr_data, sizeof(float) * sim_step_time);

		host_neurons_vector.at(nrn_id).add_multimeter(mm_data, curr_data);
	}
}

void group_add_spike_generator(Group &nrn_group, float start, float end, int hz){
	printf("Added generator to %s \n", nrn_group.group_name.c_str());

	for (int nrn_id = nrn_group.id_start; nrn_id <= nrn_group.id_end; ++nrn_id) {
		host_neurons_vector.at(nrn_id).add_spike_generator(start, end, hz);
	}

}

void init_extensor() {
	// - - - - - - - - - - - -
	// CPG (Extensor)
	// - - - - - - - - - - - -
	//group_add_multimeter(C1);
	group_add_multimeter(D1_1);
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


void simulate() {
	// get synapse number
	int neuron_number = (int)host_neurons_vector.size();

	// convert filled vector to array of pointers
	Neuron* host_neurons = host_neurons_vector.data();

	// allocate memory in GPU (only after this you can init connections)
	cudaMalloc(&gpu_neurons, sizeof(Neuron) * neuron_number);

	// only after cudaMalloc (!)
	init_extensor();

	// get synapse number (after finishing of conectivity building)
	int synapse_number = (int)host_synapses_vector.size();

	printf("Neuron number : %d \n", neuron_number);
	printf("Synapse number : %d \n", synapse_number);

	// convert filled vector to the array of pointers
	Synapse* host_synapses = host_synapses_vector.data();
	// allocate memory in GPU
	cudaMalloc(&gpu_synapses, sizeof(Synapse) * synapse_number);

	// copy neurons/synapses array to the GPU
	cudaMemcpy(gpu_neurons, host_neurons, sizeof(Neuron) * neuron_number, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_synapses, host_synapses, sizeof(Synapse) * synapse_number, cudaMemcpyHostToDevice);

	int thread_in_block = 32;
	int block_size = (synapse_number / thread_in_block) + 1;

	printf("Size of NRN %zu bytes (total: %zu MB) \n", sizeof(Neuron), sizeof(Neuron) * neuron_number /  (2 << 10));
	printf("Size of SYN %zu bytes (total: %zu MB) \n", sizeof(Synapse), sizeof(Synapse) * synapse_number / (2 << 10));
	printf("Start GPU with %d threads x %d blocks (Total: %d th). With useless %d threads\n\n",
		   thread_in_block, block_size,
		   thread_in_block * block_size, thread_in_block * block_size - synapse_number);

	// the main loop
	for(int iter = 0; iter < sim_step_time; iter++) {
		sim_GPU<<<block_size, thread_in_block>>>(gpu_neurons, gpu_synapses, neuron_number, synapse_number, iter);
	}

	// copy neurons/synapses array to the HOST
	cudaMemcpy(host_neurons, gpu_neurons, sizeof(Neuron) * neuron_number, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_synapses, gpu_synapses, sizeof(Synapse) * synapse_number, cudaMemcpyDeviceToHost);

	// tell the CPU to halt further processing until the CUDA kernel has finished doing its business
	cudaDeviceSynchronize();

	// FixMe sort functions to classes
	// before cudaFree (!)
	save_result(0, host_neurons);

	// remove data from HOST

	// remove data from GPU
	cudaFree(gpu_neurons);
	cudaFree(gpu_synapses);

	//practice good housekeeping by resetting the device when you are done
	cudaDeviceReset();

}

void save_result(int test_index, Neuron* host_neurons) {
	// Printing results function
	for (auto &group: with_multimeter) {
		char cwd[256];
		getcwd(cwd, sizeof(cwd));
		printf("Save results to: %s", cwd);

		string new_name = "/" + to_string(test_index) + "_" + group.group_name + ".dat";

		ofstream myfile;
		myfile.open(cwd + new_name);

		for (int id = group.id_start; id <= group.id_end; id++) {
			float mm_data[sim_step_time];
			float curr_data[sim_step_time];

			// copy data from GPU to HOST
			cudaMemcpy(mm_data, host_neurons[id].get_mm_data(), sizeof(float) * sim_step_time, cudaMemcpyDeviceToHost);
			cudaMemcpy(curr_data, host_neurons[id].get_curr_data(), sizeof(float) * sim_step_time, cudaMemcpyDeviceToHost);

			int time = 0;
			while (time < sim_step_time) {
				myfile << id << " " << time / 10.0f << " " << mm_data[time] << " " << curr_data[time] << "\n";
				time += 1;
			}
		}
		myfile.close();
	}
}



int main() {
	// set randon seed
	srand(time(NULL));

	simulate();

	return 0;
}