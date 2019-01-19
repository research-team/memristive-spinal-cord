#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <ctime>
#include "Synapse.cpp"
#include "Group.cpp"

using namespace std;

const unsigned int neurons_in_group = 5;

// gropus
Group C1;
Group C2;

Neuron *gpu_neurons;
Synapse *gpu_synapses;

vector<Neuron> host_neurons_vector;
vector<Synapse> host_synapses_vector;

int global_id = 0;


__global__
void sim_GPU(Neuron *neurons, Synapse *synapses, int nrn_size, int syn_size, int block_width, int sim_time_step) {
	// get thread ID
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * block_width;

	// main simulation loop
	for(int sim_iter = 0; sim_iter < sim_time_step; ++sim_iter) {
		// update neurons state
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

		#ifdef DEBUG
			if(thread_id == 0) printf("- - - - -\n");
		#endif
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
//	weight *= 0.4; // 0.8
//
//	float time_delta = syn_delay * 0.4f; //0.2
//	float weight_delta = weight * 0.3f;
	float syn_delay_dist = rand_dist(syn_delay, 1);
	float syn_weight_dist = rand_dist(weight, 1);

	for (int pre_id = 0; pre_id < pre_neurons.group_size; ++pre_id) {

		for (int post_id = 0; post_id < post_neurons.group_size; ++post_id) {
			int post_neuron_index = get_random_neighbor(pre_id, post_neurons.group_size);
			// append the anonymous Synapse object
			host_synapses_vector.push_back(
			        Synapse(&gpu_neurons[pre_id],  // (!) pointers should point to the memory in GPU!
			                &gpu_neurons[post_neuron_index],
			                syn_delay_dist,
			                syn_weight_dist));
		}
	}
}

void form_group(Group &nrn_group, string group_name, int nrns_in_group = neurons_in_group) {
	printf("Formed %s IDs [%d ... %d] = %d\n",
			group_name.c_str(), global_id, global_id + nrns_in_group - 1, nrns_in_group);

	nrn_group.group_name = group_name;
	nrn_group.id_start = global_id;
	nrn_group.id_end = global_id + nrns_in_group - 1;
	nrn_group.group_size = nrns_in_group;

	for (int local_id = 0; local_id < nrns_in_group; ++local_id) {
		// append the anonymous Neuron object
		host_neurons_vector.push_back(
			Neuron(global_id, group_name, 3.0)
		);
		global_id++;
	}
}

void init_groups() {
	form_group(C1, "C1");
	form_group(C2, "C2");
}

void init_extensor() {
	connect_fixed_outdegree(C1, C2, 1.0, 10.0);
	connect_fixed_outdegree(C2, C1, 5.0, 5.0);
}

void simulate() {
	// simulation properties
	int sim_time = 3;
	float step = 0.1;
	int sim_step_time = (int)(sim_time / step);

	// get synapse number
	int neuron_number = (int)host_neurons_vector.size();

	Neuron* host_neurons = host_neurons_vector.data();

	// allocate memory in GPU (only after this you can init connections)
	cudaMalloc(&gpu_neurons, sizeof(Neuron) * neuron_number);

	// only after cudaMalloc (!)
	init_extensor();

	// get synapse number
	int synapse_number = (int)host_synapses_vector.size();

	printf("Neuron number : %d \n", neuron_number);
	printf("Synapse number : %d \n", synapse_number);

	// convert vector to the array of pointers
	Synapse* host_synapses = host_synapses_vector.data();
	// allocate memory in GPU
	cudaMalloc(&gpu_synapses, sizeof(Synapse) * synapse_number);

	// copy neurons/synapses array to the GPU
	cudaMemcpy(gpu_neurons, host_neurons, sizeof(Neuron) * neuron_number, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_synapses, host_synapses, sizeof(Synapse) * synapse_number, cudaMemcpyHostToDevice);

	// calculate threads block size (2D-square block)
	int block_width = (int)sqrt(synapse_number) + 1;
	dim3 nthreads(block_width, block_width);

	// call the GPU calculation. <<<blocks, threads>>>
	sim_GPU<<<1, nthreads>>>(gpu_neurons, gpu_synapses,
	                         neuron_number, synapse_number,
	                         block_width, sim_step_time);

	// copy neurons/synapses array to the HOST
	cudaMemcpy(host_neurons, gpu_neurons, sizeof(Neuron) * neuron_number, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_synapses, gpu_synapses, sizeof(Synapse) * synapse_number, cudaMemcpyDeviceToHost);


#ifdef DEBUG
	printf("\n---- D E B U G G I N G ----\n");
	// all nrn
	for (int i = 0; i < neuron_number; ++i) {
		printf("DEB NRN: i %d = %d \n", i, host_neurons[i].get_ref_t());
	}
	printf("\n--------\n");
	// all syn
	for (int i = 0; i < synapse_number; ++i) {
		printf("DEB SYN: i %d = %d \n", i, host_synapses[i].syn_delay);
	}
#endif

	cudaFree(gpu_neurons);
	cudaFree(gpu_synapses);
}


int main() {
	srand(time(NULL)); //123
	init_groups();
	simulate();
//	show_results(test_index);

	return 0;
}