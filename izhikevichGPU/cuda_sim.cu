#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include "Synapse.cpp"

#ifdef __JETBRAINS_IDE__
	#define __host__
	#define __device__
	#define __shared__
	#define __constant__
	#define __global__
#endif

#define DEBUG

__global__
void sim_GPU(Neuron *neurons, Synapse *synapses, int nrn_size, int syn_size, int block_width, int sim_step_time) {
	/*
	 *
	 */
	// get thread ID
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * block_width;

	// main simulation loop
	for(int sim_iter = 0; sim_iter < sim_step_time; ++sim_iter) {
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
			neurons[thread_id].has_spike = false;
		}
		__syncthreads();
	}
}


int main() {
	// simulation properties
	int neuron_number = 30;
	int synapse_number = 50;
	int sim_time = 3;
	float step = 0.1;
	int sim_step_time = (int)(sim_time / step);

	// allocate memory in HOST
	Neuron *host_neurons = new Neuron[neuron_number];
	Synapse *host_synapses = new Synapse[synapse_number];
	// ToDo create a vector, then convert to this type of data (array of pointers)

	// allocate memory in GPU
	Neuron *gpu_neurons;
	cudaMalloc(&gpu_neurons, sizeof(Neuron) * neuron_number);

	Synapse *gpu_synapses;
	cudaMalloc(&gpu_synapses, sizeof(Synapse) * synapse_number);

	// prepare neuron data
	for (int i = 0; i < neuron_number; ++i) {
		host_neurons[i].set_sim_time(sim_time);
		host_neurons[i].set_ref_t(3.0);
		host_neurons[i].set_has_spike();
	}

	// prepare synapse data
	for (int i = 0; i < synapse_number; ++i) {
		int pre_id = rand() % neuron_number;
		int post_id = rand() % neuron_number;
		host_synapses[i].curr_syn_delay = i;
		// (!) pointers should point to the memory in GPU!
		host_synapses[i].pre_neuron = &gpu_neurons[pre_id];
		host_synapses[i].post_neuron = &gpu_neurons[post_id];
	}

	// copy neurons/synapses array to the GPU
	cudaMemcpy(gpu_neurons, host_neurons, sizeof(Neuron) * neuron_number, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_synapses, host_synapses, sizeof(Synapse) * synapse_number, cudaMemcpyHostToDevice);

	// calulate threads block size
	int block_width = (int)sqrt(synapse_number) + 1;
	dim3 nthreads(block_width, block_width);	// 2D-square block

	// call the GPU calculation. <<<blocks, threads>>>
	sim_GPU <<<1, nthreads>>>(gpu_neurons, gpu_synapses, neuron_number, synapse_number, block_width, sim_step_time);

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
			printf("DEB SYN: i %d = %d \n", i, host_synapses[i].curr_syn_delay);
		}
	#endif

	return 0;
}