/*
 * https://developer.nvidia.com/cuda-gpus
 * GeForce 840M	-- Compute Capability 5.0
 */

#ifdef __JETBRAINS_IDE__
	#define __host__
	#define __device__
	#define __shared__
	#define __constant__
	#define __global__
#endif
#include <cstdlib>
#include <stdio.h>
#include <math.h>

__device__
int anotherFunc(int id) {
	return id * 100;
}


class Neuron {
public:
	bool has_spike = false;
};


struct Synapse {
public:
	Neuron* post_neuron{}; // post neuron
	Neuron* pre_neuron{}; // pre neuron
	int syn_delay{};		 // [steps] synaptic delay. Converts from ms to steps
	int curr_syn_delay{};		 // [steps] synaptic delay. Converts from ms to steps
	float weight{};		 // [pA] synaptic weight
	int timer = -1;
};


__global__
void simulation_GPU(Neuron *neurons, Synapse *synapses, int nrn_size, int syn_size, int sim_step_time) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * 11;
	if (thread_id < nrn_size) {
		printf("thread glob : %d, thread X: %d, thread Y : %d \n", thread_id, x, y);
		for(int step_time = 0; step_time < sim_step_time; ++step_time) {
			printf("\t %d %d\n", step_time, thread_id);

			// wait until all threads in block have finished to this point
			__syncthreads();
		}

	}
}


int main() {
	// simulation properties
	int neuron_number = 112;
	int synapse_number = 10;
	int sim_time = 3;
	float step = 0.1;
	int sim_step_time = (int)(sim_time / step);

	// GPU properties
	int thread_num = 64;
	int blocks_num = neuron_number / thread_num;
	if( neuron_number % thread_num)
		blocks_num++;

	// allocate memory in HOST
	Neuron *host_neurons = new Neuron[neuron_number];
	Synapse *host_synapses = new Synapse[synapse_number];

	// allocate memory in GPU
	Neuron *device_neurons;
	cudaMalloc(&device_neurons, sizeof(Neuron) * neuron_number);

	Synapse *device_synapses;
	cudaMalloc(&device_synapses, sizeof(Synapse) * synapse_number);

	// prepare data
	for (int i = 0; i < neuron_number; ++i) {
		host_neurons[i].has_spike = true;
	}

	// prepare data
	for (int i = 0; i < synapse_number; ++i) {
		host_synapses[i].curr_syn_delay = 10;
	}

	// copy neuron pointers to the GPU
	cudaMemcpy(device_neurons, host_neurons, sizeof(Neuron) * neuron_number, cudaMemcpyHostToDevice);
	cudaMemcpy(device_synapses, host_synapses, sizeof(Synapse) * synapse_number, cudaMemcpyHostToDevice);

	// call the GPU calculation
	printf("N number = %d \n", neuron_number);
	int x = (int)sqrt(neuron_number) + 1;
	dim3 nthreads(x, x);

	simulation_GPU<<<1, nthreads>>>(device_neurons, device_synapses, neuron_number, synapse_number, sim_step_time);

	// copy out neurons pointers to the HOST
	cudaMemcpy(host_neurons, device_neurons, sizeof(Neuron) * neuron_number, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_synapses, device_synapses, sizeof(Synapse) * synapse_number, cudaMemcpyDeviceToHost);

	/*
	 // copy out synapse pointers to the HOST
	// FixMe doesn't work properly
	for (int i = 0; i < neuron_number; ++i) {
		Synapse *b;
		cudaMemcpy(b, device_neurons[i].synapses, sizeof(Synapse) * 10, cudaMemcpyDeviceToHost);

		host_neurons[i].synapses = b;
	}

	// DEBUGGING : show the data
	for(int i = 0; i < neuron_number; ++i) {
		printf("gpu [%d] = %d \n", i, host_neurons[i].tag);
		for(int j = 0; j < 10; j++)
			printf("%d ", host_neurons[i].synapses[j].a);
		printf("\n");
	}
	 */

	return 0;
}