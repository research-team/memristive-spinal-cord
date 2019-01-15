#include <cstdlib>
#include <stdio.h>
#include <math.h>

#ifdef __JETBRAINS_IDE__
	#define __host__
	#define __device__
	#define __shared__
	#define __constant__
	#define __global__
#endif

#define DEBUG

struct Neuron {
public:
	int ref_t{};
	bool has_spike{};
};

struct Synapse {
public:
	Neuron* post_neuron{}; // post neuron
	Neuron* pre_neuron{}; // pre neuron
	int syn_delay{};		 // [steps] synaptic delay. Converts from ms to steps
	int curr_syn_delay;		 // [steps] synaptic delay. Converts from ms to steps
	float weight{};		 // [pA] synaptic weight
	int timer = -1;
};

__device__
void neuron_update(Neuron *neuron, int step_time, int thread_id) {
	neuron->ref_t += 5;
	#ifdef DEBUG
		printf("S: %d, T: %d, NRN %p \n", step_time, thread_id, neuron);
	#endif
}

__device__
void synapse_update(Synapse *synapse, int step_time, int thread_id) {
	synapse->curr_syn_delay += 10;
	#ifdef DEBUG
		printf("S: %d, T: %d, SYN %p \n", step_time, thread_id, synapse);
	#endif
}

__global__
void sim_GPU(Neuron *neurons, Synapse *synapses, int nrn_size, int syn_size, int block_width, int sim_step_time) {
	// get thread ID
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thread_id = x + y * block_width;

	// main simulation loop
	for(int step_time = 0; step_time < sim_step_time; ++step_time) {
		// update neurons state by pointer
		if (thread_id < nrn_size) {
			neuron_update(&neurons[thread_id], step_time, thread_id);
		}
		// wait until all threads in block have finished to this point and calculated neurons state
		__syncthreads();

		// update synapses state by pointer
		if (thread_id < syn_size) {
			synapse_update(&synapses[thread_id], step_time, thread_id);
		}
		// wait until all threads in block have finished to this point and calculated synapses state
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
		host_neurons[i].ref_t = i;
	}

	// prepare synapse data
	for (int i = 0; i < synapse_number; ++i) {
		host_synapses[i].curr_syn_delay = i;
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
			printf("DEB NRN: i %d = %d \n", i, host_neurons[i].ref_t);
		}
		printf("\n--------\n");
		// all syn
		for (int i = 0; i < synapse_number; ++i) {
			printf("DEB SYN: i %d = %d \n", i, host_synapses[i].curr_syn_delay);
		}
	#endif

	return 0;
}