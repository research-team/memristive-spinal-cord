#include <stdio.h>
#include <cstdlib>
#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__
#endif


// called from GPU to the GPU
__device__
int anotherFunc(int id) {
	return id * 100;
}


struct Neuron {
	int tag;
};


__global__
void simulation_GPU(Neuron *neurons, int size) {
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for(int neuronID = 0; neuronID < size; ++neuronID) {
		if (neuronID == threadID) {
			neurons[neuronID].tag = anotherFunc(neuronID);
		}
	}
}


int main() {
	int N = 100;

	Neuron * cpu_x = new Neuron[N];

	Neuron * gpu_x;
	cudaMalloc(&gpu_x, sizeof(Neuron) * N);

	for (int i = 0; i < N; ++i) {
		cpu_x[i].tag = i;
	}

	cudaMemcpy(gpu_x, cpu_x, sizeof(Neuron) * N, cudaMemcpyHostToDevice);


	simulation_GPU<<<1, 100>>>(gpu_x, N);


	cudaMemcpy(cpu_x, gpu_x, sizeof(Neuron) * N, cudaMemcpyDeviceToHost);


	printf("\n----------\n");

	for(int i = 0; i < N; ++i)
		printf("gpu [%d] = %d \n", i, cpu_x[i].tag);


	return 0;
}
