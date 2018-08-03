#include <cstdlib>
#include <iostream>
#include <openacc.h>
#include "Neuron.h"
#include "Synapse.h"

using namespace std;

Neuron* neurons;
unsigned int neuron_number;

void show_results(){
	/** Printing results function
	 *
	 * @param  neurons - the vector of pointers
	 * @return void
	 */

	for (int i = 0; i < neuron_number; i++) {
		neurons[i].devcopyout();
		printf("ID: %d\n", neurons[i].id);

		// print spikes
		printf("Spikes: [");

		for(int j = 0; j < 100; j++)
			printf("%f, ", neurons[i].get_spikes()[j]);
		printf("]\n");

		// print V_m
		printf("Voltage: [");
		for(int k = 0; k < 100; k++)
			printf("%f, ", neurons[i].get_mm()[k]);
		printf("]\n---------------\n");
	}
}

void init_neurons(){
	/** Neurons initialization function
	 *
	 * @param: [unsigned int] network_size - number of neurons
	 * @param: [ptr] neurons - vector link of pointers to object
	 *
	 * @return: void
	 */
	neurons = new Neuron[neuron_number];
	for (int i = 0; i < neuron_number; ++i) {
		neurons[i] = Neuron();
		neurons[i].setID(i);

	}
}


/*void init_synapses(vector<Neuron*>& neurons){
	*//** Synapse initialization function
	 *
	 * @param: [ptr] neurons - vector link of pointers to object
	 *
	 * @return: void
	 *//*
	new Synapse(neurons[0], neurons[1], 10.0, 1.0);
}*/


void simulate() {
	/** Simulation main loop function
	 *
	 * @param: [ptr] neurons - vector link of pointers to object
	 *
	 * @return: void
	 */
	float T_sim = 1000.0;
	int step_in_ms = 10;
	clock_t t = clock();

	// Start simulation
	#pragma acc parallel vector_length(100) copyin(neurons[0:neuron_number])
	{
#pragma acc loop

		for (int iter = 0; iter < step_in_ms * T_sim; iter++) {
			for (int i = 0; i < neuron_number; i++) {
				neurons[i].update_state();
			}
		}
	}

	printf ("Clicks: %d clicks | Time: %f s\n", t, ((float)t)/CLOCKS_PER_SEC);
}


int main(int argc, char *argv[]) {
	neuron_number = 5;

	init_neurons();
	//init_synapses(neurons);
	simulate();

	show_results();
	return 0;
}
