#include <openacc.h>
#include <cstdlib>
#include <iostream>
#include <iostream>
#include <fstream>
#include <random>
#include "Neuron.h"

using namespace std;

const unsigned int neuron_number = 200;

// Init the neuron objects
typedef Neuron* nrn;
nrn * neurons = new nrn[neuron_number];

static float T_sim = 1000.0;
static float ms_in_1step = 0.1f; //0.01f; // ms in one step ALSO: simulation step
static short steps_in_1ms = (short)(1 / ms_in_1step);

// random
random_device rd;
mt19937 mt(rd());
uniform_real_distribution<float> ref_t_rand(1.0f, 3.0f);

void show_results() {
	/// Printing results function
	ofstream myfile;
	myfile.open ("/home/alex/sim_results.txt");

	for (int nrn_id = 0; nrn_id < 2; nrn_id++) {
		myfile << "ID: "<< neurons[nrn_id]->getID() << "\n";
		myfile << "Obj: "<< neurons[nrn_id]->getThis() << "\n";
		myfile << "Iter: "<< neurons[nrn_id]->getSimIter() << "\n";

		// print spikes
		myfile << "Spikes: [";
		for(int j = 0; j < 100; j++) {
			myfile << neurons[nrn_id]->get_spikes()[j] << ", ";
		}
		myfile << "]\n";

		// print V_m
		myfile << "Voltage: [";
		for(int k = 0; k < neurons[nrn_id]->get_mm_size(); k++){
			myfile << neurons[nrn_id]->get_mm()[k] << ", ";
		}
		myfile << "]\n---------------\n";
	}
	myfile.close();
}

void init_neurons() {
	/// Neurons initialization function
	for (int i = 0; i < neuron_number; ++i) {
		neurons[i] = new Neuron(i, 3.0f); // Float pointrand() / float(RAND_MAX) * 70.f + 1.f);
	}
	neurons[0]->makeGenerator(70.f);
}

void init_synapses() {
	/// Synapse initialization function
	neurons[0]->add_neighbor(neurons[1], 3.0, 300.0);
}

void simulate() {
	/// Simulation main loop function
	int id = 0;
	int iter = 0;
	clock_t t = clock();

	#pragma acc data copy(neurons)
	#pragma acc parallel vector_length(200)
	{
		#pragma acc loop gang worker seq
		for (iter = 0; iter < T_sim * steps_in_1ms; iter++) {
			#pragma acc loop vector
			for (id = 0; id < neuron_number; id++) {
				neurons[id]->update_state();
			}
		}
	}
	printf ("Time: %f s\n", (float)t / CLOCKS_PER_SEC);
}

int main(int argc, char *argv[]) {
	init_neurons();
	init_synapses();
	simulate();
	show_results();
	return 0;
}
