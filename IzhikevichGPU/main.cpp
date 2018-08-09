#include <openacc.h>
#include <cstdlib>
#include <iostream>
#include <iostream>
#include <fstream>
#include <random>
#include "Neuron.h"

using namespace std;

const unsigned int neuron_number = 4;

// Init the neuron objects
typedef Neuron* nrn;
nrn * neurons = new nrn[neuron_number];

const float T_sim = 500.0;
const float ms_in_1step = 0.1f; //0.01f; // ms in one step ALSO: simulation step
const short steps_in_1ms = (short)(1 / ms_in_1step);

// random
//random_device rd;
//mt19937 gen(rd());
//uniform_real_distribution<int> randomID(0, neuron_number-1);

void show_results() {
	/// Printing results function
	ofstream myfile;
	myfile.open ("/home/alex/sim_results.txt");

	for (int nrn_id = 0; nrn_id < neuron_number; nrn_id++) {
		myfile << "ID: "<< neurons[nrn_id]->getID() << "\n";
		myfile << "Obj: "<< neurons[nrn_id]->getThis() << "\n";
		myfile << "Iter: "<< neurons[nrn_id]->getSimulationIter() << "\n";

		if (neurons[nrn_id]->withSpikedetector()) {
			myfile << "Spikes: [";
			for (int j = 0; j < 100; j++) {
				myfile << neurons[nrn_id]->getSpikes()[j] << ", ";
			}
			myfile << "]\n";
		}

		if (neurons[nrn_id]->withMultimeter()) {
			myfile << "Voltage: [";
			for (int k = 0; k < neurons[nrn_id]->getVoltageArraySize(); k++) {
				myfile << neurons[nrn_id]->getVoltage()[k] << ", ";
			}
			myfile << "]\n";

			myfile << "I_potential: [";
			for(int k = 0; k < neurons[nrn_id]->getVoltageArraySize(); k++){
				myfile << neurons[nrn_id]->getCurrents()[k] << ", ";
			}
		}

		myfile << "]\n---------------\n";
	}
	myfile.close();
}

void init_neurons() {
	/// Neurons initialization function
	for (int i = 0; i < neuron_number; ++i) {
		neurons[i] = new Neuron(i, 2.0f);
	}

	// additional devices to the neurons
	for (int i = 0; i < 4; ++i) {
		neurons[i]->addSpikedetector();
		neurons[i]->addMultimeter();
	}

	// TEST connections
	//for (int i = 0; i < neuron_number; ++i) {
	//	for (int j = 0; j < 30; ++j) {
	//		neurons[i]->connectWith( neurons[rand() % neuron_number], 1.0, (rand() % 2)? 300 : -300); // neuronID, delay, weight
	//	}
	//}
	//for(int i = 0; i < neuron_number; i+=10)
	neurons[0]->addGenerator(180.f);

}

/*
Neuron* formGroup() {
	return
}

void connectFixedOutdegree(){

}
*/

void init_synapses() {
	/// Synapse initialization function
	neurons[0]->connectWith(neurons[1], 3.0, 300.0);
	neurons[0]->connectWith(neurons[2], 3.0, 300.0);
	neurons[1]->connectWith(neurons[3], 3.0, 300.0);
	neurons[2]->connectWith(neurons[3], 3.0, 300.0);
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
