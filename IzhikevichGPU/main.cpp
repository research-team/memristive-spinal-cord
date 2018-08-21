//#include <openacc.h>
#include <cstdlib>
#include <iostream>
#include <iostream>
#include <fstream>
#include <random>
#include "Neuron.h"

using namespace std;

const unsigned int neuron_number = 4;
const unsigned int neurons_in_group = 40;
const unsigned int synapses_number = 200;
// Init the neuron objects
typedef Neuron* nrn;
nrn * neurons = new nrn[neuron_number];

const float T_sim = 500.0;
const float ms_in_1step = 0.1f; //0.01f; // ms in one step ALSO: simulation step
const short steps_in_1ms = (short)(1 / ms_in_1step);

Neuron* group11[neurons_in_group];
Neuron* group12[neurons_in_group];
Neuron* group13[neurons_in_group];
Neuron* group14[neurons_in_group];
Neuron* group21[neurons_in_group];
Neuron* group22[neurons_in_group];
Neuron* group23[neurons_in_group];
Neuron* group24[neurons_in_group];
Neuron* group25[neurons_in_group];
Neuron* group26[neurons_in_group];
Neuron* group27[neurons_in_group];
Neuron* group31[neurons_in_group];
Neuron* group32[neurons_in_group];
Neuron* group33[neurons_in_group];
Neuron* group34[neurons_in_group];
Neuron* group35[neurons_in_group];
Neuron* group36[neurons_in_group];
Neuron* group37[neurons_in_group];
Neuron* group41[neurons_in_group];
Neuron* group42[neurons_in_group];
Neuron* group43[neurons_in_group];
Neuron* group44[neurons_in_group];
Neuron* group45[neurons_in_group];
Neuron* group46[neurons_in_group];
Neuron* group47[neurons_in_group];
Neuron* group51[neurons_in_group];
Neuron* group52[neurons_in_group];
Neuron* group53[neurons_in_group];
Neuron* group54[neurons_in_group];
Neuron* group55[neurons_in_group];
Neuron* group56[neurons_in_group];
Neuron* group61[neurons_in_group];
Neuron* group62[neurons_in_group];
Neuron* group63[neurons_in_group];
Neuron* group64[neurons_in_group];
Neuron* group65[neurons_in_group];

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
	for (int i = 0; i < neuron_number; ++i) {
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

void formGroup(int index, Neuron* group) {
	//Neuron* group[neurons_in_group];

	int j = 0;
	for (int i = index; i < index + neurons_in_group; ++i){
		auto t = &group;
		t[j++] = neurons[i];
	}

	//return *group;
}

void connectFixedOutdegree(Neuron* a, Neuron* b, float syn_delay, float weight){
	for(int i = 0; i < neurons_in_group; i++)
	{
		for(int j = 0; j < synapses_number; j++)
		{
			int b_index = rand() % 40;
			a[i].connectWith(&(b[b_index]), syn_delay, weight);
		}
	}
}

void init_groups() {
	int i = 0;
	formGroup(neurons_in_group * i++, *group11);
	formGroup(neurons_in_group * i++, *group12);
	formGroup(neurons_in_group * i++, *group13);
	formGroup(neurons_in_group * i++, *group14);

	formGroup(neurons_in_group * i++, *group21);
	formGroup(neurons_in_group * i++, *group22);
	formGroup(neurons_in_group * i++, *group23);
	formGroup(neurons_in_group * i++, *group24);
	formGroup(neurons_in_group * i++, *group25);
	formGroup(neurons_in_group * i++, *group26);
	formGroup(neurons_in_group * i++, *group27);

	formGroup(neurons_in_group * i++, *group31);
	formGroup(neurons_in_group * i++, *group32);
	formGroup(neurons_in_group * i++, *group33);
	formGroup(neurons_in_group * i++, *group34);
	formGroup(neurons_in_group * i++, *group35);
	formGroup(neurons_in_group * i++, *group36);
	formGroup(neurons_in_group * i++, *group37);

	formGroup(neurons_in_group * i++, *group41);
	formGroup(neurons_in_group * i++, *group42);
	formGroup(neurons_in_group * i++, *group43);
	formGroup(neurons_in_group * i++, *group44);
	formGroup(neurons_in_group * i++, *group45);
	formGroup(neurons_in_group * i++, *group46);
	formGroup(neurons_in_group * i++, *group47);

	formGroup(neurons_in_group * i++, *group51);
	formGroup(neurons_in_group * i++, *group52);
	formGroup(neurons_in_group * i++, *group53);
	formGroup(neurons_in_group * i++, *group54);
	formGroup(neurons_in_group * i++, *group55);
	formGroup(neurons_in_group * i++, *group56);

	formGroup(neurons_in_group * i++, *group61);
	formGroup(neurons_in_group * i++, *group62);
	formGroup(neurons_in_group * i++, *group63);
	formGroup(neurons_in_group * i++, *group64);
	formGroup(neurons_in_group * i++, *group65);
}

void init_synapses() {
	/// Synapse initialization function
	connectFixedOutdegree(*group11, *group12, 2., 15.);
	connectFixedOutdegree(*group11, *group21, 2., 15.);
	connectFixedOutdegree(*group11, *group23, 0.1, 7.);
	connectFixedOutdegree(*group12, *group13, 1., 15.);
	connectFixedOutdegree(*group12, *group14, 1., 15.);
	connectFixedOutdegree(*group13, *group14, 1., 15.);

	connectFixedOutdegree(*group21, *group22, 1., 20.);
	connectFixedOutdegree(*group21, *group23, 1., 4.);
	connectFixedOutdegree(*group22, *group21, 1., 20.);
	connectFixedOutdegree(*group23, *group24, 2., 15.);
	connectFixedOutdegree(*group23, *group31, 1., 15.);
	connectFixedOutdegree(*group23, *group33, .1, 6.);
	connectFixedOutdegree(*group24, *group25, 1., 15.);
	connectFixedOutdegree(*group24, *group26, 1., 15.);
	connectFixedOutdegree(*group24, *group27, 1., 15.);
	connectFixedOutdegree(*group25, *group26, 1., 15.);
	connectFixedOutdegree(*group26, *group27, 1., 15.);

	connectFixedOutdegree(*group31, *group32, 1., 17.);
	connectFixedOutdegree(*group31, *group33, 1.5, 4.);
	connectFixedOutdegree(*group32, *group31, 1., 20.);
	connectFixedOutdegree(*group33, *group34, 2., 17.);
	connectFixedOutdegree(*group33, *group41, 1., 15.);
	connectFixedOutdegree(*group33, *group43, .1, 6.);
	connectFixedOutdegree(*group34, *group35, 1., 15.);
	connectFixedOutdegree(*group34, *group36, 1., 15.);
	connectFixedOutdegree(*group34, *group37, 1., 15.);
	connectFixedOutdegree(*group35, *group36, 1., 15.);
	connectFixedOutdegree(*group36, *group37, 1., 15.);

	connectFixedOutdegree(*group41, *group42, 1., 17.);
	connectFixedOutdegree(*group41, *group43, 1.5, 4.);
	connectFixedOutdegree(*group42, *group41, 1., 20.);
	connectFixedOutdegree(*group43, *group44, 2., 17.);
	connectFixedOutdegree(*group43, *group51, 1., 15.);
	connectFixedOutdegree(*group43, *group53, .1, 9.);
	connectFixedOutdegree(*group44, *group45, 1., 15.);
	connectFixedOutdegree(*group44, *group46, 1., 15.);
	connectFixedOutdegree(*group44, *group47, 1., 15.);
	connectFixedOutdegree(*group45, *group46, 1., 15.);
	connectFixedOutdegree(*group46, *group47, 1., 15.);

	connectFixedOutdegree(*group51, *group52, 1., 17.);
	connectFixedOutdegree(*group51, *group53, 1.5, 4.);
	connectFixedOutdegree(*group52, *group51, 1., 20.);
	connectFixedOutdegree(*group53, *group54, 2., 17.);
	connectFixedOutdegree(*group53, *group55, 1., 15.);
	connectFixedOutdegree(*group53, *group56, 1., 15.);
	connectFixedOutdegree(*group53, *group61, 1., 15.);
	connectFixedOutdegree(*group53, *group63, .1, 6.);
	connectFixedOutdegree(*group54, *group55, 1., 15.);
	connectFixedOutdegree(*group55, *group56, 1., 15.);

	connectFixedOutdegree(*group61, *group62, 1., 17.);
	connectFixedOutdegree(*group61, *group63, 1.5, 4.);
	connectFixedOutdegree(*group62, *group61, 1., 20.);
	connectFixedOutdegree(*group63, *group64, 2., 15.);
	connectFixedOutdegree(*group63, *group65, 1., 15.);
	connectFixedOutdegree(*group64, *group65, 1., 15.);
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
	init_groups();
	init_synapses();
	simulate();
	show_results();
	return 0;
}
