#include <cstdlib>
#include <iostream>
//#include <openacc.h>
#include "Neuron.h"
//#include "Synapse.h"

#include <iostream>
#include <fstream>
using namespace std;

Neuron* neurons;
unsigned int neuron_number = 200;

float T_sim = 1000.0;
float ms_in_1step = 0.1f; //0.01f; // ms in one step ALSO: simulation step
short steps_in_1ms = (short)(1 / ms_in_1step);


void show_results(){
	/// Printing results function
	ofstream myfile;
	myfile.open ("/home/alex/sim_results.txt");


	for (int nrn_id = 0; nrn_id < 10; nrn_id++) {
		myfile << "ID: "<< neurons[nrn_id].getID() << "\n";
		myfile << "Obj: "<< neurons[nrn_id].getThis() << "\n";
		myfile << "Iter: "<< neurons[nrn_id].getSimIter() << "\n";

		// print spikes
		myfile << "Spikes: [";
		for(int j = 0; j < 100; j++) {
			float a = neurons[nrn_id].get_spikes()[j];
			myfile << a << ", ";
		}
		myfile << "]\n";

		// print V_m
		myfile << "Voltage: [";
		for(int k = 0; k < neurons[nrn_id].get_mm_size(); k++){
			float a = neurons[nrn_id].get_mm()[k];
			myfile << a << ", ";
		}
		myfile << "]\n---------------\n";
	}
	myfile.close();
}

void init_neurons(){
	/// Neurons initialization function

	neurons = new Neuron[neuron_number];
	for (int id = 0; id < neuron_number; ++id) {
		neurons[id] = Neuron();
		neurons[id].setID(id);
		neurons[id].setI( rand() / float(RAND_MAX) * 70.f + 1.f );
	}
}


/*
void init_synapses(){
	/// Synapse initialization function
	new Synapse(&neurons[0], &neurons[1], 10.0, 1.0);
}
*/


void simulate() {
	/// Simulation main loop function
	int id = 0;
	int iter = 0;
	clock_t t = clock();

	//#pragma acc data copy(neurons)
	//#pragma acc parallel vector_length(200)
	//{
		//#pragma acc loop gang worker seq
		for (iter = 0; iter < T_sim * steps_in_1ms; iter++) {
			//#pragma acc loop vector
			for (id = 0; id < neuron_number; id++) {
				neurons[id].update_state();
			}
		}
	//}
	printf ("Time: %f s\n", (float)t / CLOCKS_PER_SEC);
	/*
	#pragma acc data copy(neurons)
	#pragma acc parallel vector_length(neuron_number)
	{
		#pragma acc loop gang worker seq
		for (iter = 0; iter < T_sim * steps_in_1ms; iter++) {
			#pragma acc loop vector
			for (id = 0; id < neuron_number; id++) {
				neurons[id].update_state();
			}
		}
	}
	 */
}

int main(int argc, char *argv[]) {
	init_neurons();
	//init_synapses(neurons);
	simulate();
	show_results();
	return 0;
}
