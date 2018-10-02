#include <cstdlib>
#include <iostream>
#include <iostream>
#include <fstream>
#include <random>
#include "Neuron.h"
#include <unistd.h>



using namespace std;

const unsigned int neuron_number = 2000;
const unsigned int neurons_in_group = 40;
const unsigned int neurons_in_ip = neurons_in_group;
const unsigned int neurons_in_test = 169;
const unsigned int synapses_number = 10;
// Init the neuron objects
Neuron* neurons[neuron_number];

extern const float T_sim = 150.0;
const float ms_in_1step = 0.1f; //0.01f; // ms in one step ALSO: simulation step
const short steps_in_1ms = (short) (1 / ms_in_1step);

unsigned int groups_number = 0;

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

Neuron* ip1[neurons_in_group];
Neuron* ip2[neurons_in_group];
Neuron* ip3[neurons_in_group];
Neuron* ip4[neurons_in_group];
Neuron* ip5[neurons_in_group];
Neuron* ip6[neurons_in_group];
Neuron* test[neurons_in_group];

void show_results() {
	/// Printing results function
	char cwd[256];
	getcwd(cwd, sizeof(cwd));
	printf("Save results to: %s", cwd);
	string filename = "/sim_results.txt";
	ofstream myfile;
	myfile.open (cwd + filename);
	for (auto &neuron : neurons) {
		myfile << "ID: "<< neuron->getID() << "\n";
		myfile << "Obj: "<< neuron->getThis() << "\n";
		myfile << "Name: "<< neuron->getName() << "\n";
		myfile << "Iter: "<< neuron->getSimulationIter() << "\n";

		if (neuron->withSpikedetector()) {
			myfile << "Spikes: [";
			for (int j = 0; j < neuron->getIterSpikesArray(); j++) {
				myfile << neuron->getSpikes()[j] << ", ";
			}
			myfile << "]\n";
		}

		if (neuron->withMultimeter()) {
			myfile << "Voltage: [";
			for (int k = 0; k < neuron->getVoltageArraySize(); k++) {
				myfile << neuron->getVoltage()[k] << ", ";
			}
			myfile << "]\n";

			myfile << "I_potential: [";
			for(int k = 0; k < neuron->getVoltageArraySize(); k++){
				myfile << neuron->getCurrents()[k] << ", ";
			}
			myfile << "]\n";
		}
		myfile << "\n---------------\n";
	}
	myfile.close();
}

void init_neurons() {
	/// Neurons initialization function
	for (int i = 0; i < neuron_number; ++i) {
		neurons[i] = new Neuron(i, 2.0f);
	}

	// additional devices to the neurons
	for (auto &neuron : neurons)
		neuron->addSpikedetector();
}

void formGroup(int index, Neuron* group[], char* name, int nrns_in_group = neurons_in_group) {
	int j = 0;
	printf("Formed %s IDs [%d ... %d]\n", name, index, index + nrns_in_group - 1);
	for (int i = index; i < index + nrns_in_group; ++i){
		group[j] = neurons[i];
		neurons[i]->name = name;
		///printf("%s %d %d\n", name, i, j);
		///printf("i = %d j = %d, obj: %p, id(%d)\n", i, j, group[j]->getThis(), group[j]->getID());
		j++;
	}
}

void connectFixedOutDegree(Neuron* a[], Neuron* b[], float syn_delay, float weight, int n = neurons_in_group) {
	for (int i = 0; i < n; i++) {
		for(int j = 0; j < synapses_number; j++) {
			int b_index = rand() % n;
			a[i]->connectWith(a[i], b[b_index], syn_delay, weight);
			//printf("%s [%d/%d] connected to %s [%d/%d]. W %f, D %f \n", a[i]->getName(), i+1, n, b[b_index]->getName(), j+1, n, weight, syn_delay);
		}
	}
}


void init_groups() {
	int i = 0;
	formGroup(neurons_in_group * i++, group11, "group11");
	formGroup(neurons_in_group * i++, group12, "group12");
	formGroup(neurons_in_group * i++, group13, "group13");
	formGroup(neurons_in_group * i++, group14, "group14");
	formGroup(neurons_in_group * i++, group21, "group21");
	formGroup(neurons_in_group * i++, group22, "group22");
	formGroup(neurons_in_group * i++, group23, "group23");
	formGroup(neurons_in_group * i++, group24, "group24");
	formGroup(neurons_in_group * i++, group25, "group25");
	formGroup(neurons_in_group * i++, group26, "group26");
	formGroup(neurons_in_group * i++, group27, "group27");
	formGroup(neurons_in_group * i++, group31, "group31");
	formGroup(neurons_in_group * i++, group32, "group32");
	formGroup(neurons_in_group * i++, group33, "group33");
	formGroup(neurons_in_group * i++, group34, "group34");
	formGroup(neurons_in_group * i++, group35, "group35");
	formGroup(neurons_in_group * i++, group36, "group36");
	formGroup(neurons_in_group * i++, group37, "group37");
	formGroup(neurons_in_group * i++, group41, "group41");
	formGroup(neurons_in_group * i++, group42, "group42");
	formGroup(neurons_in_group * i++, group43, "group43");
	formGroup(neurons_in_group * i++, group44, "group44");
	formGroup(neurons_in_group * i++, group45, "group45");
	formGroup(neurons_in_group * i++, group46, "group46");
	formGroup(neurons_in_group * i++, group47, "group47");
	formGroup(neurons_in_group * i++, group51, "group51");
	formGroup(neurons_in_group * i++, group52, "group52");
	formGroup(neurons_in_group * i++, group53, "group53");
	formGroup(neurons_in_group * i++, group54, "group54");
	formGroup(neurons_in_group * i++, group55, "group55");
	formGroup(neurons_in_group * i++, group56, "group56");
	formGroup(neurons_in_group * i++, group61, "group61");
	formGroup(neurons_in_group * i++, group62, "group62");
	formGroup(neurons_in_group * i++, group63, "group63");
	formGroup(neurons_in_group * i++, group64, "group64");
	formGroup(neurons_in_group * i++, group65, "group65");

	formGroup(neurons_in_group * i++, ip1, "ip1", neurons_in_ip);
	formGroup(neurons_in_ip * i++, ip2, "ip2", neurons_in_ip);
	formGroup(neurons_in_ip * i++, ip3, "ip3", neurons_in_ip);
	formGroup(neurons_in_ip * i++, ip4, "ip4", neurons_in_ip);
	formGroup(neurons_in_ip * i++, ip5, "ip5", neurons_in_ip);
	formGroup(neurons_in_ip * i++, ip6, "ip6", neurons_in_ip);

	formGroup(neurons_in_ip * i++, test, "test", neurons_in_test);

	for (auto &neuron : group11) {
		printf("Neuron %p\n", neuron->getThis());
		neuron->addGenerator(180.0f);
	}

	for (auto &neuron : neurons) {
		neuron->addSpikedetector();
		neuron->addMultimeter();
	}
}

void init_synapses() {
	/// Synapse initialization function
	connectFixedOutDegree(group11, group12, 2.0, 150.0);
	connectFixedOutDegree(group11, group21, 2.0, 150.0);
	connectFixedOutDegree(group11, group23, 0.1, 70.0);
	connectFixedOutDegree(group12, group13, 1.0, 150.0);
	connectFixedOutDegree(group12, group14, 1.0, 150.0);
	connectFixedOutDegree(group13, group14, 1.0, 150.0);

	connectFixedOutDegree(group21, group22, 1.0, 200.0);
	connectFixedOutDegree(group21, group23, 1.0, 40.0);
	connectFixedOutDegree(group22, group21, 1.0, 200.0);
	connectFixedOutDegree(group23, group24, 2.0, 150.0);
	connectFixedOutDegree(group23, group31, 1.0, 150.0);
	connectFixedOutDegree(group23, group33, 0.1, 60.0);
	connectFixedOutDegree(group24, group25, 1.0, 150.0);
	connectFixedOutDegree(group24, group26, 1.0, 150.0);
	connectFixedOutDegree(group24, group27, 1.0, 150.0);
	connectFixedOutDegree(group25, group26, 1.0, 150.0);
	connectFixedOutDegree(group26, group27, 1.0, 150.0);
//
	connectFixedOutDegree(group31, group32, 1.0, 170.0);
	connectFixedOutDegree(group31, group33, 1.5, 40.0);
	connectFixedOutDegree(group32, group31, 1.0, 200.0);
	connectFixedOutDegree(group33, group34, 2.0, 170.0);
	connectFixedOutDegree(group33, group41, 1.0, 150.0);
	connectFixedOutDegree(group33, group43, 0.1, 60.0);
	connectFixedOutDegree(group34, group35, 1.0, 150.0);
	connectFixedOutDegree(group34, group36, 1.0, 150.0);
	connectFixedOutDegree(group34, group37, 1.0, 150.0);
	connectFixedOutDegree(group35, group36, 1.0, 150.0);
	connectFixedOutDegree(group36, group37, 1.0, 150.0);
//
	connectFixedOutDegree(group41, group42, 1.0, 170.0);
	connectFixedOutDegree(group41, group43, 1.5, 40.0);
	connectFixedOutDegree(group42, group41, 1.0, 200.0);
	connectFixedOutDegree(group43, group44, 2.0, 170.0);
	connectFixedOutDegree(group43, group51, 1.0, 150.0);
	connectFixedOutDegree(group43, group53, 0.1, 90.0);
	connectFixedOutDegree(group44, group45, 1.0, 150.0);
	connectFixedOutDegree(group44, group46, 1.0, 150.0);
	connectFixedOutDegree(group44, group47, 1.0, 150.0);
	connectFixedOutDegree(group45, group46, 1.0, 150.0);
	connectFixedOutDegree(group46, group47, 1.0, 150.0);
//
	connectFixedOutDegree(group51, group52, 1.0, 170.0);
	connectFixedOutDegree(group51, group53, 1.5, 40.0);
	connectFixedOutDegree(group52, group51, 1.0, 200.0);
	connectFixedOutDegree(group53, group54, 2.0, 170.0);
	connectFixedOutDegree(group53, group55, 1.0, 150.0);
	connectFixedOutDegree(group53, group56, 1.0, 150.0);
	connectFixedOutDegree(group53, group61, 1.0, 150.0);
	connectFixedOutDegree(group53, group63, 0.1, 60.0);
	connectFixedOutDegree(group54, group55, 1.0, 150.0);
	connectFixedOutDegree(group55, group56, 1.0, 150.0);
//
	connectFixedOutDegree(group61, group62, 1.0, 170.0);
	connectFixedOutDegree(group61, group63, 1.5, 40.0);
	connectFixedOutDegree(group62, group61, 1.0, 200.0);
	connectFixedOutDegree(group63, group64, 2.0, 150.0);
	connectFixedOutDegree(group63, group65, 1.0, 150.0);
	connectFixedOutDegree(group64, group65, 1.0, 150.0);
//
	connectFixedOutDegree(group14, ip1, 1.0, 170.0);
	connectFixedOutDegree(group27, ip2, 1.0, 170.0);
	connectFixedOutDegree(group37, ip3, 1.0, 170.0);
	connectFixedOutDegree(group47, ip4, 1.0, 170.0);
	connectFixedOutDegree(group56, ip5, 1.0, 170.0);
	connectFixedOutDegree(group65, ip6, 1.0, 170.0);
	connectFixedOutDegree(ip1, test, 1.0, 200.0, neurons_in_test);
	connectFixedOutDegree(ip2, test, 1.0, 200.0, neurons_in_test);
	connectFixedOutDegree(ip3, test, 1.0, 200.0, neurons_in_test);
	connectFixedOutDegree(ip4, test, 1.0, 200.0, neurons_in_test);
	connectFixedOutDegree(ip5, test, 1.0, 200.0, neurons_in_test);
	connectFixedOutDegree(ip6, test, 1.0, 200.0, neurons_in_test);
}

void simulate() {
	/// Simulation main loop function
	int id = 0;
	int iter = 0;
	printf("Start sim\n");

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

int main() {
	init_neurons();
	init_groups();
	init_synapses();
	simulate();
	show_results();
	return 0;
}