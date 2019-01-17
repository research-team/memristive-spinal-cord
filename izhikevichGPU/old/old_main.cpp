#include <cstdlib>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <math.h>
#include <ctime>
#include "old_neuron.h"

using namespace std;

const unsigned int neuron_number = 2300;
const unsigned int neurons_in_group = 40;
const unsigned int neurons_in_moto = 169;
const unsigned int synapses_number = 40;

// Init the neuron objects
Neuron *NEURONS[neuron_number];

// 6 cms = 125
// 15 cms = 50
// 21 cms = 25
const float speed_to_time = 25;
const float INH_COEF = 1.0;
const int EES_FREQ = 40;

extern const float T_sim = speed_to_time * 6;

const float ms_in_1step = 0.1f; // ms in one step  simulation step
const short steps_in_1ms = (short) (1 / ms_in_1step);

int global_id_start = 0;

// segment start
Neuron *C1[neurons_in_group];
Neuron *C2[neurons_in_group];
Neuron *C3[neurons_in_group];
Neuron *C4[neurons_in_group];
Neuron *C5[neurons_in_group];

Neuron *D1_1[neurons_in_group];
Neuron *D1_2[neurons_in_group];
Neuron *D1_3[neurons_in_group];
Neuron *D1_4[neurons_in_group];

Neuron *D2_1[neurons_in_group];
Neuron *D2_2[neurons_in_group];
Neuron *D2_3[neurons_in_group];
Neuron *D2_4[neurons_in_group];

Neuron *D3_1[neurons_in_group];
Neuron *D3_2[neurons_in_group];
Neuron *D3_3[neurons_in_group];
Neuron *D3_4[neurons_in_group];

Neuron *D4_1[neurons_in_group];
Neuron *D4_2[neurons_in_group];
Neuron *D4_3[neurons_in_group];
Neuron *D4_4[neurons_in_group];

Neuron *D5_1[neurons_in_group];
Neuron *D5_2[neurons_in_group];
Neuron *D5_3[neurons_in_group];
Neuron *D5_4[neurons_in_group];

Neuron *G1_1[neurons_in_group];
Neuron *G1_2[neurons_in_group];
Neuron *G1_3[neurons_in_group];

Neuron *G2_1[neurons_in_group];
Neuron *G2_2[neurons_in_group];
Neuron *G2_3[neurons_in_group];

Neuron *G3_1[neurons_in_group];
Neuron *G3_2[neurons_in_group];
Neuron *G3_3[neurons_in_group];

Neuron *G4_1[neurons_in_group];
Neuron *G4_2[neurons_in_group];
Neuron *G4_3[neurons_in_group];

Neuron *G5_1[neurons_in_group];
Neuron *G5_2[neurons_in_group];
Neuron *G5_3[neurons_in_group];

Neuron *IP_E[neurons_in_moto];
Neuron *MP_E[neurons_in_moto];
Neuron *EES[neurons_in_group];

Neuron *inh_group3[neurons_in_group];
Neuron *inh_group4[neurons_in_group];
Neuron *inh_group5[neurons_in_group];

Neuron *ees_group1[neurons_in_group];
Neuron *ees_group2[neurons_in_group];
Neuron *ees_group3[neurons_in_group];
Neuron *ees_group4[neurons_in_group];

void show_results(int test_index) {
	/// Printing results function
	char cwd[256];
	getcwd(cwd, sizeof(cwd));
	printf("Save results to: %s", cwd);
	string filename = "/" + std::to_string(test_index) + ".dat";
	ofstream myfile;
	myfile.open(cwd + filename);
	for (auto &neuron : NEURONS) {
		if (neuron->withMultimeter()) {
			float time = 0;
			for (int k = 0; k < neuron->getVoltageArraySize(); k++) {
				myfile << neuron->getID() << " " << time / 10 << " " << neuron->getVoltage()[k] << "\n";
				time += 1;
			}
		}
	}
	myfile.close();
}


void init_neurons() {
	/// Neurons initialization function
	for (int i = 0; i < neuron_number; ++i)
		NEURONS[i] = new Neuron(i, 2.0f);
}


void formGroup(Neuron *nrn_group[], char *name, int nrns_in_group = neurons_in_group, float Istep=0.0f) {
	int j = 0;
	printf("Formed %s IDs [%d ... %d] = %d\n", name, global_id_start, global_id_start + nrns_in_group - 1,
		   nrns_in_group);
	for (int i = global_id_start; i < global_id_start + nrns_in_group; ++i) {
		nrn_group[j] = NEURONS[i];
		NEURONS[i]->name = name;
		if (Istep != 0)
			NEURONS[i]->changeIstep(Istep);
		j++;
	}

	global_id_start += nrns_in_group;
}


int get_random_neighbor(int pre_neuron_index, int post_neurons) {
	int post_neuron_index = rand() % post_neurons;
	while (post_neuron_index == pre_neuron_index)
		post_neuron_index = rand() % post_neurons;
	return post_neuron_index;
}


void connectFixedOutDegree(Neuron *pre[], Neuron *post[],
						   float syn_delay, float weight, int post_neuron_number = neurons_in_group) {
	weight *= 0.4; // 0.8

	float time_delta = syn_delay * 0.4f; //0.2
	float weight_delta = weight * 0.3f;

	for (int pre_index = 0; pre_index < post_neuron_number; pre_index++) {
		for (int post_index = 0; post_index < synapses_number; post_index++) {
			int post_neuron_index = get_random_neighbor(pre_index, post_neuron_number);
			float syn_delay_dist = float(rand()) / float(RAND_MAX) * 2 * time_delta + syn_delay - time_delta;
			float syn_weight_dist = float(rand()) / float(RAND_MAX) * 2 * weight_delta + weight - weight_delta;
			//printf("Original: %f; \t New: %f \n", weight, syn_weight_dist);
			pre[pre_index]->connectWith(pre[pre_index], post[post_neuron_index], syn_delay_dist, syn_weight_dist);
		}
	}
}


void init_groups() {
	formGroup(C1, "C1");
	formGroup(C2, "C2");
	formGroup(C3, "C3");
	formGroup(C4, "C4");
	formGroup(C5, "C5");

	formGroup(D1_1, "D1_1");
	formGroup(D1_2, "D1_2");
	formGroup(D1_3, "D1_3");
	formGroup(D1_4, "D1_4");

	formGroup(D2_1, "D2_1");
	formGroup(D2_2, "D2_2");
	formGroup(D2_3, "D2_3");
	formGroup(D2_4, "D2_4");

	formGroup(D3_1, "D3_1");
	formGroup(D3_2, "D3_2");
	formGroup(D3_3, "D3_3");
	formGroup(D3_4, "D3_4");

	formGroup(D4_1, "D4_1");
	formGroup(D4_2, "D4_2");
	formGroup(D4_3, "D4_3");
	formGroup(D4_4, "D4_4");

	formGroup(D5_1, "D5_1");
	formGroup(D5_2, "D5_2");
	formGroup(D5_3, "D5_3");
	formGroup(D5_4, "D5_4");

	formGroup(G1_1, "G1_1");
	formGroup(G1_2, "G1_2");
	formGroup(G1_3, "G1_3");

	formGroup(G2_1, "G2_1");
	formGroup(G2_2, "G2_2");
	formGroup(G2_3, "G2_3");

	formGroup(G3_1, "G3_1");
	formGroup(G3_2, "G3_2");
	formGroup(G3_3, "G3_3");

	formGroup(G4_1, "G4_1");
	formGroup(G4_2, "G4_2");
	formGroup(G4_3, "G4_3");

	formGroup(G5_1, "G5_1");
	formGroup(G5_2, "G5_2");
	formGroup(G5_3, "G5_3");

	formGroup(IP_E, "IP_E", neurons_in_moto, 1.1);
	formGroup(MP_E, "MP_E", neurons_in_moto);
	formGroup(EES, "EES", neurons_in_group);

	formGroup(inh_group3, "inh_group3", neurons_in_group);
	formGroup(inh_group4, "inh_group4", neurons_in_group);
	formGroup(inh_group5, "inh_group5", neurons_in_group);

	formGroup(ees_group1, "ees_group1", neurons_in_group);
	formGroup(ees_group2, "ees_group2", neurons_in_group);
	formGroup(ees_group3, "ees_group3", neurons_in_group);
	formGroup(ees_group4, "ees_group4", neurons_in_group);

	for (auto &neuron : C1)
		neuron->addSpikeGenerator(0, speed_to_time, 200);
	for (auto &neuron : C2)
		neuron->addSpikeGenerator(speed_to_time, speed_to_time*2, 200);
	for (auto &neuron : C3)
		neuron->addSpikeGenerator(speed_to_time*2, speed_to_time*3, 200);
	for (auto &neuron : C4)
		neuron->addSpikeGenerator(speed_to_time*3, speed_to_time*5, 200);
	for (auto &neuron : C5)
		neuron->addSpikeGenerator(speed_to_time*5, speed_to_time*6, 200);
	for (auto &neuron : EES)
		neuron->addSpikeGenerator(0, T_sim, EES_FREQ);
	for (auto &neuron : MP_E)
		neuron->addMultimeter();
}


void init_extensor() {
	// - - - - - - - - - - - -
	// CPG (Extensor)
	// - - - - - - - - - - - -
	connectFixedOutDegree(C3, inh_group3, 0.5, 20.0);
	connectFixedOutDegree(C4, inh_group4, 0.5, 20.0);
	connectFixedOutDegree(C5, inh_group5, 0.5, 20.0);

	connectFixedOutDegree(inh_group3, G1_3, 2.8, 20.0);

	connectFixedOutDegree(inh_group4, G1_3, 1.0, 20.0);
	connectFixedOutDegree(inh_group4, G2_3, 1.0, 20.0);

	connectFixedOutDegree(inh_group5, G1_3, 2.0, 20.0);
	connectFixedOutDegree(inh_group5, G2_3, 1.0, 20.0);
	connectFixedOutDegree(inh_group5, G3_3, 1.0, 20.0);
	connectFixedOutDegree(inh_group5, G4_3, 1.0, 20.0);

	/// D1
	// input from sensory
	connectFixedOutDegree(C1, D1_1, 1.0, 1.0); // 3.0
	connectFixedOutDegree(C1, D1_4, 1.0, 1.0); // 3.0
	connectFixedOutDegree(C2, D1_1, 1.0, 1.0); // 3.0
	connectFixedOutDegree(C2, D1_4, 1.0, 1.0); // 3.0
	// input from EES
	connectFixedOutDegree(EES, D1_1, 1.0, 17); // was 27 // 17 Threshold / 7 ST
	connectFixedOutDegree(EES, D1_4, 1.0, 17); // was 27 // 17 Threshold / 7 ST
	// inner connectomes
	connectFixedOutDegree(D1_1, D1_2, 1, 7.0);
	connectFixedOutDegree(D1_1, D1_3, 1, 16.0);
	connectFixedOutDegree(D1_2, D1_1, 1, 7.0);
	connectFixedOutDegree(D1_2, D1_3, 1, 20.0);
	connectFixedOutDegree(D1_3, D1_1, 1, -10.0 * INH_COEF);    // 10
	connectFixedOutDegree(D1_3, D1_2, 1, -10.0 * INH_COEF);    // 10
	connectFixedOutDegree(D1_4, D1_3, 2, -10.0 * INH_COEF);    // 10
	// output to
	connectFixedOutDegree(D1_3, G1_1, 3, 12.5);
	connectFixedOutDegree(D1_3, ees_group1, 1.0, 60); // 30

	// EES group connectomes
	connectFixedOutDegree(ees_group1, ees_group2, 1.0, 20.0);

	/// D2 ///
	// input from Sensory
	connectFixedOutDegree(C2, D2_1, 0.7, 2.0); // 4
	connectFixedOutDegree(C2, D2_4, 0.7, 2.0); // 4
	connectFixedOutDegree(C3, D2_1, 0.7, 2.0); // 4
	connectFixedOutDegree(C3, D2_4, 0.7, 2.0); // 4
	// input from Group (1)
	connectFixedOutDegree(ees_group1, D2_1, 2.0, 1.7); // 5.0
	connectFixedOutDegree(ees_group1, D2_4, 2.0, 1.7); // 5.0
	// inner connectomes
	connectFixedOutDegree(D2_1, D2_2, 1.0, 7.0);
	connectFixedOutDegree(D2_1, D2_3, 1.0, 20.0);
	connectFixedOutDegree(D2_2, D2_1, 1.0, 7.0);
	connectFixedOutDegree(D2_2, D2_3, 1.0, 20.0);
	connectFixedOutDegree(D2_3, D2_1, 1.0, -10.0 * INH_COEF);    // 10
	connectFixedOutDegree(D2_3, D2_2, 1.0, -10.0 * INH_COEF);    // 10
	connectFixedOutDegree(D2_4, D2_3, 2.0, -10.0 * INH_COEF);    // 10
	// output to generator
	connectFixedOutDegree(D2_3, G2_1, 1.0, 30.5);

	// EES group connectomes
	connectFixedOutDegree(ees_group2, ees_group3, 1.0, 20.0);

	/// D3
	// input from Sensory
	connectFixedOutDegree(C3, D3_1, 0.7, 1.9); // 2.5 - 2.2 - 1.3
	connectFixedOutDegree(C3, D3_4, 0.7, 1.9); // 2.5 - 2.2 - 1.3
	connectFixedOutDegree(C4, D3_1, 0.7, 1.9); // 2.5 - 2.2 - 1.3
	connectFixedOutDegree(C4, D3_4, 0.7, 1.9); // 2.5 - 2.2 - 1.3
	// input from Group (2)
	connectFixedOutDegree(ees_group2, D3_1, 1.1, 1.7); // 6
	connectFixedOutDegree(ees_group2, D3_4, 1.1, 1.7); // 6
	// inner connectomes
	connectFixedOutDegree(D3_1, D3_2, 1.0, 7.0);
	connectFixedOutDegree(D3_1, D3_3, 1.0, 20.0);
	connectFixedOutDegree(D3_2, D3_1, 1.0, 7.0);
	connectFixedOutDegree(D3_2, D3_3, 1.0, 20.0);
	connectFixedOutDegree(D3_3, D3_1, 1.0, -10 * INH_COEF);    // 10
	connectFixedOutDegree(D3_3, D3_2, 1.0, -10 * INH_COEF);    // 10
	connectFixedOutDegree(D3_4, D3_3, 2.0, -10 * INH_COEF);    // 10
	// output to generator
	connectFixedOutDegree(D3_3, G3_1, 1, 25.0);
	// suppression of the generator
	connectFixedOutDegree(D3_3, G1_3, 1.5, 30.0);

	// EES group connectomes
	connectFixedOutDegree(ees_group3, ees_group4, 2.0, 20.0);

	/// D4
	// input from Sensory
	connectFixedOutDegree(C4, D4_1, 1.0, 2.0); // 2.5
	connectFixedOutDegree(C4, D4_4, 1.0, 2.0); // 2.5
	connectFixedOutDegree(C5, D4_1, 1.0, 2.0); // 2.5
	connectFixedOutDegree(C5, D4_4, 1.0, 2.0); // 2.5
	// input from Group (3)
	connectFixedOutDegree(ees_group3, D4_1, 1.0, 1.7); // 6.0
	connectFixedOutDegree(ees_group3, D4_4, 1.0, 1.7); // 6.0
	// inner connectomes
	connectFixedOutDegree(D4_1, D4_2, 1.0, 7.0);
	connectFixedOutDegree(D4_1, D4_3, 1.0, 20.0);
	connectFixedOutDegree(D4_2, D4_1, 1.0, 7.0);
	connectFixedOutDegree(D4_2, D4_3, 1.0, 20.0);
	connectFixedOutDegree(D4_3, D4_1, 1.0, -10.0 * INH_COEF); // 10
	connectFixedOutDegree(D4_3, D4_2, 1.0, -10.0 * INH_COEF); // 10
	connectFixedOutDegree(D4_4, D4_3, 2.0, -10.0 * INH_COEF); // 10
	// output to the generator
	connectFixedOutDegree(D4_3, G4_1, 3.0, 20.0);
	// suppression of the generator
	connectFixedOutDegree(D4_3, G2_3, 1.0, 30.0);


	/// D5
	// input from Sensory
	connectFixedOutDegree(C5, D5_1, 1.0, 2.0);
	connectFixedOutDegree(C5, D5_4, 1.0, 2.0);
	// input from Group (4)
	connectFixedOutDegree(ees_group4, D5_1, 1.0, 1.7); // 5.0
	connectFixedOutDegree(ees_group4, D5_4, 1.0, 1.7); // 5.0
	// inner connectomes
	connectFixedOutDegree(D5_1, D5_2, 1.0, 7.0);
	connectFixedOutDegree(D5_1, D5_3, 1.0, 20.0);
	connectFixedOutDegree(D5_2, D5_1, 1.0, 7.0);
	connectFixedOutDegree(D5_2, D5_3, 1.0, 20.0);
	connectFixedOutDegree(D5_3, D5_1, 1.0, -10.0 * INH_COEF);
	connectFixedOutDegree(D5_3, D5_2, 1.0, -10.0 * INH_COEF);
	connectFixedOutDegree(D5_4, D5_3, 2.0, -10.0 * INH_COEF);
	// output to the generator
	connectFixedOutDegree(D5_3, G5_1, 3, 30.0);
	// suppression of the genearator
	connectFixedOutDegree(D5_3, G1_3, 1.0, 30.0);
	connectFixedOutDegree(D5_3, G2_3, 1.0, 30.0);
	connectFixedOutDegree(D5_3, G3_3, 1.0, 30.0);
	connectFixedOutDegree(D5_3, G4_3, 1.0, 30.0);

	/// G1 ///
	// inner connectomes
	connectFixedOutDegree(G1_1, G1_2, 1.0, 10.0);
	connectFixedOutDegree(G1_1, G1_3, 1.0, 10.0);
	connectFixedOutDegree(G1_2, G1_1, 1.0, 10.0);
	connectFixedOutDegree(G1_2, G1_3, 1.0, 10.0);
	connectFixedOutDegree(G1_3, G1_1, 0.7, -50.0 * INH_COEF);
	connectFixedOutDegree(G1_3, G1_2, 0.7, -50.0 * INH_COEF);
	// output to IP_E
	connectFixedOutDegree(G1_1, IP_E, 3, 25.0); // 18 normal
	connectFixedOutDegree(G1_1, IP_E, 3, 25.0); // 18 normal

	/// G2 ///
	// inner connectomes
	connectFixedOutDegree(G2_1, G2_2, 1.0, 10.0);
	connectFixedOutDegree(G2_1, G2_3, 1.0, 20.0);
	connectFixedOutDegree(G2_2, G2_1, 1.0, 10.0);
	connectFixedOutDegree(G2_2, G2_3, 1.0, 20.0);
	connectFixedOutDegree(G2_3, G2_1, 0.5, -30.0 * INH_COEF);
	connectFixedOutDegree(G2_3, G2_2, 0.5, -30.0 * INH_COEF);
	// output to IP_E
	connectFixedOutDegree(G2_1, IP_E, 1.0, 65.0); // 35 normal
	connectFixedOutDegree(G2_2, IP_E, 1.0, 65.0); // 35 normal

	/// G3 ///
	// inner connectomes
	connectFixedOutDegree(G3_1, G3_2, 1.0, 14.0); //12
	connectFixedOutDegree(G3_1, G3_3, 1.0, 20.0);
	connectFixedOutDegree(G3_2, G3_1, 1.0, 12.0);
	connectFixedOutDegree(G3_2, G3_3, 1.0, 20.0);
	connectFixedOutDegree(G3_3, G3_1, 0.5, -30.0 * INH_COEF);
	connectFixedOutDegree(G3_3, G3_2, 0.5, -30.0 * INH_COEF);
	// output to IP_E
	connectFixedOutDegree(G3_1, IP_E, 2, 25.0);   // 20 normal
	connectFixedOutDegree(G3_1, IP_E, 2, 25.0);   // 20 normal

	/// G4 ///
	// inner connectomes
	connectFixedOutDegree(G4_1, G4_2, 1.0, 10.0);
	connectFixedOutDegree(G4_1, G4_3, 1.0, 10.0);
	connectFixedOutDegree(G4_2, G4_1, 1.0, 5.0);
	connectFixedOutDegree(G4_2, G4_3, 1.0, 10.0);
	connectFixedOutDegree(G4_3, G4_1, 0.5, -30.0 * INH_COEF);
	connectFixedOutDegree(G4_3, G4_2, 0.5, -30.0 * INH_COEF);
	// output to IP_E
	connectFixedOutDegree(G4_1, IP_E, 1.0, 17.0);
	connectFixedOutDegree(G4_1, IP_E, 1.0, 17.0);

	/// G5 ///
	// inner connectomes
	connectFixedOutDegree(G5_1, G5_2, 1.0, 7.0);
	connectFixedOutDegree(G5_1, G5_3, 1.0, 10.0);
	connectFixedOutDegree(G5_2, G5_1, 1.0, 7.0);
	connectFixedOutDegree(G5_2, G5_3, 1.0, 10.0);
	connectFixedOutDegree(G5_3, G5_1, 0.5, -30.0 * INH_COEF);
	connectFixedOutDegree(G5_3, G5_2, 0.5, -30.0 * INH_COEF);
	// output to IP_E
	connectFixedOutDegree(G5_1, IP_E, 2, 20.0); // normal 18
	connectFixedOutDegree(G5_1, IP_E, 2, 20.0); // normal 18

	connectFixedOutDegree(IP_E, MP_E, 1, 11); // 14
	connectFixedOutDegree(EES, MP_E, 2, 50); // 50
	//connectFixedOutDegree(Ia, MP_E, 1, 50)
}


void simulate() {
	/// Simulation main loop function
	int id = 0;
	int iter = 0;
	printf("Start sim\n");

	////pragma acc data copy(NEURONS)
	#pragma acc parallel vector_length(200)
	{
		#pragma acc loop gang worker seq
		for (iter = 0; iter < T_sim * steps_in_1ms; iter++) {
			#pragma acc loop vector
			for (id = 0; id < neuron_number; id++) {
				NEURONS[id]->update_state();
			}
		}
	}
	////pragma acc data copyout(NEURONS[0:neuron_number])
}


int main() {
	for(int test_index = 0; test_index < 5; test_index++) {
		srand(time(NULL)); //123
		init_neurons();
		init_groups();
		init_extensor();
		simulate();
		show_results(test_index);

		for (auto &i : NEURONS)
			delete i;

		global_id_start = 0;
	}
	return 0;
}