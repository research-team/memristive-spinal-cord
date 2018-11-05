#include <cstdlib>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <math.h>
#include <ctime>
#include "Neuron.h"

using namespace std;

const unsigned int neuron_number = 4000;
const unsigned int neurons_in_group = 40;
const unsigned int neurons_in_moto = 169;
const unsigned int synapses_number = 20;
const float INH_COEF = 1.0;

// Init the neuron objects
Neuron* NEURONS[neuron_number];

extern const float T_sim = 150.0;
const float ms_in_1step = 0.1f; // ms in one step  simulation step
const short steps_in_1ms = (short) (1 / ms_in_1step);

int global_id_start = 0;

// segment start
Neuron* C1[neurons_in_group];
Neuron* C2[neurons_in_group];
Neuron* C3[neurons_in_group];
Neuron* C4[neurons_in_group];
Neuron* C5[neurons_in_group];

Neuron* group1[neurons_in_group];
Neuron* group2[neurons_in_group];
Neuron* group3[neurons_in_group];
Neuron* group4[neurons_in_group];

Neuron* D1_1[neurons_in_group];
Neuron* D1_2[neurons_in_group];
Neuron* D1_3[neurons_in_group];
Neuron* D1_4[neurons_in_group];

Neuron* D2_1[neurons_in_group];
Neuron* D2_2[neurons_in_group];
Neuron* D2_3[neurons_in_group];
Neuron* D2_4[neurons_in_group];

Neuron* D3_1[neurons_in_group];
Neuron* D3_2[neurons_in_group];
Neuron* D3_3[neurons_in_group];
Neuron* D3_4[neurons_in_group];

Neuron* D4_1[neurons_in_group];
Neuron* D4_2[neurons_in_group];
Neuron* D4_3[neurons_in_group];
Neuron* D4_4[neurons_in_group];

Neuron* D5_1[neurons_in_group];
Neuron* D5_2[neurons_in_group];
Neuron* D5_3[neurons_in_group];
Neuron* D5_4[neurons_in_group];

Neuron* G1_1[neurons_in_group];
Neuron* G1_2[neurons_in_group];
Neuron* G1_3[neurons_in_group];

Neuron* G2_1[neurons_in_group];
Neuron* G2_2[neurons_in_group];
Neuron* G2_3[neurons_in_group];

Neuron* G3_1[neurons_in_group];
Neuron* G3_2[neurons_in_group];
Neuron* G3_3[neurons_in_group];

Neuron* G4_1[neurons_in_group];
Neuron* G4_2[neurons_in_group];
Neuron* G4_3[neurons_in_group];

Neuron* G5_1[neurons_in_group];
Neuron* G5_2[neurons_in_group];
Neuron* G5_3[neurons_in_group];

Neuron* IP_E[neurons_in_moto];
Neuron* IP_F[neurons_in_moto];

Neuron* MP_E[neurons_in_moto];
Neuron* MP_F[neurons_in_moto];

Neuron* R_E[neurons_in_moto];
Neuron* R_F[neurons_in_moto];

Neuron* Ia_E[neurons_in_moto];
Neuron* Ia_F[neurons_in_moto];
Neuron* Ib_E[neurons_in_moto];
Neuron* Ib_F[neurons_in_moto];

Neuron* Extensor[neurons_in_moto];
Neuron* Flexor[neurons_in_moto];

Neuron* Ia[neurons_in_group];
Neuron* EES[neurons_in_group];
Neuron* C_0[neurons_in_group];
Neuron* C_1[neurons_in_group];

void show_results() {
	/// Printing results function
	char cwd[256];
	getcwd(cwd, sizeof(cwd));
	printf("Save results to: %s", cwd);
	string filename = "/sim_results.txt";
	ofstream myfile;
	myfile.open (cwd + filename);
	for (auto &neuron : NEURONS) {
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
		NEURONS[i] = new Neuron(i, 2.0f);
	}

	// additional devices to the NEURONS
	for (auto &neuron : NEURONS)
		neuron->addSpikedetector();
}


void formGroup(Neuron* nrn_group[], char* name, int nrns_in_group = neurons_in_group) {
	int j = 0;
	printf("Formed %s IDs [%d ... %d] = %d\n", name, global_id_start, global_id_start + nrns_in_group - 1, nrns_in_group);
	for (int i = global_id_start; i < global_id_start + nrns_in_group; ++i){
		nrn_group[j] = NEURONS[i];
		NEURONS[i]->name = name;
		///printf("%s %d %d\n", name, i, j);
		///printf("i = %d j = %d, obj: %p, id(%d)\n", i, j, group[j]->getThis(), group[j]->getID());
		j++;
	}

	global_id_start += nrns_in_group;
}


void connectFixedOutDegree(Neuron* a[], Neuron* b[], float syn_delay, float weight, int n = neurons_in_group) {
	srand(123);
	float delta = 0.5;

	for (int i = 0; i < n; i++) {
		for(int j = 0; j < synapses_number; j++) {
			int b_index = rand() % n;
			float syn_delay_dist = float(rand()) / float(RAND_MAX) * 2 * delta + syn_delay - delta;

			a[i]->connectWith(a[i], b[b_index], syn_delay_dist, weight);
			//printf("len = %f \n", sizeof(b) / sizeof(Neuron));
			//printf("Connected: %s [%d] -> %s [%d] (connected %f percent, 1:%d) with syn_delay: %f / weight: %f \n",
			//		a[i]->name,
			//		a[i]->getID(),
			//		b[b_index]->name,
			//		b[b_index]->getID(),
			//	    synapses_number * sizeof(*b) / sizeof(b) * 100,
			//	    synapses_number,
			//		syn_delay_dist,
			//		weight);
		}
	}
	//printf("\n");
}


void init_groups() {
	formGroup(C1, "C1");
	formGroup(C2, "C2");
	formGroup(C3, "C3");
	formGroup(C4, "C4");
	formGroup(C5, "C5");

	formGroup(group1, "group1");
	formGroup(group2, "group2");
	formGroup(group3, "group3");
	formGroup(group4, "group4");

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

	formGroup(IP_E, "IP_E", neurons_in_moto);
	formGroup(IP_F, "IP_F", neurons_in_moto);

	formGroup(MP_E, "MP_E", neurons_in_moto);
	formGroup(MP_F, "MP_F", neurons_in_moto);

	formGroup(R_E, "R_E", neurons_in_moto);
	formGroup(R_F, "R_F", neurons_in_moto);

	formGroup(Ia_E, "Ia_E", neurons_in_moto);
	formGroup(Ia_F, "Ia_F", neurons_in_moto);

	formGroup(Ib_E, "Ib_E", neurons_in_moto);
	formGroup(Ib_F, "Ib_F", neurons_in_moto);

	formGroup(Extensor, "Extensor", neurons_in_moto);
	formGroup(Flexor, "Flexor", neurons_in_moto);

	formGroup(Ia, "Ia", neurons_in_group);
	formGroup(EES, "EES", neurons_in_group);
	formGroup(C_0, "C=0", neurons_in_group);
	formGroup(C_1, "C=1", neurons_in_group);

	for (auto &neuron : C1)
		neuron->addSpikeGenerator(0.0, 25.0, 200);
	for (auto &neuron : C2)
		neuron->addSpikeGenerator(25.0, 50.0, 200);
	for (auto &neuron : C3)
		neuron->addSpikeGenerator(50.0, 75.0, 200);
	for (auto &neuron : C4)
		neuron->addSpikeGenerator(75.0, 125.0, 200);
	for (auto &neuron : C5)
		neuron->addSpikeGenerator(125.0, 150.0, 200);
	for (auto &neuron : EES)
		neuron->addSpikeGenerator(0.0, 150.0, 40);

	for (auto &neuron : NEURONS) {
		neuron->addSpikedetector();
		neuron->addMultimeter();
	}
}


void init_extensor() {
	// - - - - - - - - - - - -
	// CPG (Extensor)
	// - - - - - - - - - - - -

	/// D1
	// input from
	connectFixedOutDegree(C1, D1_1, 1.0, 30.0);
	connectFixedOutDegree(C1, D1_4, 1.0, 15.0);
	connectFixedOutDegree(C2, D1_1, 1.0, 30.0);
	connectFixedOutDegree(C2, D1_4, 1.0, 15.0);
	//FixMe ?????
	connectFixedOutDegree(C3, D1_4, 1.0, 15.0);
	// inner connectomes
	connectFixedOutDegree(D1_1, D1_2, 2.0, 7.0); // 10
	connectFixedOutDegree(D1_1, D1_3, 2.0, 10.0);
	connectFixedOutDegree(D1_2, D1_1, 2.0, 7.0);
	connectFixedOutDegree(D1_2, D1_3, 2.0, 10.0);
	connectFixedOutDegree(D1_3, D1_1, 2.0, -50.0 * INH_COEF); //30
	connectFixedOutDegree(D1_3, D1_2, 2.0, -50.0 * INH_COEF); //30
	connectFixedOutDegree(D1_4, D1_3, 2.0, -50.0 * INH_COEF);
	// output to
	connectFixedOutDegree(D1_3, G1_1, 1.0, 30.0);

///	connectFixedOutDegree(D1_3, group1, 2.0, 5.0);
///	connectFixedOutDegree(group1, D2_1, 2.0, 5.0);

///	connectFixedOutDegree(group1, group2, 2.0, 29.0);

	/// D2
	// input from
	connectFixedOutDegree(C2, D2_1, 1.0, 30.0);
	connectFixedOutDegree(C2, D2_4, 1.0, 15.0);
	connectFixedOutDegree(C3, D2_1, 1.0, 30.0);
	connectFixedOutDegree(C3, D2_4, 1.0, 15.0);
	//FixMe ?????
	connectFixedOutDegree(C4, D2_4, 1.0, 15.0);
	// inner connectomes
	connectFixedOutDegree(D2_1, D2_2, 2.0, 7.0);
	connectFixedOutDegree(D2_1, D2_3, 2.0, 10.0);
	connectFixedOutDegree(D2_2, D2_1, 2.0, 7.0);
	connectFixedOutDegree(D2_2, D2_3, 2.0, 10.0);
	connectFixedOutDegree(D2_3, D2_1, 2.0, -50.0 * INH_COEF);
	connectFixedOutDegree(D2_3, D2_2, 2.0, -50.0 * INH_COEF);
	connectFixedOutDegree(D2_4, D2_3, 2.0, -50.0 * INH_COEF);
	// output to
	connectFixedOutDegree(D2_3, G2_1, 1.0, 30.0); //30

	/// D3
	// input from
	connectFixedOutDegree(C3, D3_1, 1.0, 30.0);
	connectFixedOutDegree(C3, D3_4, 1.0, 15.0);
	connectFixedOutDegree(C4, D3_1, 1.0, 30.0);
	connectFixedOutDegree(C4, D3_4, 1.0, 15.0);
	//FixMe ?????
	connectFixedOutDegree(C5, D3_4, 1.0, 10.0);
///	connectFixedOutDegree(group2, D3_1, 2.0, 29.0);
	// inner connectomes
	connectFixedOutDegree(D3_1, D3_2, 2.0, 7.0);
	connectFixedOutDegree(D3_1, D3_3, 2.0, 10.0);
	connectFixedOutDegree(D3_2, D3_1, 2.0, 7.0);
	connectFixedOutDegree(D3_2, D3_3, 2.0, 10.0);
	connectFixedOutDegree(D3_3, D3_1, 2.0, -50.0 * INH_COEF);
	connectFixedOutDegree(D3_3, D3_2, 2.0, -50.0 * INH_COEF);
	connectFixedOutDegree(D3_4, D3_3, 2.0, -50.0 * INH_COEF);
	// output to
	connectFixedOutDegree(D3_3, G3_1, 3.0, 30.0);
	// suppression
	connectFixedOutDegree(D3_3, G1_3, 1.0, 30.0);
///	connectFixedOutDegree(group2, group3, 2.0, 29.0);

	/// D4
	// input from
	connectFixedOutDegree(C4, D4_1, 1.0, 30.0);
	connectFixedOutDegree(C4, D4_4, 1.0, 15.0);
	connectFixedOutDegree(C5, D4_1, 1.0, 30.0);
	connectFixedOutDegree(C5, D4_4, 1.0, 15.0);
///	connectFixedOutDegree(group3, D4_1, 2.0, 29.0);
	// inner connectomes
	connectFixedOutDegree(D4_1, D4_2, 2.0, 7); // 7
	connectFixedOutDegree(D4_1, D4_3, 2.0, 10.0);
	connectFixedOutDegree(D4_2, D4_1, 2.0, 7); // 7
	connectFixedOutDegree(D4_2, D4_3, 2.0, 10.0);
	connectFixedOutDegree(D4_3, D4_1, 2.0, -30.0 * INH_COEF);
	connectFixedOutDegree(D4_3, D4_2, 2.0, -30.0 * INH_COEF);
	connectFixedOutDegree(D4_4, D4_3, 2.0, -30.0 * INH_COEF);
	// output to
	connectFixedOutDegree(D4_3, G4_1, 3.0, 20.0); // 30
	// suppression
	connectFixedOutDegree(D4_3, G2_3, 1.0, 30.0);

///	connectFixedOutDegree(group3, group4, 2.0, 29.0);

	/// D5
	// input from
	connectFixedOutDegree(C5, D5_1, 1.0, 30.0);
	connectFixedOutDegree(C5, D5_4, 1.0, 15.0);
///	connectFixedOutDegree(group4, D5_1, 2.0, 29.0);
	// inner connectomes
	connectFixedOutDegree(D5_1, D5_2, 2.0, 7.0);
	connectFixedOutDegree(D5_1, D5_3, 2.0, 10.0);
	connectFixedOutDegree(D5_2, D5_1, 2.0, 7.0);
	connectFixedOutDegree(D5_2, D5_3, 2.0, 10.0);
	connectFixedOutDegree(D5_3, D5_1, 2.0, -50.0 * INH_COEF);
	connectFixedOutDegree(D5_3, D5_2, 2.0, -50.0 * INH_COEF);
	connectFixedOutDegree(D5_4, D5_3, 2.0, -50.0 * INH_COEF);
	// output to
	connectFixedOutDegree(D5_3, G5_1, 4.0, 30.0);
	// suppression
	connectFixedOutDegree(D5_3, G4_3, 1.0, 30.0);
	connectFixedOutDegree(D5_3, G3_3, 1.0, 30.0);
	connectFixedOutDegree(D5_3, G2_3, 1.0, 30.0);
	connectFixedOutDegree(D5_3, G1_3, 1.0, 30.0);

	/// G1
	// inner connectomes
	connectFixedOutDegree(G1_1, G1_2, 1.0, 10.0);
	connectFixedOutDegree(G1_1, G1_3, 1.0, 10.0);
	connectFixedOutDegree(G1_2, G1_1, 1.0, 10.0);
	connectFixedOutDegree(G1_2, G1_3, 1.0, 10.0);
	connectFixedOutDegree(G1_3, G1_1, 1.0, -30.0 * INH_COEF);
	connectFixedOutDegree(G1_3, G1_2, 1.0, -30.0 * INH_COEF);
	// output to IP_E
	connectFixedOutDegree(G1_1, IP_E, 1.0, 35.0); //30 135
	connectFixedOutDegree(G1_2, IP_E, 1.0, 35.0); //30 135

	/// G2
	// inner connectomes
	connectFixedOutDegree(G2_1, G2_2, 1.0, 7.0);
	connectFixedOutDegree(G2_1, G2_3, 1.0, 10.0);
	connectFixedOutDegree(G2_2, G2_1, 1.0, 7.0);
	connectFixedOutDegree(G2_2, G2_3, 1.0, 10.0);
	connectFixedOutDegree(G2_3, G2_1, 1.0, -30.0 * INH_COEF);
	connectFixedOutDegree(G2_3, G2_2, 1.0, -30.0 * INH_COEF);
	// output to IP_E
	connectFixedOutDegree(G2_1, IP_E, 1.0, 30.0);
	connectFixedOutDegree(G2_2, IP_E, 1.0, 30.0);

	/// G3
	// inner connectomes
	connectFixedOutDegree(G3_1, G3_2, 2.0, 7.0);
	connectFixedOutDegree(G3_1, G3_3, 2.0, 10.0);
	connectFixedOutDegree(G3_2, G3_1, 2.0, 7.0);
	connectFixedOutDegree(G3_2, G3_3, 2.0, 10.0);
	connectFixedOutDegree(G3_3, G3_1, 1.0, -30.0 * INH_COEF);
	connectFixedOutDegree(G3_3, G3_2, 1.0, -30.0 * INH_COEF);
	// output to IP_E
	connectFixedOutDegree(G3_1, IP_E, 1.0, 30.0);
	connectFixedOutDegree(G3_2, IP_E, 1.0, 30.0);

	/// G4
	// inner connectomes
	connectFixedOutDegree(G4_1, G4_2, 2.0, 5.0);
	connectFixedOutDegree(G4_1, G4_3, 2.0, 10.0);
	connectFixedOutDegree(G4_2, G4_1, 2.0, 5.0);
	connectFixedOutDegree(G4_2, G4_3, 2.0, 10.0);
	connectFixedOutDegree(G4_3, G4_1, 1.0, -30.0 * INH_COEF);
	connectFixedOutDegree(G4_3, G4_2, 1.0, -30.0 * INH_COEF);
	// output to IP_E
	connectFixedOutDegree(G4_1, IP_E, 3.0, 30.0);
	connectFixedOutDegree(G4_2, IP_E, 3.0, 30.0);

	/// G5
	// inner connectomes
	connectFixedOutDegree(G5_1, G5_2, 2.0, 7.0);
	connectFixedOutDegree(G5_1, G5_3, 2.0, 10.0);
	connectFixedOutDegree(G5_2, G5_1, 2.0, 7.0);
	connectFixedOutDegree(G5_2, G5_3, 2.0, 10.0);
	connectFixedOutDegree(G5_3, G5_1, 1.0, -30.0 * INH_COEF);
	connectFixedOutDegree(G5_3, G5_2, 1.0, -30.0 * INH_COEF);
	// output to IP_E
	connectFixedOutDegree(G5_1, IP_E, 5.0, 30.0);
	connectFixedOutDegree(G5_2, IP_E, 5.0, 30.0);
}


void init_ref_arc() {
	// - - - - - - - - - - - -
	// Reflectory arc
	// - - - - - - - - - - - -
///	connectFixedOutDegree(EES, D1_1, 2.0, 20.0);
	connectFixedOutDegree(EES, Ia, 1.0, 20.0);

	connectFixedOutDegree(C1, C_1, 1.0, 20.0);
	connectFixedOutDegree(C2, C_1, 1.0, 20.0);
	connectFixedOutDegree(C3, C_1, 1.0, 20.0);
	connectFixedOutDegree(C4, C_1, 1.0, 20.0);
	connectFixedOutDegree(C5, C_1, 1.0, 20.0);

	connectFixedOutDegree(C_0, IP_E, 2.0, -20.0 * INH_COEF);
	connectFixedOutDegree(C_1, IP_F, 2.0, -20.0 * INH_COEF);

	connectFixedOutDegree(IP_E, MP_E, 2.0, 20.0);
	connectFixedOutDegree(IP_E, Ia_E, 2.0, 20.0);

	connectFixedOutDegree(MP_E, Extensor, 2.0, 20.0);
	connectFixedOutDegree(MP_E, R_E, 2.0, 20.0);

	connectFixedOutDegree(IP_F, MP_F, 2.0, 20.0);
	connectFixedOutDegree(IP_F, Ia_F, 2.0, 20.0);

	connectFixedOutDegree(MP_F, Flexor, 2.0, 20.0);
	connectFixedOutDegree(MP_F, R_F, 2.0, 20.0);

	connectFixedOutDegree(Ib_F, Ib_E, 2.0, -20.0 * INH_COEF);
	connectFixedOutDegree(Ib_F, MP_F, 2.0, -20.0 * INH_COEF);
	connectFixedOutDegree(Ib_E, Ib_F, 2.0, -20.0 * INH_COEF);
	connectFixedOutDegree(Ib_E, MP_E, 2.0, -5.0 * INH_COEF);

	connectFixedOutDegree(Ia_F, Ia_E, 2.0, -20.0 * INH_COEF);
	connectFixedOutDegree(Ia_F, MP_E, 2.0, -5.0 * INH_COEF);
	connectFixedOutDegree(Ia_E, Ia_F, 2.0, -20.0 * INH_COEF);
	connectFixedOutDegree(Ia_E, MP_F, 2.0, -20.0 * INH_COEF);

	connectFixedOutDegree(R_F, R_E, 2.0, -20.0 * INH_COEF);
	connectFixedOutDegree(R_F, Ia_F, 2.0, -20.0 * INH_COEF);
	connectFixedOutDegree(R_F, MP_F, 2.0, -20.0 * INH_COEF);

	connectFixedOutDegree(R_E, R_F, 2.0, -20.0 * INH_COEF);
	connectFixedOutDegree(R_E, Ia_E, 2.0, -20.0 * INH_COEF);
	connectFixedOutDegree(R_E, MP_E, 2.0, -5.0 * INH_COEF);

	connectFixedOutDegree(Ia, MP_F, 2.0, 20.0);
	connectFixedOutDegree(Ia, Ia_F, 2.0, 20.0);
	connectFixedOutDegree(Ia, Ib_F, 2.0, 20.0);

	connectFixedOutDegree(Ia, MP_E, 1.0, 20.0);
	connectFixedOutDegree(Ia, Ia_E, 2.0, 20.0);
	connectFixedOutDegree(Ia, Ib_E, 2.0, 20.0);
}


void simulate() {
	/// Simulation main loop function
	int id = 0;
	int iter = 0;
	printf("Start sim\n");

	// The main nested loop
	clock_t t = clock();

	#pragma acc data copy(neurons)
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
	printf ("Time: %f s\n", (float)t / CLOCKS_PER_SEC);
}


int main() {
	init_neurons();
	init_groups();
	init_extensor();
	//init_flexor();
	init_ref_arc();
	simulate();
	show_results();
	return 0;
}