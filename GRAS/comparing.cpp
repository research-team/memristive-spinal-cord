#include <cstdio>
#include <cmath>
#include <utility>
#include <vector>
#include <ctime>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <unistd.h>

using namespace std;

float SIM_STEP = 0.025;
int nrn_number = 10000;
int SIM_TIME_IN_STEPS = 30000;
int REF_TIME = (int)(3 / SIM_STEP);

template <typename type>
void init_array(type *array, unsigned int size, type value) {
	for(int i = 0; i < size; i++)
		array[i] = value;
}

void FitzHughNagumo_model(){
	float alpha = 0.1;      // (0, 1)
	float epsilon = 0.01;   // > 0
	float gamma = 0.5;      // >= 0

	float V_m[nrn_number];
	float W_m[nrn_number];
	bool has_spike[nrn_number];
	int nrn_ref_time_timer[nrn_number];

	init_array<float>(V_m, nrn_number, 0);
	init_array<float>(W_m, nrn_number, 0);
	init_array<bool>(has_spike, nrn_number, false);
	init_array<int>(nrn_ref_time_timer, nrn_number, 0);

	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++) {
		#pragma omp parallel for num_threads(4)
		for (unsigned int tid = 0; tid < nrn_number; tid++) {
			float V_old = V_m[tid];
			float W_old = W_m[tid];

			// re-calculate V_m and U_m
			V_m[tid] = V_old + SIM_STEP * (V_old * (V_old - alpha) * (1 - V_old) - W_old);
			W_m[tid] = W_old + SIM_STEP * (V_old - gamma * W_old) * epsilon;
		}
	}
}

void Izhikevich_model(){
	const float C = 100;        // [pF] membrane capacitance
	const float V_rest = -72;   // [mV] resting membrane potential
	const float V_thld = -55;   // [mV] spike threshold
	const float k = 0.7;        // [pA * mV-1] constant ("1/R")
	const float a = 0.02;       // [ms-1] time scale of the recovery variable U_m. Higher a, the quicker recovery
	const float b = 0.2;        // [pA * mV-1] sensitivity of U_m to the sub-threshold fluctuations of the V_m
	const float c = -80;        // [mV] after-spike reset value of V_m
	const float d = 6;          // [pA] after-spike reset value of U_m
	const float V_peak = 35;

	float old_v[nrn_number];
	float old_u[nrn_number];
	bool has_spike[nrn_number];
	int nrn_ref_time_timer[nrn_number];

	init_array<float>(old_v, nrn_number, V_rest);
	init_array<float>(old_u, nrn_number, 0);
	init_array<bool>(has_spike, nrn_number, false);
	init_array<int>(nrn_ref_time_timer, nrn_number, 0);

	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++) {
		#pragma omp parallel for num_threads(4)
		for (unsigned int tid = 0; tid < nrn_number; tid++) {
			float V_old = old_v[tid];
			float U_old = old_u[tid];

			// re-calculate V_m and U_m
			float V_m = V_old + SIM_STEP * (k * (V_old - V_rest) * (V_old - V_thld) - U_old) / C;
			float U_m = U_old + SIM_STEP * a * (b * (V_old - V_rest) - U_old);

			// set bottom border of the membrane potential
			if (V_m < c)
				V_m = c;
			// set top border of the membrane potential
			if (V_m >= V_thld)
				V_m = V_peak;
			// check if threshold
			if ((V_m >= V_thld) && (nrn_ref_time_timer[tid] == 0)) {
				// set spike status
				has_spike[tid] = true;
				// redefine V_old and U_old
				old_v[tid] = c;
				old_u[tid] += d;
				// set the refractory period
				nrn_ref_time_timer[tid] = REF_TIME;
			} else {
				// redefine V_old and U_old
				old_v[tid] = V_m;
				old_u[tid] = U_m;
				// update the refractory period timer
				if (nrn_ref_time_timer[tid] > 0)
					nrn_ref_time_timer[tid]--;
			}
		}
	}
}

void SimpleDigitalNeuron_model(){
	float V_m[nrn_number];
	bool has_spike[nrn_number];
	int nrn_ref_time_timer[nrn_number];

	init_array<float>(V_m, nrn_number, 20800);
	init_array<bool>(has_spike, nrn_number, false);
	init_array<int>(nrn_ref_time_timer, nrn_number, 0);

	// main simulation loop
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++) {
		#pragma omp parallel for num_threads(4)
		// updating neurons at each iteration
		for (unsigned int tid = 0; tid < nrn_number; tid++) {
			// (threshold && not in refractory period)
			if ((V_m[tid] >= 65000) && (nrn_ref_time_timer[tid] == 0)) {
				V_m[tid] = 0;
				has_spike[tid] = true;
				nrn_ref_time_timer[tid] = REF_TIME;
			} else {
				// if neuron in the refractory state -- ignore synaptic inputs. Re-calculate membrane potential
				if (nrn_ref_time_timer[tid] > 0 && (V_m[tid] < 20800 || V_m[tid] > 20820)) {
					// if membrane potential > -72
					if (V_m[tid] > 20820)
						V_m[tid] -= 5;
					else
						V_m[tid] += 5;
				}
				// update the refractory period timer
				if (nrn_ref_time_timer[tid] > 0)
					nrn_ref_time_timer[tid]--;
			}
		}
	}
}

int main() {
	//
	auto start = std::chrono::system_clock::now();
	/// models
//	Izhikevich_model();
//	SimpleDigitalNeuron_model();
	FitzHughNagumo_model();

	auto end = std::chrono::system_clock::now();

	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << elapsed.count() << " ms \n";


	return 0;
}