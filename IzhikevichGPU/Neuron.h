#ifndef IZHIKEVICHGPU_NEURON_H
#define IZHIKEVICHGPU_NEURON_H

#include <openacc.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>

using namespace std;

class Neuron {
private:
	/// Idetification and recordable variables
	int id{};
	float *spike_times;
	float *membrane_potential;

	int mm_record_step;

	int iterS = 0;
	int iterM = 0;
	// vector<Neuron*> neighbors;

	unsigned short simulation_iter = 0;
	/// Stuff variables
	float T_sim = 1000.0;
	float ms_in_1step = 0.1f; //0.01f; // ms in one step ALSO: simulation step
	short steps_in_1ms = (short)(1 / ms_in_1step);

	/// Parameters
	float C = 100.0f;			// [pF] membrane capacitance
	float V_rest = -60.0f;		// [mV] resting membrane potential
	float V_th = -40.0f;	// [mV] Spike threshold
	float k = 0.7f;
	float a = 0.03f;			// [ms-1] time scale of the recovery variable U_m
	float b = -2.0f;			// [pA * mV-1]  sensitivity of U_m to the subthreshold fluctuations of the membrane potential V_m
	float c = -50.0f;		// [mV] after-spike reset value of V_m
	float d = 100.0f;			// [pA]  after-spike reset value of U_m
	float V_peak = 35.0f;      // [mV] spike cutoff value

	int ref_t = 0; 			// [step] refractory period

	/// State
	float V_m = V_rest;		// [mV] membrane potential
	float U_m = 0.0f;		// [pA] Membrane potential recovery variable
	float I = 0.0f;			// [pA] input current

	float V_old = V_m;
	float U_old = U_m;

	float h{};// simulation_iter * ms_in_1step;


public:
	Neuron(int id, float I) {
		this->id = id;
		this->I = I;
		///Neuron object constructor
		this->ref_t = ms_to_step(3.0f);
		mm_record_step = ms_to_step(0.1f);

		spike_times = new float[ ms_to_step(T_sim) / ref_t ];
		membrane_potential = new float[ (ms_to_step(T_sim) / mm_record_step)];
	}

	/// Convert steps to milliseconds
	float step_to_ms(int step){ return step * ms_in_1step; }
	/// Convert milliseconds to step
	int ms_to_step(float ms){ return (int)(ms * steps_in_1ms); }
	int getID(){ return this->id; }
	float* get_spikes() { return spike_times; }
	float* get_mm() { return membrane_potential; }
	int get_mm_size() { return (ms_to_step(T_sim) / mm_record_step); }

	unsigned int getSimIter(){ return simulation_iter;}

	Neuron* getThis(){ return this; }

	//#pragma acc routine vector
	void update_state() {
		/// Invoked every simulation step, update the neuron state
		V_m = V_old + ms_in_1step * (k * (V_old - V_rest) * (V_old - V_th) - U_old + I) / C;
		U_m = U_old + ms_in_1step * a * (b * (V_old - V_rest) - U_old);

		// save the membrane potential value every 0.1 ms
		if (simulation_iter % mm_record_step == 0) {
			membrane_potential[iterM++] = V_m;
		}

		// threshold crossing
		if (V_m >= V_peak){
			V_old = c;
			U_old += d;
			spike_times[iterS++] = step_to_ms(simulation_iter);
		} else {
			V_old = V_m;
			U_old = U_m;
		}
		simulation_iter++;
	}

	~Neuron() {
		#pragma acc exit data delete(this)
		delete spike_times;
		delete membrane_potential;
	}
};

#endif //IZHIKEVICHGPU_NEURON_H
