#ifndef IZHIKEVICHGPU_NEURON_H
#define IZHIKEVICHGPU_NEURON_H

#include <openacc.h>
#include <cstdlib>

using namespace std;

class Neuron {
public:
	/// Indetification and recordable variables
	int id{};
	float *spike_times{nullptr};
	float *membrane_potential{nullptr};

	int iterS = 0;
	int iterM = 0;
	// vector<Neuron*> neighbors;

	/// Stuff variables
	unsigned short simulation_iter = 0;
	float ms_in_1step = 0.01; // ms in one step ALSO: simulation step
	short steps_in_1ms = (short)(1 / ms_in_1step);

	/// Parameters
	float a = 0.02;			// [ms] time scale of the recovery variable U_m
	float b = 0.2;			// [?] sensitivity of U_m to the subthreshold fluctuations of the membrane potential V_m
	float c = -65.0f;		// [mV] after-spike reset value of V_m
	float d = 2.0;			// [?] after-spike reset value of U_m
	float V_th = -55.0f;	// [mV] Spike threshold
	float ref_t = 0; 		// [step] refractory period
	float I_e = 0.0f; 		// [pA] Constant input current
	float V_min = -75.0f;	// [mV] Absolute lower value for the membrane potential

	/// State
	float V_m = -70.0f;		// [mV] membrane potential
	float U_m = 10.0;		// [?] Membrane potential recovery variable
	float I = 0.0f;			// [pA] input current


	Neuron() {
		///Neuron object constructor

		this->ref_t = ms_to_step(3.0f);

		spike_times = (float *)malloc(100 * sizeof(float));
		membrane_potential = (float *)malloc(1000 * sizeof(float));

		#pragma acc enter data create(this)
		#pragma acc update device(this)
	}

	float step_to_ms(int step){
		/// Convert steps to milliseconds
		return step * ms_in_1step;
	}

	short ms_to_step(float ms){
		/// Convert milliseconds to step
		return short(ms * steps_in_1ms);
	}

	void setID(int id){
		this->id = id;
	}

	#pragma acc routine vector
	void update_state() {
		/// Invoked every simulation step, update the neuron state

		// save the membrane potential value every 0.1 ms
		if (simulation_iter % 10 == 0) {
			membrane_potential[iterM++] = V_m;
		}

		const float h = simulation_iter * ms_in_1step;
		float I_syn = 0.0;

		if (ref_t > 0) {
			// if in refractory period
			float V_old = V_m;
			float U_old = U_m;

			V_m += h * ( 0.04 * V_old * V_old + 5.0 * V_old + 140.0 - U_old + I + I_e )  + I_syn;
			U_m += h * a * ( b * V_old - U_old );
		} else {
			// if not in refractory period
			V_m += h * 0.5 * ( 0.04 * V_m * V_m + 5.0 * V_m + 140.0 - U_m + I + I_e + I_syn );
			V_m += h * 0.5 * ( 0.04 * V_m * V_m + 5.0 * V_m + 140.0 - U_m + I + I_e + I_syn );
			U_m += h * a * ( b * V_m - U_m );
		}

		// lower bound of membrane potential
		V_m = ( V_m < V_min ? V_min : V_m );

		// threshold crossing
		if (V_m >= V_th && ref_t <= 0){
			V_m = c;
			U_m = U_m + d;
			ref_t = ms_to_step(3.0);
			// send event
			// TODO send function
			spike_times[iterS++] = step_to_ms(simulation_iter);
		}

		// set new input current
		I = 1.0f;

		++simulation_iter;
		--ref_t;
	};
	void devcopyout(){
		#pragma acc exit data copyout(this)
	}

	float* get_spikes() {
		return spike_times;
	}

	float* get_mm() {
		return membrane_potential;
	}

	~Neuron() {
		#pragma acc exit data delete(this)
		delete spike_times;
		delete membrane_potential;
	}
};

#endif //IZHIKEVICHGPU_NEURON_H
