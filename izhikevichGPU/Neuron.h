#ifndef IZHIKEVICHGPU_NEURON_H
#define IZHIKEVICHGPU_NEURON_H

//#include <openacc.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <utility>

using namespace std;

extern const float T_sim;

class Neuron {
	/// Synapse structure
	struct Synapse {
		Neuron* post_neuron{}; // post neuron ID
		int syn_delay{};		 // [steps] synaptic delay. Converts from ms to steps
		float weight{};		 // [pA] synaptic weight
		int timer{};			 // [steps] changeable value of synaptic delay
		float changing_weight{};

		Synapse() = default;
		Synapse(Neuron* post, float delay, float w) {
			this-> post_neuron = post;
			this-> syn_delay = ms_to_step(delay);
			this-> weight = w;
			this-> timer = -1;
			this-> changing_weight = w;
		}
	};

private:
	/// Object variables
	int id{};								// neuron ID
	float *spike_times{};					// array of spike time
	float *membrane_potential{};			// array of membrane potential values
	float *I_potential{};					// array of I

	int mm_record_step = ms_to_step(0.1f); 	// step of recording membrane potential
	int iterSpikesArray = 0;				// current index of array of the spikes
	int iterVoltageArray = 0; 				// current index of array of the V_m
	int simulation_iter = 0;		        // current simulation step
	bool hasMultimeter = false;				// if neuron has multimeter
	bool hasSpikedetector = false;			// if neuron has spikedetector
	bool hasGenerator = false;				// if neuron has generator
	int begin_spiking = 0;
	int end_spiking = 0;
	int spike_each_step = 0;
	/// Stuff variables
	const float I_tau = 6.0f;				                            // step of I decreasing/increasing
	static constexpr float ms_in_1step = 0.1f;	                        // how much milliseconds in 1 step
	static const int steps_in_1ms = static_cast<int>(1 / ms_in_1step);  // how much steps in 1 ms

	/// Parameters (const)
	const float C = 100.0f;			// [pF] membrane capacitance
	const float V_rest = -72.0f;	// [mV] resting membrane potential
	const float V_th = -55.0f;		// [mV] spike threshold
	const float k = 0.7f;			// [pA * mV-1] constant ("1/R")
	const float a = 0.02f;			//  0.03 [ms-1] time scale of the recovery variable U_m
	//determines the time scale of the recovery variable u. The larger the value of a, the quicker the recovery
	const float b = 0.2f;			// -2 [pA * mV-1]  sensitivity of U_m to the sub-threshold fluctuations of the V_m
	/* describes the sensitivity of the recovery variable U to the sub-threshold fluctuations
	 * of the membrane potential V. Larger values of b couple U and  V more strongly,
	 * resulting in possible sub-threshold oscillations and low-threshold spiking dynamics
	 */
	const float c = -80.0f;			// [mV] after-spike reset value of V_m
	const float d = 6.0f;			// 100 [pA] after-spike reset value of U_m
	/*
	 * describes the after-spike reset of the recovery variable ucaused by slow high-threshold Na+
	 * and K+conductances.
	 */
	const float V_peak = 35.0f;		// [mV] spike cutoff value
	int ref_t{}; 					// [step] refractory period

	/// State (changable)
	float V_m = V_rest;		// [mV] membrane potential
	float U_m = 0.0f;		// [pA] membrane potential recovery variable
	float I = 0.0f;			// [pA] input current
	float V_old = V_m;		// [mV] previous value for the V_m
	float step_I = 2.0f;		// [mV] previous value for the V_m
	float U_old = U_m;		// [pA] previous value for the U_m
	float current_ref_t = 0;

public:
	Synapse* synapses = new Synapse[400];	// array of synapses
	int num_synapses{0};                    // current number of synapses (neighbors)
	char* name{};

	Neuron() = default;

	Neuron(int id, float ref_t) {
		this->id = id;
		this->ref_t = ms_to_step(ref_t);
	}



	void changeIstep(float step_I) {
		this->step_I = step_I;
	}

	void addMultimeter() {
		// set flag that this neuron has the multimeter
		hasMultimeter = true;
		// allocate memory for recording V_m
		membrane_potential = new float[ (ms_to_step(T_sim) / mm_record_step) ];
		// allocate memory for recording I
		I_potential = new float[ (ms_to_step(T_sim) / mm_record_step) ];

	}

	void addSpikedetector() {
		// set flag that this neuron has the multimeter
		hasSpikedetector = true;
		// allocate memory for recording spikes
		spike_times = new float[ ms_to_step(T_sim) / this->ref_t ];
	}

	void addSpikeGenerator(float begin, float end, float hz) {
		this->begin_spiking = ms_to_step(begin);
		this->end_spiking = ms_to_step(end);
		this->spike_each_step = ms_to_step(1.0f / hz * 1000);
		// set flag that this neuron has the multimeter
		hasGenerator = true;
		// allocate memory for recording spikes
		spike_times = new float[ ms_to_step(T_sim) / this->ref_t ];
	}

	void addSpikeGenerator() {
		hasGenerator = true;
	}

	bool withMultimeter() {
		return hasMultimeter;
	}

	bool withSpikedetector() {
		return hasSpikedetector;
	}

	float step_to_ms(int step) {
		return step * ms_in_1step;  // convert steps to milliseconds
	}

	static int ms_to_step(float ms) {
		return (int) (ms * steps_in_1ms);   // convert milliseconds to step
	}

	Neuron* getThis() {
		return this;
	}

	char* getName() {
		return this->name;
	}

	int getID() {
		return this->id;
	}

	float* getSpikes() {
		return spike_times;
	}

	float* getVoltage() {
		return membrane_potential;
	}

	float* getCurrents() {
		return I_potential;
	}

	int getVoltageArraySize() {
		return (ms_to_step(T_sim) / mm_record_step);
	}

	int getSimulationIter() {
		return simulation_iter;
	}

	int getIterSpikesArray() {
		return iterSpikesArray;
	}

	/// Invoked every simulation step, update the neuron state
	//#pragma acc routine vector
	void update_state() {
		if (current_ref_t > 0) {
			// calculate V_m and U_m WITHOUT synaptic weight
			// (absolute refractory period)
			V_m = V_old + ms_in_1step * (k * (V_old - V_rest) * (V_old - V_th) - U_old) / C;
			U_m = U_old + ms_in_1step * a * (b * (V_old - V_rest) - U_old);

		} else {
			// calculate V_m and U_m WITH synaptic weight
			// (action potential)
			V_m = V_old + ms_in_1step * (k * (V_old - V_rest) * (V_old - V_th) - U_old + I * 200) / C;
			U_m = U_old + ms_in_1step * a * (b * (V_old - V_rest) - U_old);
		}

		if (hasGenerator &&
			simulation_iter >= begin_spiking &&
			simulation_iter < end_spiking && (simulation_iter % spike_each_step == 0)){
			I = 400.0;
		}

		// save the V_m and I value every mm_record_step if hasMultimeter
		if (hasMultimeter && simulation_iter % mm_record_step == 0) {
			membrane_potential[iterVoltageArray] = V_m; //V_m
			I_potential[iterVoltageArray] = I;
			iterVoltageArray++;
		}

		if (V_m < c)
			V_m = c;

		// threshold crossing (spike)
		if (V_m >= V_peak) {
			// set timers for all neuron synapses
			for (int i = 0; i < num_synapses; i++) {
				synapses[i].timer = synapses[i].syn_delay;
			}

			// redefine V_old and U_old
			V_old = c;
			U_old += d;

			// save spike time if hasSpikedetector
			if (hasSpikedetector) {
				spike_times[iterSpikesArray] = simulation_iter * ms_in_1step; // from step to ms
				iterSpikesArray++;
			}
			// set the refractory period
			current_ref_t = ref_t;
		} else {
			// redefine V_old and U_old
			V_old = V_m;
			U_old = U_m;
		}

		// update timers in all neuron synapses
		for (int i = 0; i < num_synapses; i++ ) {
			Synapse* syn = synapses + i;
			// send spike
			if (syn->timer == 0) {
				if (!syn->post_neuron->hasGenerator && syn->post_neuron->I <= 600 && syn->post_neuron->I >= -600) {
					syn->post_neuron->I += syn->weight;
				}
				syn->timer = -1; // set timer to -1 (thats mean no need to update timer in future without spikes)
			}
			// decrement timers
			if (syn->timer > 0) {
				syn->timer--;
			}
		}

		// update I (currents) of the neuron
		// doesn't change the I of generator NEURONS!!!
		if (I != 0) {
			if (I > 0)
				I /= step_I;	// decrease I = 2
			if (I < 0)
				I /= 1.1;	// decrease I
			if (I > 0 && I <= 1)
				I = 0;
			if (I <=0 && I >= -1)	// avoid the near value to 0)
				I = 0;
		}

		// update the refractory period timer
		if (current_ref_t > 0)
			current_ref_t--;

		// update the simulation iteration
		simulation_iter++;
	}

	void connectWith(Neuron* pre_neuron, Neuron* post_neuron, float syn_delay, float weight) {
		/// adding the new synapse to the neuron
		Synapse* syn = new Synapse(post_neuron, syn_delay, weight);
		pre_neuron->synapses[pre_neuron->num_synapses++] = *syn;
	}


	~Neuron() {
		//#pragma acc exit data delete(this)
		if (hasSpikedetector)
			delete[] spike_times;

		if (hasMultimeter) {
			delete[] membrane_potential;
			delete[] I_potential;
		}
		delete[] synapses;
	}
};

#endif //IZHIKEVICHGPU_NEURON_H