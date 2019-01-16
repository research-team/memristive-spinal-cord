#include <cstdlib>
#include <stdio.h>
#include <math.h>

#ifdef __JETBRAINS_IDE__
	#define __host__
	#define __device__
	#define __shared__
	#define __constant__
	#define __global__
#endif

#define DEBUG

class Neuron {
private:
	// Object variables
	int id{};						// neuron ID
	int sim_time_step{};			// [step] simulation time in steps

	bool has_multimeter = false;	// if neuron has multimeter
	bool has_spikedetector = false;	// if neuron has spikedetector
	bool has_generator = false;		// if neuron has generator

	float *spike_times{};			// [ms] array of spikes time
	float *membrane_potential{};	// [mV] array of membrane potential values
	float *current_potential{};		// [pA] array of current values

	// Stuff variables
	const float current_tau = 6.0;	// [pA] step of current decreasing/increasing
	const int steps_in_ms = 10;		// [step] how much steps in 1 ms

	// Parameters (const)
	const float C = 100.0f;			// [pF] membrane capacitance
	const float V_rest = -72.0f;	// [mV] resting membrane potential
	const float V_th = -55.0f;		// [mV] spike threshold
	const float k = 0.7f;			// [pA * mV-1] constant ("1/R")
	// determines the time scale of the recovery variable u.
	// The larger the value of a, the quicker the recovery
	const float b = 0.2f;			// [pA * mV-1]  sensitivity of U_m to the sub-threshold fluctuations of the V_m
	const float a = 0.02f;			// [ms-1] time scale of the recovery variable U_m
	const float c = -80.0f;			// [mV] after-spike reset value of V_m
	const float d = 6.0f;			// [pA] after-spike reset value of U_m
	const float V_peak = 35.0f;		// [mV] spike cutoff value
	int ref_time_step{}; 			// [step] refractory period

	// State (changable)
	float V_m = V_rest;		// [mV] membrane potential
	float U_m = 0.0f;		// [pA] membrane potential recovery variable
	float I = 0.0f;			// [pA] input current
	float V_old = V_m;		// [mV] previous value for the V_m
	float step_I = 2.0f;	// [mV] previous value for the V_m
	float U_old = U_m;		// [pA] previous value for the U_m
	float curr_ref_t = 0;	// [step] refractory period timer

public:
	Neuron() = default;

	char* group_name{};
	bool has_spike{};

	void changeIstep(float step_I) {
		this->step_I = step_I;
	}

	void set_has_spike() {
		has_spike = true;
	}

	void addMultimeter() {
		has_multimeter = true;	// set flag that this neuron has the multimeter
		membrane_potential = new float[sim_time_step];	// allocate memory for recording V_m
		current_potential = new float[sim_time_step];	// allocate memory for recording I
	}

	void addSpikedetector() {
		has_spikedetector = true;	// set flag that this neuron has the spikedetector
		spike_times = new float[ sim_time_step / ref_time_step ];	// allocate memory for recording spikes
	}

	void addSpikeGenerator() {
		has_generator = true;
	}

	bool withMultimeter() {
		return has_multimeter;
	}

	bool withSpikedetector() {
		return has_spikedetector;
	}

	float step_to_ms(int step) {
		return step / steps_in_ms;	// convert steps to milliseconds
	}

	int ms_to_step(float ms) {
		return (int) (ms * steps_in_ms);	// convert milliseconds to step
	}

	char* getName() {
		return group_name;
	}

	int getID() {
		return id;
	}

	void set_ref_t(float ref_t){
		ref_time_step = ms_to_step(ref_t);
	}

	int get_ref_t(){
		return ref_time_step;
	}

	void set_sim_time(float t_sim) {
		sim_time_step = ms_to_step(t_sim);
	}

	__device__
	void spike_event(float weight){
		I += weight;
	}

	__device__
	void update(int sim_iter, int thread_id){
		/*
		if (curr_ref_t > 0) {
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
			curr_ref_t = ref_t;
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
		if (curr_ref_t > 0)
			curr_ref_t--;
		 */
		#ifdef DEBUG
			printf("S: %d, T: %d, NRN %p \n", sim_iter, thread_id, this);
		#endif
	}


};
