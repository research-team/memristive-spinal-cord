#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <string>

#ifdef __JETBRAINS_IDE__
	#define __host__
	#define __device__
	#define __global__
#endif

#define DEBUG

using namespace std;

const int steps_in_ms = 10;		// [step] how much steps in 1 ms
const float ms_in_step = 1.0f / steps_in_ms;	// [step] how much ms in 1 step

int ms_to_step(float ms) {
	return (int) (ms * steps_in_ms);	// convert milliseconds to step
}

class Neuron {
private:
	// Object variables
	int id{};                       // neuron ID
	int sim_time_steps{};           // [step] simulation time in steps

	// Stuff variables
	float *spike_times{};           // [ms] array of spikes time
	float *membrane_potential{};    // [mV] array of membrane potential values
	float *current_potential{};     // [pA] array of current values
	const float step_I = 2.0f;      // [pA[ step of current decreasing/increasing

	// Parameters (const)
	const float C = 100.0f;         // [pF] membrane capacitance
	const float V_rest = -72.0f;    // [mV] resting membrane potential
	const float V_th = -55.0f;      // [mV] spike threshold
	const float k = 0.7f;           // [pA * mV-1] constant ("1/R")
	const float b = 0.2f;           // [pA * mV-1] sensitivity of U_m to the sub-threshold fluctuations of the V_m
	const float a = 0.02f;          // [ms-1] time scale of the recovery variable U_m. Higher a, the quicker recovery
	const float c = -80.0f;         // [mV] after-spike reset value of V_m
	const float d = 6.0f;           // [pA] after-spike reset value of U_m
	const float V_peak = 35.0f;     // [mV] spike cutoff value
	int ref_t_step{};               // [step] refractory period time in steps

	// State (changable)
	float V_m = V_rest;             // [mV] membrane potential
	float U_m = 0.0f;               // [pA] membrane potential recovery variable
	float I = 0.0f;                 // [pA] input current
	float V_old = V_m;              // [mV] previous value for the V_m
	float U_old = U_m;              // [pA] previous value for the U_m
	int ref_t_timer = 0;            // [step] refractory period timer

	// for neurons with generator
	int begin_spiking = 0;          // [step] time step of spike begining
	int end_spiking = 0;            // [step] time step of spike ending
	int spike_each_step = 0;        // [step] send spike each N step

	// with detectors
	int index_spikes_array = 0;     // [step] current index of array of the spikes

	bool has_multimeter = false;    // if neuron has multimeter
	bool has_spikedetector = false; // if neuron has spikedetector
	bool has_generator = false;     // if neuron has generator

public:
	Neuron() = default;

	Neuron(int id, string group_name, float ref_t) {
		this->id = id;
		this->group_name = group_name;
		this->ref_t_step = ms_to_step(ref_t);
	}

	string group_name{};            // contains name of the nuclei group
	bool has_spike{};               // flag if neuron has spike

	__device__
	void set_has_spike(bool spike_status) {
		has_spike = spike_status;
	}

	void add_multimeter(float *mm_data, float* curr_data) {
		has_multimeter = true;	// set flag that this neuron has the multimeter
		membrane_potential = mm_data;
		current_potential = curr_data;
	}

	void add_spikedetector() {
		has_spikedetector = true;	// set flag that this neuron has the spikedetector
	}

	bool with_multimeter() { return has_multimeter; }
	bool with_spikedetector() { return has_spikedetector; }

	__device__
	// convert steps to milliseconds
	float step_to_ms(int step) { return step / steps_in_ms; }

	// convert milliseconds to step
	int ms_to_step(float ms) { return (int) (ms * steps_in_ms); }

	float* get_mm_data() { return membrane_potential; }
	float* get_curr_data() { return current_potential; }

	string get_name() { return group_name; }

	int get_id() { return id; }

	void set_id(int id) { this->id = id; }

	int get_ref_t() { return ref_t_step; }
	void set_ref_t(float ref_t) { ref_t_step = ms_to_step(ref_t); }

	void set_sim_time(float t_sim) { sim_time_steps = ms_to_step(t_sim); }

	__device__
	void spike_event(float weight){
		if (I <= 600 && I >= -600)
			I += weight;
	}

	__host__
	void add_spike_generator(float begin, float end, float hz) {
		begin_spiking = ms_to_step(begin);
		end_spiking = ms_to_step(end);
		spike_each_step = ms_to_step(1.0f / hz * 1000);
		// set flag that this neuron has the multimeter
		has_generator = true;
		// allocate memory for recording spikes
		spike_times = new float[ sim_time_steps / ref_t_step];
	}

	__device__
	void update(int sim_iter, int thread_id){
		if (ref_t_timer > 0) {
			// absolute refractory period : calculate V_m and U_m WITHOUT synaptic weight
			V_m = V_old + ms_in_step * (k * (V_old - V_rest) * (V_old - V_th) - U_old) / C;
			U_m = U_old + ms_in_step * a * (b * (V_old - V_rest) - U_old);

		} else {
			// action potential : calculate V_m and U_m WITH synaptic weight
			// FixMe mult with 200 (! hardcode !)
			V_m = V_old + ms_in_step * (k * (V_old - V_rest) * (V_old - V_th) - U_old + I * 200) / C;
			U_m = U_old + ms_in_step * a * (b * (V_old - V_rest) - U_old);
		}

		if (has_generator &&
			sim_iter >= begin_spiking &&
			sim_iter < end_spiking &&
			(sim_iter % spike_each_step == 0)){
			I = 400.0;
		}

		// save the V_m and I value every iter step if has multimeter
		if (has_multimeter) {
			// ToDo remove at production
			// id was added just for testing
			membrane_potential[sim_iter] = id;// V_m;
			current_potential[sim_iter] = id * 1000; //I;
		}

		if (V_m < c)
			V_m = c;

		// threshold crossing (spike)
		if (V_m >= V_peak) {
			has_spike = true;

			// redefine V_old and U_old
			V_old = c;
			U_old += d;

			// save spike time if has spikedetector
			if (has_spikedetector) {
				spike_times[index_spikes_array] = step_to_ms(sim_iter);
				index_spikes_array++;
			}

			// set the refractory period
			ref_t_timer = ref_t_step;
		} else {
			// redefine V_old and U_old
			V_old = V_m;
			U_old = U_m;
		}

		// FixMe doesn't change the I of generator NEURONS!!!
		// update currents of the neuron
		if (I != 0) {
			// decrease current potential
			if (I > 0) I /= step_I;	// for positive current
			if (I < 0) I /= 1.1;	// for negative current
			// avoid the near value to 0
			if (I > 0 && I <= 1) I = 0;
			if (I <=0 && I >= -1) I = 0;
		}

		// update the refractory period timer
		if (ref_t_timer > 0)
			ref_t_timer--;

		#ifdef DEBUG
			printf("Iter: %d, Thread: %d, NRN (%p) \n", sim_iter, thread_id, this);
		#endif
	}
};
