#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <string>

#ifdef __JETBRAINS_IDE__
	#define __host__
	#define __device__
	#define __global__
#endif

using namespace std;

const float ms_in_step = 0.1;   // [step] how much ms in 1 step
const int steps_in_ms = (int)(1 / ms_in_step);                    // [step] how much steps in 1 ms

// convert milliseconds to step
__host__
int ms_to_step(float ms) { return (int) (ms * steps_in_ms);}

class Neuron;

class Synapse {
public:
	Neuron* post_neuron{};     // [pointer] post neuron
	float weight{};            // [pA] synaptic weight
	int syn_delay{};           // [step] synaptic delay. Converts from ms to steps
	int syn_delay_timer = -1;  // [step] timer of synaptic delay

	Synapse() = default;
	Synapse(Neuron* post, float syn_delay, float weight) {
		this->post_neuron = post;
		this->syn_delay = ms_to_step(syn_delay);
		this->weight = weight;
	}
};

class Neuron {
private:
	// Stuff variables
	int id{};                       // neuron ID
	float *spike_times{};           // [ms] array of spikes time
	float *membrane_potential{};    // [mV] array of membrane potential values
	float *current_potential{};     // [pA] array of current values
	const float step_I = 2.0f;      // [pA[ step of current decreasing/increasing


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

	bool has_generator = false;     // if neuron has generator
	bool has_multimeter = false;    // if neuron has multimeter
	bool has_spikedetector = false; // if neuron has spikedetector

	int sim_iteration = 0;

public:
	string group_name = "";         // contains name of the nuclei group
	Neuron() = default;
	Neuron(int id, string group_name, float ref_t) {
		this->id = id;
		this->group_name = group_name;
		this->ref_t_step = ms_to_step(ref_t);
	}

	int num_synapses = 0;
	Synapse* synapses[800]{};    // array of synapses

	void add_multimeter(float* mm_data, float* curr_data) {
		has_multimeter = true;      // set flag that this neuron has the multimeter
		membrane_potential = mm_data;
		current_potential = curr_data;
	}

	void add_spikedetector() {
		has_spikedetector = true;	// set flag that this neuron has the spikedetector
	}

	bool with_multimeter() { return has_multimeter; }
	bool with_spikedetector() { return has_spikedetector; }

	// convert steps to milliseconds
	__device__
	float step_to_ms(int step) { return step / steps_in_ms; }

	float* get_mm_data() { return membrane_potential; }
	float* get_curr_data() { return current_potential; }

	string get_name() { return group_name; }

	__device__
	int get_id() { return id; }

	void set_id(int id) { this->id = id; }

	int get_ref_t() { return ref_t_step; }
	void set_ref_t(float ref_t) { ref_t_step = ms_to_step(ref_t); }

	__host__
	void add_spike_generator(float begin, float end, float hz) {
		begin_spiking = ms_to_step(begin);
		end_spiking = ms_to_step(end);
		spike_each_step = ms_to_step(1.0f / hz * 1000);
		// set flag that this neuron has the multimeter
		has_generator = true;
	}

	__device__
	void update(){
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
				sim_iteration >= begin_spiking &&
				sim_iteration < end_spiking &&
				(sim_iteration % spike_each_step == 0)){
			I = 400.0;
		}

		// save the V_m and I value every iter step if has multimeter
		if (has_multimeter) {
			// ToDo remove at production
			// id was added just for testing
			membrane_potential[sim_iteration] = V_m;
			current_potential[sim_iteration] = I;
		}

		if (V_m < c)
			V_m = c;

		// threshold crossing (spike)
		if (V_m >= V_peak) {
			// set timers for all neuron synapses
			for (int i = 0; i < num_synapses; i++) {
				synapses[i]->syn_delay_timer = synapses[i]->syn_delay;
			}

			// redefine V_old and U_old
			V_old = c;
			U_old += d;

			// save spike time if has spikedetector
			if (has_spikedetector) {
				spike_times[index_spikes_array] = step_to_ms(sim_iteration);
				index_spikes_array++;
			}

			// set the refractory period
			ref_t_timer = ref_t_step;
		} else {
			// redefine V_old and U_old
			V_old = V_m;
			U_old = U_m;
		}

		// update timers in all neuron synapses
		for (int i = 0; i < num_synapses; i++) {
			// send spike
			Synapse* synapse = synapses[i];
			Neuron* post_nrn = synapse->post_neuron;
			if (synapse->syn_delay_timer == 0) {
				if (!post_nrn->has_generator &&
						post_nrn->I <= 600 &&
						post_nrn->I >= -600) {
					post_nrn->I += synapse->weight;
				}
				synapse->syn_delay_timer = -1; // set timer to -1 (thats mean no need to update timer in future without spikes)
			}
			// decrement timers
			if (synapse->syn_delay_timer > 0) {
				synapse->syn_delay_timer--;
			}
		}

		// update currents of the neuron
		if (I != 0) {
			// decrease current potential
			if (I > 0) I /= step_I;   // for positive current
			if (I < 0) I /= 1.1;      // for negative current
			// avoid the near value to 0
			if (I > 0 && I <= 1) I = 0;
			if (I <=0 && I >= -1) I = 0;
		}

		// update the refractory period timer
		if (ref_t_timer > 0)
			ref_t_timer--;

		sim_iteration++;
	}

	__host__
	void add_synapses(Synapse* synapses, int syn_size) {
		/// adding the new synapse to the neuron
		Synapse* gpu_syn;

		cudaMalloc(&gpu_syn, sizeof(Synapse) * syn_size);
		cudaMemcpy(gpu_syn, synapses, sizeof(Synapse) * syn_size, cudaMemcpyHostToDevice);

		for(int i = 0; i < syn_size; i++) {
			this->synapses[this->num_synapses++] = &gpu_syn[i];
		}
	}
};
