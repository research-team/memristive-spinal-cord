#ifndef IZHIKEVICHGPU_NEURON_H
#define IZHIKEVICHGPU_NEURON_H

#include <openacc.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <utility>

using namespace std;
class Neuron {
	/// Synapse structure
	struct Synapse {
		Neuron* post_neuron; // post neuron ID
		int syn_delay;		 // [steps] synaptic delay. Converts from ms to steps
		float weight;		 // [pA] synaptic weight
		int timer;			 // [steps] changeable value of synaptic delay

		Synapse() = default;
		Synapse(Neuron* post_neuron, float syn_delay, float weight) {
			this-> post_neuron = post_neuron;
			this-> syn_delay = ms_to_step(syn_delay);
			this-> weight = weight;
			this-> timer = -1;
		}
	};
private:
	/// Idetification and recordable variables
	int id{};								// neuron ID
	float *spike_times;						// array of spike time
	float *membrane_potential;				// array of membrane potential values
	Synapse* neighbors = new Synapse[50];	// array of synapses
	int n_neighbors{0};						// current number of synapses (neighbors)
	int neighbors_capacity = 50;			// length of the array of a synapses
	int mm_record_step = ms_to_step(0.1f); 	// step of recording membrane potential
	int iterSArray = 0;						// current index of array of the spikes
	int iterVArray = 0; 					// current index of array of the V_m
	unsigned short simulation_iter = 0;		// current simulation step
	bool generatorFlag = false;

	/// Stuff variables
	float T_sim = 1000.0;	// simulation time
	static constexpr float ms_in_1step = 0.1f;	// how much milliseconds in 1 step
	static const short steps_in_1ms = (short)(1 / ms_in_1step); // how much steps in 1 ms

	/// Parameters
	float C = 100.0f;		// [pF] membrane capacitance
	float V_rest = -60.0f;	// [mV] resting membrane potential
	float V_th = -40.0f;	// [mV] spike threshold
	float k = 0.7f;			// [pA * mV-1] constant ("1/R")
	float a = 0.03f;		// [ms-1] time scale of the recovery variable U_m
	float b = -2.0f;		// [pA * mV-1]  sensitivity of U_m to the sub-threshold fluctuations of the V_m
	float c = -50.0f;		// [mV] after-spike reset value of V_m
	float d = 100.0f;		// [pA] after-spike reset value of U_m
	float V_peak = 35.0f;	// [mV] spike cutoff value
	int ref_t = 0; 			// [step] refractory period

	/// State
	float V_m = V_rest;		// [mV] membrane potential
	float U_m = 0.0f;		// [pA] membrane potential recovery variable
	float I = 0.0f;			// [pA] input current

	float V_old = V_m;		// [mV] previous value for the V_m
	float U_old = U_m;		// [pA] previous value for the U_m

public:
	///Neuron object constructor
	Neuron(int id, float ref_t) {
		this->id = id;
		this->ref_t = ms_to_step(ref_t);
		spike_times = new float[ ms_to_step(T_sim) / this->ref_t ];
		membrane_potential = new float[ (ms_to_step(T_sim) / mm_record_step) ];
	}

	/// Convert steps to milliseconds
	float step_to_ms (int step) { return step * ms_in_1step; }
	/// Convert milliseconds to step
	static int ms_to_step(float ms){ return (int)(ms * steps_in_1ms); }
	/// Stuff getters/setters
	Neuron* getThis(){ return this; }
	int getID(){ return this->id; }
	float* get_spikes() { return spike_times; }
	float* get_mm() { return membrane_potential; }
	int get_mm_size() { return (ms_to_step(T_sim) / mm_record_step); }
	unsigned int getSimIter(){ return simulation_iter;}

	void makeGenerator(float I) {
		this->I = I;
		this->generatorFlag = true;
	}

	//#pragma acc routine vector
	/// Invoked every simulation step, update the neuron state
	void update_state() {
		V_m = V_old + ms_in_1step * (k * (V_old - V_rest) * (V_old - V_th) - U_old + I) / C;
		U_m = U_old + ms_in_1step * a * (b * (V_old - V_rest) - U_old);

		// save the membrane potential value every mm_record_step
		if (simulation_iter % mm_record_step == 0) {
			membrane_potential[iterVArray] = V_m;
			iterVArray++;
		}

		// threshold crossing
		if (V_m >= V_peak) {
			// set timers
			for (int i = 0; i < n_neighbors; i++ )
				neighbors[i].timer =neighbors[i].syn_delay;

			V_old = c;
			U_old += d;
			spike_times[iterSArray] = step_to_ms(simulation_iter);
			iterSArray++;
		} else {
			V_old = V_m;
			U_old = U_m;
		}

		// update delay timers on synapses
		//struct Synapse* ptr = neighbors;
		//struct Synapse* endPtr = neighbors + sizeof(neighbors) / sizeof(neighbors[0]);

		for (int i = 0; i < n_neighbors; i++ ){
			if (neighbors[i].timer > 0)
				neighbors[i].timer -= 1;	// decrement timer in each neighbor
			// "send spike" -- change the I of the post neuron by weight value
			if (neighbors[i].timer == 0) {
				neighbors[i].timer = -1;
				neighbors[i].post_neuron->I += neighbors[i].weight;
			}
		}

		if(this->I > 0 && !generatorFlag)
			this->I -= 2;

		simulation_iter++;
	}

	void add_neighbor(Neuron* post_neuron, float syn_delay, float weight) {
		/// adding the new synapse to the neuron
	    Synapse* syn = new Synapse(post_neuron, syn_delay, weight);
	    neighbors[n_neighbors] = *syn;

		n_neighbors++;

	    // increase array size if near to the limit
	    if (n_neighbors == neighbors_capacity) {
	        int new_neighbors_capacity = static_cast<int>(neighbors_capacity * 1.5 + 1);
	        Synapse* new_neighbors = new Synapse[new_neighbors_capacity];
			// copying
            for (int i = 0; i < n_neighbors; ++i) {
                new_neighbors[i] = neighbors[i];
            }
			// change the links
            neighbors = new_neighbors;
	    	neighbors_capacity = new_neighbors_capacity;
            delete[] new_neighbors;
	    }
	}

	~Neuron() {
		#pragma acc exit data delete(this)
		delete[] spike_times;
		delete[] membrane_potential;
		delete[] neighbors;
	}
};

#endif //IZHIKEVICHGPU_NEURON_H
