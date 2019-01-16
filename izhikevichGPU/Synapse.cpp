#include "Neuron.cpp"

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

class Synapse {
public:
	Synapse() = default;

	Neuron* pre_neuron{};		// pre neuron
	Neuron* post_neuron{};		// post neuron

	int syn_delay = 4;			// [steps] synaptic delay. Converts from ms to steps
	int curr_syn_delay{};		// [steps] synaptic delay. Converts from ms to steps
	float weight{};				// [pA] synaptic weight
	int syn_delay_timer = -1;	// [steps] timer of synaptic delay

	__device__
	void update(int sim_iter, int thread_id) {
		// add delay if synapse is not "busy" FixMe check this part of code
		printf("iter %d, T: %d, check %s timer %d \n ", sim_iter, thread_id, pre_neuron->has_spike? "YES" : "NO", syn_delay_timer);
		if (pre_neuron->has_spike and syn_delay_timer == -1) {
			syn_delay_timer = syn_delay;
		} else

		// send spike event because of expiration of synaptic delay
		if (syn_delay_timer == 0) {
			post_neuron->spike_event(weight);
			syn_delay_timer = -1;
		}

		curr_syn_delay += 10;

		// decrement timer
		if (syn_delay_timer > 0)
			--syn_delay_timer;

		#ifdef DEBUG
			printf("S: %d, T: %d, SYN %p \n", sim_iter, thread_id, this);
		#endif
	}
};