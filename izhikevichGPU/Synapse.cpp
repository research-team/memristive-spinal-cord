#include "Neuron.cpp"

#include <cstdlib>
#include <stdio.h>
#include <math.h>


class Synapse {
public:
	Synapse() = default;

	Neuron* pre_neuron{};		// [pointer] pre neuron
	Neuron* post_neuron{};		// [pointer] post neuron

	int syn_delay = 4;			// [step] synaptic delay. Converts from ms to steps
	int curr_syn_delay{};		// [step] synaptic delay. Converts from ms to steps
	float weight{};				// [pA] synaptic weight
	int syn_delay_timer = -1;	// [step] timer of synaptic delay

	__device__
	void update(int sim_iter, int thread_id) {
		// add delay if synapse is not "busy" FixMe check this part of code
		printf("iter %d, T: %d, check %s timer %d \n ", sim_iter, thread_id, pre_neuron->has_spike? "YES" : "NO", syn_delay_timer);
		if (pre_neuron->has_spike and syn_delay_timer == -1) {
			syn_delay_timer = syn_delay;
		}

		// send spike event because of expiration of synaptic delay
		if (syn_delay_timer == 0) {
			post_neuron->spike_event(weight);
			syn_delay_timer = -1;
		}


		/*
		 * // send spike
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
		 */

		curr_syn_delay += 10;

		// decrement timer
		if (syn_delay_timer > 0)
			--syn_delay_timer;

		#ifdef DEBUG
			printf("S: %d, T: %d, SYN %p \n", sim_iter, thread_id, this);
		#endif
	}
};