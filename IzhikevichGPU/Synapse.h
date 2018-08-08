#ifndef IZHIKEVICHGPU_SYNAPSE_H
#define IZHIKEVICHGPU_SYNAPSE_H

#include "Neuron.cpp"
#include <openacc.h>

using namespace std;

class Synapse {
public:
	Neuron* origin;
	Neuron* target;
	float weight;
	float syn_delay;
	float timer; // simulation iteration
	float tau_ex = 3.0f;
	const float sim_step = 0.1;

	Synapse(Neuron* origin, Neuron* target, float synaptic_weight, float synaptic_delay){
		this->origin = origin;
		this->target = target;
		this->weight = synaptic_weight;
		ms_to_step(synaptic_delay);
	}

	void ms_to_step(float synaptic_delay) {
		this->timer = synaptic_delay * (1 / sim_step);
	}
};


#endif //IZHIKEVICHGPU_SYNAPSE_H
