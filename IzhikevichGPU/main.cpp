#include <cstdlib>
#include <iostream>
#include <openacc.h>
#include <vector>

using namespace std;

class Neuron {
	public:
		short id;
		vector<float> spike_times;
		vector<float> membrane_potential;
		vector<Neuron*> neighbors;
		/*
		a		[ms]	time scale of the recovery variable u
		b		[?]		sensitivity of u to the subthreshold fluctuations of the membrane potential v
		c		[mV]	after-spike reset value of v
		d		[?]		after-spike reset of u
		V_m		[mV]	initial membrane potential
		U_m		[?]		ToDO ????
		V_th	[mV]	threshold variable
		ref_t	[ms]	refractory period time
		*/

		const float sim_step = 0.1; // ms in step
		float a = 0.02;
		float b = 0.2;
		float c = -65.0f;
		float d = 2.0;
		float V_m = -70.0f;
		float U_m = 10.0;
		float V_th = -55.0f;
		float ref_t = 3.0;
		float I_stim;
		unsigned short simulation_iter = 0;

		Neuron() {
			//nData = _nData;
			//data = new int[nData];
			//#pragma acc enter data create(this)
			//// The following pragma copies the all the data in the class to the device
			//#pragma acc update device(this)
			//// Alternatively, the following pragma copies just nData
			////#pragma acc update device(nData)
			//#pragma acc enter data create(data[0:nData])
		}
		//#pragma acc routine worker
		void update_state() {
			// Save the membrane potential value
			if (simulation_iter % 100 == 0) {
				membrane_potential.push_back(V_m);
			}
			const float h = simulation_iter * sim_step;

			// If in refractory period
			// todo check ref and non-ref
			if (ref_t > 0) {
				/*float V_old = V_m;
				float U_old = U_m;

				V_m += h * ( 0.04 * V_old * V_old + 5.0 * V_old + 140.0 - U_old + S_.I_ + P_.I_e_ )  + I_syn;
				U_m += h * a * ( b * V_old - U_old );*/


			// If not in refractory period
			} else {
				V_m += 5;
				// V_m += h * 0.5 * ( 0.04 * V_m * V_m + 5.0 * V_m + 140.0 - U_m + S_.I_ + P_.I_e_ + I_syn );
				// V_m += h * 0.5 * ( 0.04 * V_m * V_m + 5.0 * V_m + 140.0 - U_m + S_.I_ + P_.I_e_ + I_syn );
				// U_m += h * a * ( b * V_m - U_m );
			}

			// Spike
			if (V_m >= V_th){
				// send event
				// TODO send function
				spike_times.push_back( simulation_iter * sim_step );
				V_m = -70;
			}

			simulation_iter++;
		};

		~Neuron() {
			//delete [] data;
			//#pragma acc exit data delete(data[0:nData])
			//#pragma acc exit data delete(this)
		}
};


class Synapse {
	public:
		Neuron* origin;
		Neuron* target;
		float weight;
		float syn_delay;
		float timer; // simulation iteration
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


int main(int argc, char *argv[]) {
	const float sim_step = 0.1;	// sim_step

	vector<Neuron*> neurons;

	unsigned int network_size = 2;
	neurons.reserve(network_size);

	// Neuron initialization
	for (int i=0; i < network_size; i++){
		neurons.push_back(new Neuron());
	}

	// Synapse initialization
	new Synapse(neurons[0], neurons[1], 10.0, 1.0);

	float T_sim = 100.0;
	int step_in_ms = 10;

	// Start simulation
	for(int iter = 0; iter < step_in_ms * T_sim; iter++ ){
		//#pragma acc kernels loop gang(100) vector(128)
		//for (int neuron_index = 0; neuron_index < neurons.size(); neuron_index++)
		//
	}

	// Get the results
	for (int i = 0; i < neurons.size(); i++){
		printf("%d ");
		for (int j = 0; j < neurons.size(); j++){
			printf("  ");
		}
	}


	return 0;
}