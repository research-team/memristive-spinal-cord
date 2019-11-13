#include <cstdio>
#include <cmath>
#include <utility>
#include <vector>
#include <ctime>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>
// for file writing
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <unistd.h>

using namespace std;

unsigned int global_id = 0;
unsigned int SIM_TIME_IN_STEPS;
const int LEG_STEPS = 1;             // [step] number of full cycle steps
const float SIM_STEP = 0.025;        // [s] simulation step

// stuff variables
const int neurons_in_ip = 196;       // number of neurons in interneuronal pool
const int neurons_in_group = 50;     // number of neurons in a group
const int neurons_in_aff_ip = 196;   // number of neurons in interneuronal pool
const int neurons_in_afferent = 120; // number of neurons in afferent

// neuron parameters
const float E_Na = 50.0;             // [mV] Reversal potential for the Sodium current
const float E_K = -100.0;            // [mV] Reversal potential for the Potassium current
const float E_L = -72.0;             // [mV] Reversal potential for the leak current
const float E_ex = 0.0;              // [mV] Reversal potential for excitatory input
const float E_in = -80.0;            // [mV] Reversal potential for inhibitory input
const float tau_syn_exc = 0.2;       // [ms] Decay time of excitatory synaptic current (ms)
const float tau_syn_inh = 2.0;       // [ms] Decay time of inhibitory synaptic current (ms)
const float V_adj = -63.0;           // adjusts threshold to around -50 mV
const float g_bar = 1500;            // [nS] the maximal possible conductivity


class Group {
public:
	Group() = default;
	string group_name;
	unsigned int id_start{};
	unsigned int id_end{};
	unsigned int group_size{};
};

// struct for human-readable initialization of connectomes
struct SynapseMetadata {
	unsigned int pre_id;         // pre neuron ID
	unsigned int post_id;        // post neuron ID
	unsigned int synapse_delay;  // [step] synaptic delay of the synapse (axonal delay is included to this delay)
	float synapse_weight;        // [nS] synaptic weight. Interpreted as changing conductivity of neuron membrane

	SynapseMetadata(int pre_id, int post_id, float synapse_delay, float synapse_weight){
		this->pre_id = pre_id;
		this->post_id = post_id;
		this->synapse_delay = lround(synapse_delay * (1 / SIM_STEP) + 0.5);  // round
		this->synapse_weight = synapse_weight;
	}
};

// struct for human-readable initialization of connectomes
struct GroupMetadata {
	Group group;
	float* g_exc;                // [nS] array of excitatory conductivity
	float* g_inh;                // [nS] array of inhibition conductivity
	float* voltage_array;        // [mV] array of membrane potential
	vector<float> spike_vector;  // [ms] spike times

	explicit GroupMetadata(Group group){
		this->group = move(group);
		voltage_array = new float[SIM_TIME_IN_STEPS];
		g_exc = new float[SIM_TIME_IN_STEPS];
		g_inh = new float[SIM_TIME_IN_STEPS];
	}
};

vector<GroupMetadata> all_groups;
vector<SynapseMetadata> all_synapses;

// form structs of neurons global ID and groups name
Group form_group(const string& group_name, int nrns_in_group = neurons_in_group) {
	Group group = Group();
	group.group_name = group_name;     // name of a neurons group
	group.id_start = global_id;        // first ID in the group
	group.id_end = global_id + nrns_in_group - 1;  // the latest ID in the group
	group.group_size = nrns_in_group;  // size of the neurons group

	all_groups.emplace_back(group);

	global_id += nrns_in_group;
	printf("Formed %s IDs [%d ... %d] = %d\n", group_name.c_str(), global_id - nrns_in_group, global_id - 1, nrns_in_group);

	return group;
}

float clipped_distribution(float mean, float sttdev) {
	random_device r;
	default_random_engine generator(r());
	normal_distribution<float> d_distr(mean, sttdev);

	float sigma = abs(mean) / 5;
	float probability = 0.001;
	float k = log(sqrt(2 * M_PI * probability * probability * sigma * sigma));
	float res = k < 0 ? sigma * sqrt(-2 * k) : sigma * sqrt(2 * k);
	float low = mean - res;
	float high = mean + res;

	float num = 0;
	while (true) {
		num = d_distr(generator);
		if(num >= low && num <= high) {
			return num;
		}
	}
}

int ms_to_step(float ms) {
	return (int)(ms / SIM_STEP);
}

float step_to_ms(int step) {
	return step * SIM_STEP;
}

void connect_one_to_all(const Group& pre_neurons, const Group& post_neurons, float syn_delay, float weight) {
	// Seed with a real random value, if available
	random_device r;
	default_random_engine generator(r());
	normal_distribution<float> delay_distr(syn_delay, syn_delay / 5);
	normal_distribution<float> weight_distr(weight, weight / 10);

	for (unsigned int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++) {
		for (unsigned int post_id = post_neurons.id_start; post_id <= post_neurons.id_end; post_id++) {
			all_synapses.emplace_back(pre_id, post_id, delay_distr(generator), weight_distr(generator));
		}
	}

	printf("Connect %s to %s [one_to_all] (1:%d). Total: %d W=%.2f, D=%.1f\n", pre_neurons.group_name.c_str(),
	       post_neurons.group_name.c_str(), post_neurons.group_size, pre_neurons.group_size * post_neurons.group_size,
	       weight, syn_delay);
}

void connect_fixed_outdegree(const Group& pre_neurons, const Group& post_neurons, float syn_delay, float syn_weight) {
	// connect neurons with uniform distribution and normal distributon for syn delay and syn_weight
	random_device r;
	default_random_engine generator(r());

	// connectomes rule
	uniform_int_distribution<int> post_nrn_id_distr(post_neurons.id_start, post_neurons.id_end);
	uniform_int_distribution<int> syn_outdegree_distr((int)(neurons_in_group * 0.6), neurons_in_group);

	int outdegree = syn_outdegree_distr(generator);

	for (unsigned int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++) {
		for (int i = 0; i < outdegree; i++) {
			int rand_post_id = post_nrn_id_distr(generator);
			// normal clipped distribution
			float syn_delay_distr = clipped_distribution(syn_delay, (syn_delay / 10));
			float syn_weight_distr = clipped_distribution(syn_weight, (syn_weight / 5));

			// check borders
			if (syn_delay_distr <= SIM_STEP)
				syn_delay_distr = SIM_STEP;

			if (syn_weight_distr >= g_bar)
				syn_weight_distr = g_bar;

			if (syn_weight_distr <= -g_bar)
				syn_weight_distr = -g_bar;
			// add meta data to the vector
			all_synapses.emplace_back(pre_id, rand_post_id, syn_delay_distr, syn_weight_distr);
		}
	}

	printf("Connect %s to %s [fixed_outdegree] (1:%d). Total: %d W=%.2f, D=%.1f\n",
	       pre_neurons.group_name.c_str(), post_neurons.group_name.c_str(),
	       outdegree, pre_neurons.group_size * outdegree, syn_weight, syn_delay);
}

void init_network(float inh_coef, int pedal, int has5ht) {
	float quadru_coef = pedal? 0.5 : 1;
	float sero_coef = has5ht? 5.3 : 1;

	/// groups of neurons
	Group EES = form_group("EES");
	Group E1 = form_group("E1");
	Group E2 = form_group("E2");
	Group E3 = form_group("E3");
	Group E4 = form_group("E4");
	Group E5 = form_group("E5");

	Group CV1 = form_group("CV1", 1);
	Group CV2 = form_group("CV2", 1);
	Group CV3 = form_group("CV3", 1);
	Group CV4 = form_group("CV4", 1);
	Group CV5 = form_group("CV5", 1);
	Group CD4 = form_group("CD4", 1);
	Group CD5 = form_group("CD5", 1);

	Group OM1_0 = form_group("OM1_0");
	Group OM1_1 = form_group("OM1_1");
	Group OM1_2_E = form_group("OM1_2_E");
	Group OM1_2_F = form_group("OM1_2_F");
	Group OM1_3 = form_group("OM1_3");

	Group OM2_0 = form_group("OM2_0");
	Group OM2_1 = form_group("OM2_1");
	Group OM2_2_E = form_group("OM2_2_E");
	Group OM2_2_F = form_group("OM2_2_F");
	Group OM2_3 = form_group("OM2_3");

	Group OM3_0 = form_group("OM3_0");
	Group OM3_1 = form_group("OM3_1");
	Group OM3_2_E = form_group("OM3_2_E");
	Group OM3_2_F = form_group("OM3_2_F");
	Group OM3_3 = form_group("OM3_3");

	Group OM4_0 = form_group("OM4_0");
	Group OM4_1 = form_group("OM4_1");
	Group OM4_2_E = form_group("OM4_2_E");
	Group OM4_2_F = form_group("OM4_2_F");
	Group OM4_3 = form_group("OM4_3");

	Group OM5_0 = form_group("OM5_0");
	Group OM5_1 = form_group("OM5_1");
	Group OM5_2_E = form_group("OM5_2_E");
	Group OM5_2_F = form_group("OM5_2_F");
	Group OM5_3 = form_group("OM5_3");

	Group MN_E = form_group("MN_E", 210);
	Group MN_F = form_group("MN_F", 180);

	Group Ia_E_aff = form_group("Ia_E_aff", neurons_in_afferent);
	Group Ia_F_aff = form_group("Ia_F_aff", neurons_in_afferent);

	Group R_E = form_group("R_E");
	Group R_F = form_group("R_F");

	Group Ia_E_pool = form_group("Ia_E_pool", neurons_in_aff_ip);
	Group Ia_F_pool = form_group("Ia_F_pool", neurons_in_aff_ip);

	Group eIP_E = form_group("eIP_E", neurons_in_ip);
	Group eIP_F = form_group("eIP_F", neurons_in_ip);

	Group iIP_E = form_group("iIP_E", neurons_in_ip);
	Group iIP_F = form_group("iIP_F", neurons_in_ip);

	/// connectomes
	connect_fixed_outdegree(EES, E1, 1, 50);
	connect_fixed_outdegree(E1, E2, 1, 20);
	connect_fixed_outdegree(E2, E3, 1, 20);
	connect_fixed_outdegree(E3, E4, 1, 20);
	connect_fixed_outdegree(E4, E5, 1, 20);

	connect_one_to_all(CV1, iIP_E, 0.5, 5);
	connect_one_to_all(CV2, iIP_E, 0.5, 5);
	connect_one_to_all(CV3, iIP_E, 0.5, 5);
	connect_one_to_all(CV4, iIP_E, 0.5, 5);
	connect_one_to_all(CV5, iIP_E, 0.5, 5);

	/// OM 1
	// input from EES group 1
	connect_one_to_all(CV1, OM1_0, 0.5, 0.2);
	connect_one_to_all(CV2, OM1_0, 0.5, 0.2);
	// [inhibition]
	connect_one_to_all(CV3, OM1_3, 0.5, 0.2);
	connect_one_to_all(CV4, OM1_3, 0.5, 100);
	connect_one_to_all(CV5, OM1_3, 0.5, 0.1);
	// E1
	connect_fixed_outdegree(E1, OM1_0, 1, 70);
//	// inner connectomes
	connect_fixed_outdegree(OM1_0, OM1_1, 0.2, 1);//6
	connect_fixed_outdegree(OM1_1, OM1_2_E, 0.5, 0.1 );//*2
	connect_fixed_outdegree(OM1_1, OM1_2_F, 1, 0.09);
	connect_fixed_outdegree(OM1_1, OM1_3, 1, 0.009); // 0.9
	connect_fixed_outdegree(OM1_2_E, OM1_1, 2.5, 0.001);//00000003
	connect_fixed_outdegree(OM1_2_F, OM1_1, 2.5, 0.001);//00000001
	connect_fixed_outdegree(OM1_2_E, OM1_3, 1, 0.009 );//*4
	connect_fixed_outdegree(OM1_2_F, OM1_3, 1, 0.008 );//*2
	connect_fixed_outdegree(OM1_3, OM1_1, 0.3, -0.1 * inh_coef);
	connect_fixed_outdegree(OM1_3, OM1_2_E, 0.5, -0.1* inh_coef);
	connect_fixed_outdegree(OM1_3, OM1_2_F, 0.5, -0.1 * inh_coef);
	// output to OM2
	connect_fixed_outdegree(OM1_2_F, OM2_2_F, 4, 1);
	// output to IP
	connect_fixed_outdegree(OM1_2_E, eIP_E, 1.5, 0.08 * 3);
//   connect_fixed_outdegree(OM1_2_F, eIP_F, 4, 0.009 * 6, neurons_in_ip);

	// /// OM 2
	// // input from EES group 2
	// connect_fixed_outdegree(E2, OM2_0, 2, 0.2);
	// // input from sensory [CV]
	// connect_one_to_all(CV2, OM2_0, 0.5, 0.87	 * quadru_coef * sero_coef);
	// connect_one_to_all(CV3, OM2_0, 0.5, 0.95 * quadru_coef * sero_coef);
	// // [inhibition]
	// connect_one_to_all(CV4, OM2_3, 1, 0.8);
	// connect_one_to_all(CV5, OM2_3, 1, 0.8);
	// // inner connectomes
	// connect_fixed_outdegree(OM2_0, OM2_1, 1, 0.05);
	// connect_fixed_outdegree(OM2_1, OM2_2_E, 1, 0.5);
	// connect_fixed_outdegree(OM2_1, OM2_2_F, 1, 0.5);
	// connect_fixed_outdegree(OM2_1, OM2_3, 1, 0.5);
	// connect_fixed_outdegree(OM2_2_E, OM2_1, 2.5, 0.5);
	// connect_fixed_outdegree(OM2_2_F, OM2_1, 2.5, 0.5);
	// connect_fixed_outdegree(OM2_2_E, OM2_3, 1, 0.5);
	// connect_fixed_outdegree(OM2_2_F, OM2_3, 1, 0.5);
	// connect_fixed_outdegree(OM2_3, OM2_1, 1, -1 * inh_coef);
	// connect_fixed_outdegree(OM2_3, OM2_2_E, 1, -1 * inh_coef);
	// connect_fixed_outdegree(OM2_3, OM2_2_F, 1, -1 * inh_coef); //-70
	// // output to OM3
	// connect_fixed_outdegree(OM2_2_F, OM3_2_F, 4, 0.5);
	// // output to IP
	// connect_fixed_outdegree(OM2_2_E, eIP_E, 2, 1, neurons_in_ip); // 5
	// connect_fixed_outdegree(OM2_2_F, eIP_F, 4, 1, neurons_in_ip);

	// /// OM 3
	// // input from EES group 3
	// connect_fixed_outdegree(E3, OM3_0, 1, 7);
	// // input from sensory [CV]
	// connect_one_to_all(CV3, OM3_0, 0.5, 10.5 * quadru_coef * sero_coef);
	// connect_one_to_all(CV4, OM3_0, 0.5, 10.5 * quadru_coef * sero_coef);
	// // [inhibition]
	// connect_one_to_all(CV5, OM3_3, 1, 80);
	// // input from sensory [CD]
	// connect_one_to_all(CD4, OM3_0, 1, 11);
	// // inner connectomes
	// connect_fixed_outdegree(OM3_0, OM3_1, 1, 50);
	// connect_fixed_outdegree(OM3_1, OM3_2_E, 1, 23);
	// connect_fixed_outdegree(OM3_1, OM3_2_F, 1, 30);
	// connect_fixed_outdegree(OM3_1, OM3_3, 1, 3);
	// connect_fixed_outdegree(OM3_2_E, OM3_1, 2.5, 23);
	// connect_fixed_outdegree(OM3_2_F, OM3_1, 2.5, 3);
	// connect_fixed_outdegree(OM3_2_E, OM3_3, 1, 3);
	// connect_fixed_outdegree(OM3_2_F, OM3_3, 1, 3);
	// connect_fixed_outdegree(OM3_3, OM3_1, 1, -70 * inh_coef);
	// connect_fixed_outdegree(OM3_3, OM3_2_E, 1, -70 * inh_coef);
	// connect_fixed_outdegree(OM3_3, OM3_2_F, 1, -5 * inh_coef);
	// // output to OM3
	// connect_fixed_outdegree(OM3_2_F, OM4_2_F, 4, 30);
	// // output to IP
	// connect_fixed_outdegree(OM3_2_E, eIP_E, 2, 8, neurons_in_ip); // 7 - 8
	// connect_fixed_outdegree(OM3_2_F, eIP_F, 4, 5, neurons_in_ip);

	// /// OM 4
	// // input from EES group 4
	// connect_fixed_outdegree(E4, OM4_0, 2, 7);
	// // input from sensory [CV]
	// connect_one_to_all(CV4, OM4_0, 0.5, 10.5 * quadru_coef * sero_coef);
	// connect_one_to_all(CV5, OM4_0, 0.5, 10.5 * quadru_coef * sero_coef);
	// // input from sensory [CD]
	// connect_one_to_all(CD4, OM4_0, 1, 11);
	// connect_one_to_all(CD5, OM4_0, 1, 11);
	// // inner connectomes
	// connect_fixed_outdegree(OM4_0, OM4_1, 3, 50);
	// connect_fixed_outdegree(OM4_1, OM4_2_E, 1, 25);
	// connect_fixed_outdegree(OM4_1, OM4_2_F, 1, 23);
	// connect_fixed_outdegree(OM4_1, OM4_3, 1, 3);
	// connect_fixed_outdegree(OM4_2_E, OM4_1, 2.5, 25);
	// connect_fixed_outdegree(OM4_2_F, OM4_1, 2.5, 3);
	// connect_fixed_outdegree(OM4_2_E, OM4_3, 1, 3);
	// connect_fixed_outdegree(OM4_2_F, OM4_3, 1, 3);
	// connect_fixed_outdegree(OM4_3, OM4_1, 1, -70 * inh_coef);
	// connect_fixed_outdegree(OM4_3, OM4_2_E, 1, -70 * inh_coef);
	// connect_fixed_outdegree(OM4_3, OM4_2_F, 1, -3 * inh_coef);
	// // output to OM4
	// connect_fixed_outdegree(OM4_2_F, OM5_2_F, 4, 30);
	// // output to IP
	// connect_fixed_outdegree(OM4_2_E, eIP_E, 2, 7, neurons_in_ip);
	// connect_fixed_outdegree(OM4_2_F, eIP_F, 4, 5, neurons_in_ip);

	/// OM 5
	//	// input from EES group 5
	//	connect_fixed_outdegree(E5, OM5_0, 1, 100);
	//	// input from sensory [CV]
	//	connect_one_to_all(CV5, OM5_0, 0.5, 10.8 * quadru_coef * sero_coef);
	//	// input from sensory [CD]
	//	connect_one_to_all(CD5, OM5_0, 1, 60);
	//	// inner connectomes
	//	connect_fixed_outdegree(OM5_0, OM5_1, 1, 1);
	//	connect_fixed_outdegree(OM5_1, OM5_2_E, 1, 10);
	//	connect_fixed_outdegree(OM5_1, OM5_2_F, 1, 10);
	//	connect_fixed_outdegree(OM5_1, OM5_3, 1, 10);
	//	connect_fixed_outdegree(OM5_2_E, OM5_1, 2.5, 1);
	//	connect_fixed_outdegree(OM5_2_F, OM5_1, 2.5, 1);
	//	connect_fixed_outdegree(OM5_2_E, OM5_3, 1, 0.00050);
	//	connect_fixed_outdegree(OM5_2_F, OM5_3, 1, 0.00050);
	//	connect_fixed_outdegree(OM5_3, OM5_1, 1, -0.1 * inh_coef);
	//	connect_fixed_outdegree(OM5_3, OM5_2_E, 1, -0.1 * inh_coef);
	//	connect_fixed_outdegree(OM5_3, OM5_2_F, 1, -0.1 * inh_coef);
	//// output to IP
	//	connect_fixed_outdegree(OM5_2_E, eIP_E, 1, 10, neurons_in_ip); // 2.5
	//	connect_fixed_outdegree(OM5_2_F, eIP_F, 4, 5, neurons_in_ip);


	/// reflex arc
//	connect_fixed_outdegree(iIP_E, eIP_F, 0.5, -0.001);
//	connect_fixed_outdegree(iIP_F, eIP_E, 0.5, -0.001);
//
//	connect_fixed_outdegree(iIP_E, OM1_2_F, 0.5, -0.005);
//	connect_fixed_outdegree(iIP_E, OM2_2_F, 0.5, -0.005);
//	connect_fixed_outdegree(iIP_E, OM3_2_F, 0.5, -0.005);
//	connect_fixed_outdegree(iIP_E, OM4_2_F, 0.5, -0.005);
//
//	connect_fixed_outdegree(EES, Ia_E_aff, 1, 500);
//	connect_fixed_outdegree(EES, Ia_F_aff, 1, 500);
//
//	connect_fixed_outdegree(eIP_E, MN_E, 0.5, 2.3);
//	connect_fixed_outdegree(eIP_F, MN_F, 5, 8);
//
//	connect_fixed_outdegree(iIP_E, Ia_E_pool, 1, 2);
//	connect_fixed_outdegree(iIP_F, Ia_F_pool, 1, 2);
//
//	connect_fixed_outdegree(Ia_E_pool, MN_F, 2, -0.8); // -4
//	connect_fixed_outdegree(Ia_E_pool, Ia_F_pool, 1, -0.01);
//	connect_fixed_outdegree(Ia_F_pool, MN_E, 2, -0.6);
//	connect_fixed_outdegree(Ia_F_pool, Ia_E_pool, 1, -0.01);
//
//	connect_fixed_outdegree(Ia_E_aff, MN_E, 3.5, 4.5);
//	connect_fixed_outdegree(Ia_F_aff, MN_F, 3.5, 6);
//
//	connect_fixed_outdegree(MN_E, R_E, 2, 0.1);
//	connect_fixed_outdegree(MN_F, R_F, 2, 0.1);
//
//	connect_fixed_outdegree(R_E, MN_E, 1, -0.005);
//	connect_fixed_outdegree(R_E, R_F, 1, -0.1);
//
//	connect_fixed_outdegree(R_F, MN_F, 1, -0.01);
//	connect_fixed_outdegree(R_F, R_E, 1, -0.1);
//
//	connect_fixed_outdegree(iIP_E, eIP_F, 0.5, -10, neurons_in_ip);
//	connect_fixed_outdegree(iIP_F, eIP_E, 0.5, -10, neurons_in_ip);
//
//	connect_fixed_outdegree(iIP_E, OM1_2_F, 0.5, -0.1, neurons_in_ip);
//	connect_fixed_outdegree(iIP_E, OM2_2_F, 0.5, -0.1, neurons_in_ip);
//	connect_fixed_outdegree(iIP_E, OM3_2_F, 0.5, -0.1, neurons_in_ip);
//	connect_fixed_outdegree(iIP_E, OM4_2_F, 0.5, -0.1, neurons_in_ip);
//
//	connect_fixed_outdegree(EES, Ia_E_aff, 1, 500);
//	connect_fixed_outdegree(EES, Ia_F_aff, 1, 500);
//
//	connect_fixed_outdegree(eIP_E, MN_E, 0.5, 2.3, neurons_in_moto); // 2.2
//	connect_fixed_outdegree(eIP_F, MN_F, 5, 8, neurons_in_moto);
//
//	connect_fixed_outdegree(iIP_E, Ia_E_pool, 1, 10, neurons_in_ip);
//	connect_fixed_outdegree(iIP_F, Ia_F_pool, 1, 10, neurons_in_ip);
//
//	connect_fixed_outdegree(Ia_E_pool, MN_F, 1, -4, neurons_in_ip);
//	connect_fixed_outdegree(Ia_E_pool, Ia_F_pool, 1, -1, neurons_in_ip);
//	connect_fixed_outdegree(Ia_F_pool, MN_E, 1, -4, neurons_in_ip);
//	connect_fixed_outdegree(Ia_F_pool, Ia_E_pool, 1, -1, neurons_in_ip);
//
//	connect_fixed_outdegree(Ia_E_aff, MN_E, 2, 8, neurons_in_moto);
//	connect_fixed_outdegree(Ia_F_aff, MN_F, 2, 6, neurons_in_moto);
//
//	connect_fixed_outdegree(MN_E, R_E, 2, 1);
//	connect_fixed_outdegree(MN_F, R_F, 2, 1);
//
//	connect_fixed_outdegree(R_E, MN_E, 2, -0.5, neurons_in_moto);
//	connect_fixed_outdegree(R_E, R_F, 2, -1);
//
//	connect_fixed_outdegree(R_F, MN_F, 2, -0.5, neurons_in_moto);
//	connect_fixed_outdegree(R_F, R_E, 2, -1);*/
}

void save(int test_index, GroupMetadata &metadata, const string& folder){
	ofstream file;
	string file_name = "/dat/" + to_string(test_index) + "_" + metadata.group.group_name + ".dat";

	file.open(folder + file_name);
	// save voltage
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++)
		file << metadata.voltage_array[sim_iter] << " ";
	file << endl;

	// save g_exc
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++)
		file << metadata.g_exc[sim_iter] << " ";
	file << endl;

	// save g_inh
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++)
		file << metadata.g_inh[sim_iter] << " ";
	file << endl;

	// save spikes
	for (float const& value: metadata.spike_vector) {
		file << value << " ";
	}
	file.close();

	cout << "Saved to: " << folder + file_name << endl;
}

void save_result(int test_index, int save_all) {
	string current_path = getcwd(nullptr, 0);

	printf("[Test #%d] Save %s results to: %s \n", test_index, (save_all == 0)? "MOTO" : "ALL", current_path.c_str());

	for(GroupMetadata &metadata : all_groups) {
		if (save_all == 0) {
			if(metadata.group.group_name == "MN_E")
				save(test_index, metadata, current_path);
			if(metadata.group.group_name == "MN_F")
				save(test_index, metadata, current_path);
		} else {
			save(test_index, metadata, current_path);
		}
	}
}

// get datasize of current variable type and its number
template <typename type>
unsigned int datasize(unsigned int size) {
	return sizeof(type) * size;
}

// fill array with current value
template <typename type>
void init_array(type *array, unsigned int size, type value) {
	for(int i = 0; i < size; i++)
		array[i] = value;
}
// fill array by normal distribution
template <typename type>
void rand_normal_init_array(type *array, unsigned int size, type mean, type stddev) {
	random_device r;
	default_random_engine generator(r());
	normal_distribution<float> distr(mean, stddev);

	for(unsigned int i = 0; i < size; i++)
		array[i] = (type)distr(generator);
}

int get_skin_stim_time(int cms) {
	if (cms == 21)
		return 25;
	if (cms == 15)
		return 50;
	return 125;
}

void copy_data_to(GroupMetadata &metadata,
                  const float* nrn_v_m,
                  const float* nrn_g_exc,
                  const float* nrn_g_inh,
                  const bool *nrn_has_spike,
                  unsigned int sim_iter) {

	float nrn_mean_volt = 0;
	float nrn_mean_g_exc = 0;
	float nrn_mean_g_inh = 0;

	for(unsigned int tid = metadata.group.id_start; tid <= metadata.group.id_end; tid++) {
		nrn_mean_volt += nrn_v_m[tid];
		nrn_mean_g_exc += nrn_g_exc[tid];
		nrn_mean_g_inh += nrn_g_inh[tid];
		if (nrn_has_spike[tid]) {
			metadata.spike_vector.push_back(step_to_ms(sim_iter) + 0.25);
		}
	}
	metadata.g_exc[sim_iter] = nrn_mean_g_exc / metadata.group.group_size;
	metadata.g_inh[sim_iter] = nrn_mean_g_inh / metadata.group.group_size;
	metadata.voltage_array[sim_iter] = nrn_mean_volt / metadata.group.group_size;
}

void bimodal_distr_for_moto_neurons(float *nrn_diameter) {
	/*
	 *
	 */
	random_device r1;
	default_random_engine gen(r1());

	int loc_active = 57;
	int scale_active = 6;
	int loc_standby = 27;
	int scale_standby = 3;

	int standby_percent = 70;
	int MN_E_beg = 1557;
	int MN_E_end = 1766;
	int MN_F_beg = 1767;
	int MN_F_end = 1946;

	int nrn_number_extensor = MN_E_end - MN_E_beg;
	int nrn_number_flexor = MN_F_end - MN_E_beg;

	int standby_size_extensor = (int)(nrn_number_extensor * standby_percent / 100);
	int standby_size_flexor = (int)(nrn_number_flexor * standby_percent / 100);

	normal_distribution<float> g_active(loc_active, scale_active);
	normal_distribution<float> g_standby(loc_standby, scale_standby);

	for (int i = MN_E_beg; i < MN_E_beg + standby_size_extensor; i++)
		nrn_diameter[i] = g_standby(gen);

	for (int i = MN_E_beg + standby_size_extensor; i <= MN_E_end; i++)
		nrn_diameter[i] = g_active(gen);

	for (int i = MN_F_beg; i < MN_F_beg + standby_size_flexor; i++)
		nrn_diameter[i] = g_standby(gen);

	for (int i = MN_F_beg + standby_size_flexor; i <= MN_F_end; i++)
		nrn_diameter[i] = g_active(gen);
}

void simulate(int cms, int ees, int inh, int ped, int ht5, int save_all, int itest) {
	chrono::time_point<chrono::system_clock> simulation_t_start, simulation_t_end;

	const unsigned int skin_stim_time = get_skin_stim_time(cms);
	const unsigned int T_simulation = 11 * skin_stim_time * LEG_STEPS;
	// calculate how much steps in simulation time [steps]
	SIM_TIME_IN_STEPS = (unsigned int)(T_simulation / SIM_STEP);

	// calculate spike frequency and C0/C1 activation time in steps
	auto ees_spike_each_step = (unsigned int)(1000 / ees / SIM_STEP);
	auto steps_activation_C0 = (unsigned int)(5 * skin_stim_time / SIM_STEP);
	auto steps_activation_C1 = (unsigned int)(6 * skin_stim_time / SIM_STEP);

	// init neuron groups and connectomes
	init_network((float) inh / 100, ped == 4, ht5 == 1);

	const auto neurons_number = global_id;
	const unsigned int synapses_number = static_cast<int>(all_synapses.size());

	/// CPU variables
	// neuron variables
	float nrn_n[neurons_number];             // dimensionless quantity [0 .. 1] of potassium channel activation
	float nrn_h[neurons_number];             // dimensionless quantity [0 .. 1] of sodium channel activation
	float nrn_m[neurons_number];             // dimensionless quantity [0 .. 1] of sodium channel inactivation
	float nrn_v_m[neurons_number];           // [mV] neuron membrane potential
	float nrn_c_m[neurons_number];           // [mV] neuron membrane potential
	float nrn_g_exc[neurons_number];         // [nS] excitatory synapse exponential conductance
	float nrn_g_inh[neurons_number];         // [nS] inhibitory synapse exponential conductance
	float nrn_threshold[neurons_number];     // [mV] threshold levels
	bool nrn_has_spike[neurons_number];      // neuron state - has spike or not
	int nrn_ref_time[neurons_number];        // [step] neuron refractory time
	int nrn_ref_time_timer[neurons_number];  // [step] neuron refractory time timer
	float nrn_diameter[neurons_number];
	float nrn_g_Na[neurons_number];
	float nrn_g_K[neurons_number];
	float nrn_g_L[neurons_number];

	int begin_C_spiking[5] = {ms_to_step(0),
	                          ms_to_step(skin_stim_time),
	                          ms_to_step(2 * skin_stim_time),
	                          ms_to_step(3 * skin_stim_time),
	                          ms_to_step(5 * skin_stim_time)};
	int end_C_spiking[5] = {ms_to_step(skin_stim_time - 0.1),
	                        ms_to_step(2 * skin_stim_time - 0.1),
	                        ms_to_step(3 * skin_stim_time - 0.1),
	                        ms_to_step(5 * skin_stim_time - 0.1),
	                        ms_to_step(6 * skin_stim_time - 0.1)};

	random_device r;
	default_random_engine generator(r());
	uniform_real_distribution<float> standard_uniform(0, 1);

	random_device r_s;
	default_random_engine generator_s(r_s());
	lognormal_distribution<float> distribution(0.6, 0.2); // 0.5 0.2

	// fill arrays by initial data
	init_array<float>(nrn_n, neurons_number, 0);             // by default neurons have closed potassium channel
	init_array<float>(nrn_h, neurons_number, 1);             // by default neurons have opened sodium channel activation
	init_array<float>(nrn_m, neurons_number, 0);             // by default neurons have closed sodium channel inactivation
	init_array<float>(nrn_v_m, neurons_number, E_L);               // by default neurons have E_L membrane state at start
	init_array<float>(nrn_g_exc, neurons_number, 0);
	init_array<float>(nrn_g_inh, neurons_number, 0);         // by default neurons have zero inhibitory synaptic conductivity
	init_array<bool>(nrn_has_spike, neurons_number, false);  // by default neurons haven't spikes at start
	init_array<int>(nrn_ref_time_timer, neurons_number, 0);  // by default neurons have ref_t timers as 0
	init_array<float>(nrn_diameter, neurons_number, 0);
	init_array<float>(nrn_g_Na, neurons_number, 0);
	init_array<float>(nrn_g_K, neurons_number, 0);
	init_array<float>(nrn_g_L, neurons_number, 0);

	rand_normal_init_array<int>(nrn_ref_time, neurons_number, (int)(3 / SIM_STEP), (int)(0.4 / SIM_STEP));  // neuron ref time, aprx interval is (1.8, 4.2)
	rand_normal_init_array<float>(nrn_threshold, neurons_number, -50, 0.4); // neurons threshold (-51.2, -48.8)

	/// fill array of diameters for moto neurons
	bimodal_distr_for_moto_neurons(nrn_diameter);

	random_device r1;
	default_random_engine gen(r1());
	uniform_real_distribution<float> d_inter_distr(1, 10);
	uniform_real_distribution<float> d_Ia_aff_distr(10, 20);

	for(int i = 0; i < neurons_number; i++) {
		nrn_diameter[i] = d_inter_distr(gen);
	}

	// EES, E1, E2, E3, E4, E5
	for (int i = 0; i < 300; i++)
		nrn_diameter[i] = 5;

	// fill array of diameters for Ia_aff neurons
	for (int i = 1947; i < 2186; i++)
		nrn_diameter[i] = d_Ia_aff_distr(gen);

	// fill C_m, g_Na, g_K, g_L arrays
	for(int i = 0; i < neurons_number; i++) {
		float d = nrn_diameter[i];
		float S = M_PI * d * d;

		if(i >= 1557 && i <= 1946)
			nrn_c_m[i] = S * 0.02;
		else
			nrn_c_m[i] = S * 0.01;

		nrn_g_K[i] = S * 3.0;
		nrn_g_L[i] = S * 0.02;
		nrn_g_Na[i] = S * 0.5;
	}

	// synapse variables
	auto *synapses_pre_nrn_id = (int *) malloc(datasize<int>(synapses_number));
	auto *synapses_post_nrn_id = (int *) malloc(datasize<int>(synapses_number));
	auto *synapses_weight = (float *) malloc(datasize<float>(synapses_number));
	auto *synapses_delay = (int *) malloc(datasize<int>(synapses_number));
	auto *synapses_delay_timer = (int *) malloc(datasize<int>(synapses_number));
	init_array<int>(synapses_delay_timer, synapses_number, -1);

	// fill arrays of synapses
	unsigned int syn_id = 0;
	for(SynapseMetadata metadata : all_synapses) {
		synapses_pre_nrn_id[syn_id] = metadata.pre_id;
		synapses_post_nrn_id[syn_id] = metadata.post_id;
		synapses_delay[syn_id] = metadata.synapse_delay;
		synapses_weight[syn_id] = metadata.synapse_weight;
		syn_id++;
	}
	all_synapses.clear();

	printf("* * * Start the main loop * * *\n");

	// stuff variables for controlling C0/C1 activation
	int local_iter = 0;

	bool C0_activated = false; // start from extensor
	bool C0_early_activated = false;
	bool CV1_activated;
	bool CV2_activated;
	bool CV3_activated;
	bool CV4_activated;
	bool CV5_activated;
	bool EES_activated;
	int shift_time_by_step = 0;
	int decrease_lvl_Ia_spikes;
	int shifted_iter_time = 0;

	simulation_t_start = chrono::system_clock::now();

	// the main simulation loop
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++) {
		if (sim_iter % 100 == 0)
			printf("%f%%\n", 1.0 * sim_iter / SIM_TIME_IN_STEPS * 100);
		CV1_activated = false;
		CV2_activated = false;
		CV3_activated = false;
		CV4_activated = false;
		CV5_activated = false;
		EES_activated = (sim_iter % ees_spike_each_step == 0);

		decrease_lvl_Ia_spikes = 0;
		// if flexor C0 activated, find the end of it and change to C1
		if (C0_activated) {
			if (local_iter != 0 && local_iter % steps_activation_C0 == 0) {
				C0_activated = false;
				local_iter = 0;
				shift_time_by_step += steps_activation_C0;
			}
			if (local_iter != 0 && (local_iter + 400) % steps_activation_C0 == 0)
				C0_early_activated = false;
			// if extensor C1 activated, find the end of it and change to C0
		} else {
			if (local_iter != 0 && local_iter % steps_activation_C1 == 0) {
				C0_activated = true;
				local_iter = 0;
				shift_time_by_step += steps_activation_C1;
			}
			if (local_iter != 0 && (local_iter + 400) % steps_activation_C1 == 0)
				C0_early_activated = true;
		}

		// update local iter (warning: can be resetted at C0/C1 activation)
		local_iter++;

		shifted_iter_time = sim_iter - shift_time_by_step;

		if ((begin_C_spiking[0] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[0])) CV1_activated = true;
		if ((begin_C_spiking[1] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[1])) CV2_activated = true;
		if ((begin_C_spiking[2] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[2])) CV3_activated = true;
		if ((begin_C_spiking[3] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[3])) CV4_activated = true;
		if ((begin_C_spiking[4] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[4])) CV5_activated = true;

		if (CV1_activated) decrease_lvl_Ia_spikes = 2;
		if (CV2_activated) decrease_lvl_Ia_spikes = 1;
		if (CV3_activated) decrease_lvl_Ia_spikes = 0;
		if (CV4_activated) decrease_lvl_Ia_spikes = 1;
		if (CV5_activated) decrease_lvl_Ia_spikes = 2;

		/** ================================================= **/
		/** ==================N E U R O N S================== **/
		/** ================================================= **/
#pragma omp parallel for num_threads(4) default(shared)
		for (unsigned int tid = 0; tid < neurons_number; tid++) {
			// Ia aff extensor/flexor, control spike number of Ia afferent by resetting neuron current
			if (1947 <= tid && tid <= 2186) {
				// rule for the 2nd level
				if (decrease_lvl_Ia_spikes == 1 && tid % 3 == 0) {
					// reset current of 1/3 of neurons
					nrn_g_exc[tid] = 0;  // set maximal inhibitory conductivity
				} else {
					// rule for the 3rd level
					if (decrease_lvl_Ia_spikes == 2 && tid % 2 == 0) {
						// reset current of 1/2 of neurons
						nrn_g_exc[tid] = 0;  // set maximal inhibitory conductivity
					}
				}
			}

			nrn_has_spike[tid] = false;

			// generating spikes for EES
			if (tid < 50 && EES_activated)
				nrn_has_spike[tid] = true;
			// iIP_F
			if (C0_activated && C0_early_activated && 3267 <= tid && tid <= 3462 && (sim_iter % 10 == 0))
				nrn_has_spike[3267 + static_cast<int>(neurons_in_ip * standard_uniform(generator))] = true;
			// skin stimulations
			if (!C0_activated) {
				if (tid == 300 && CV1_activated && standard_uniform(generator_s) >= 0.5) nrn_has_spike[tid] = true;
				if (tid == 301 && CV2_activated && standard_uniform(generator_s) >= 0.5) nrn_has_spike[tid] = true;
				if (tid == 302 && CV3_activated && standard_uniform(generator_s) >= 0.5) nrn_has_spike[tid] = true;
				if (tid == 303 && CV4_activated && standard_uniform(generator_s) >= 0.5) nrn_has_spike[tid] = true;
				if (tid == 304 && CV5_activated && standard_uniform(generator_s) >= 0.5) nrn_has_spike[tid] = true;
			}
			// the maximal value of input current
			if (nrn_g_exc[tid] > g_bar)
				nrn_g_exc[tid] = g_bar;
			if (nrn_g_inh[tid] > g_bar)
				nrn_g_inh[tid] = g_bar;
			// check the Voltage borders
			if (nrn_v_m[tid] > 100)
				nrn_v_m[tid] = 100;
			if (nrn_v_m[tid] < -100)
				nrn_v_m[tid] = -100;
			// use temporary V variable as V_m with adjust
			const float V = nrn_v_m[tid] - V_adj;

			// transition rates between open and closed states of the potassium channels
			float alpha_n = 0.032 * (15.0 - V) / (exp((15.0 - V) / 5.0) - 1.0);
			if (alpha_n != alpha_n)
				alpha_n = 0;
			float beta_n = 0.5 * exp((10.0 - V) / 40.0);
			if (beta_n != beta_n)
				beta_n = 0;

			// transition rates between open and closed states of the activation of sodium channels
			float alpha_m = 0.32 * (13.0 - V) / (exp((13.0 - V) / 4.0) - 1.0);
			if (alpha_m != alpha_m)
				alpha_m = 0;
			float beta_m = 0.28 * (V - 40.0) / (exp((V - 40.0) / 5.0) - 1.0);
			if (beta_m != beta_m)
				beta_m = 0;

			// transition rates between open and closed states of the inactivation of sodium channels
			float alpha_h = 0.128 * exp((17.0 - V) / 18.0);
			if (alpha_h != alpha_h)
				alpha_h = 0;
			float beta_h = 4.0 / (1.0 + exp((40.0 - V) / 5.0));
			if (beta_h != beta_h)
				beta_h = 0;

			// re-calculate activation variables
			nrn_n[tid] += (alpha_n - (alpha_n + beta_n) * nrn_n[tid]) * SIM_STEP;
			nrn_m[tid] += (alpha_m - (alpha_m + beta_m) * nrn_m[tid]) * SIM_STEP;
			nrn_h[tid] += (alpha_h - (alpha_h + beta_h) * nrn_h[tid]) * SIM_STEP;

			// ionic currents
			float I_NA = nrn_g_Na[tid] * pow(nrn_m[tid], 3) * nrn_h[tid] * (nrn_v_m[tid] - E_Na);
			float I_K = nrn_g_K[tid] * pow(nrn_n[tid], 4) * (nrn_v_m[tid] - E_K);
			float I_L = nrn_g_L[tid] * (nrn_v_m[tid] - E_L);
			float I_syn_exc = nrn_g_exc[tid] * (nrn_v_m[tid] - E_ex);
			float I_syn_inh = nrn_g_inh[tid] * (nrn_v_m[tid] - E_in);

			// if neuron in the refractory state -- ignore synaptic inputs. Re-calculate membrane potential
			if (nrn_ref_time_timer[tid] > 0)
				nrn_v_m[tid] += -(I_L + I_K + I_NA) / nrn_c_m[tid] * SIM_STEP;
			else
				nrn_v_m[tid] += -(I_L + I_K + I_NA + I_syn_exc + 4 * I_syn_inh) / nrn_c_m[tid] * SIM_STEP;

			// re-calculate conductance
			nrn_g_exc[tid] += -nrn_g_exc[tid] / tau_syn_exc * SIM_STEP;
			nrn_g_inh[tid] += -nrn_g_inh[tid] / tau_syn_inh * SIM_STEP;

			// check the Voltage borders
			if (nrn_v_m[tid] > 100)
				nrn_v_m[tid] = 100;
			if (nrn_v_m[tid] < -100)
				nrn_v_m[tid] = -100;

			// (threshold && not in refractory period)
			if ((nrn_v_m[tid] >= nrn_threshold[tid]) && (nrn_ref_time_timer[tid] == 0)) {
				nrn_has_spike[tid] = true;  // set spike state. It will be used in the "synapses_kernel"
				nrn_ref_time_timer[tid] = nrn_ref_time[tid];  // set the refractory period
			}

			// update the refractory period timer
			if (nrn_ref_time_timer[tid] > 0)
				nrn_ref_time_timer[tid]--;
		}

		// fill records arrays
		for(GroupMetadata &metadata : all_groups) {
			if (save_all == 0) {
				if (metadata.group.group_name == "MN_E")
					copy_data_to(metadata, nrn_v_m, nrn_g_exc, nrn_g_inh, nrn_has_spike, sim_iter);
				if (metadata.group.group_name == "MN_F")
					copy_data_to(metadata, nrn_v_m, nrn_g_exc, nrn_g_inh, nrn_has_spike, sim_iter);
			} else
				copy_data_to(metadata, nrn_v_m, nrn_g_exc, nrn_g_inh, nrn_has_spike, sim_iter);
		}

		/** ================================================= **/
		/** ==================S Y N A P S E================== **/
		/** ================================================= **/
#pragma omp parallel for num_threads(4) shared(nrn_v_m, nrn_has_spike, synapses_pre_nrn_id, synapses_post_nrn_id, synapses_delay, synapses_delay_timer, synapses_weight)
		for (unsigned int tid = 0; tid < synapses_number; tid++) {
			// add synaptic delay if neuron has spike
			if (synapses_delay_timer[tid] == -1 && nrn_has_spike[synapses_pre_nrn_id[tid]]) {
				synapses_delay_timer[tid] = synapses_delay[tid];
			}
			// if synaptic delay is zero it means the time when synapse increase I by synaptic weight
			if (synapses_delay_timer[tid] == 0) {
				// post neuron ID = synapses_post_nrn_id[tid][syn_id], thread-safe (!)
				if (synapses_weight[tid] >= 0) {
#pragma omp atomic
					nrn_g_exc[synapses_post_nrn_id[tid]] += synapses_weight[tid];
				} else {
#pragma omp atomic
					nrn_g_inh[synapses_post_nrn_id[tid]] -= synapses_weight[tid];
				}
				// make synapse timer a "free" for next spikes
				synapses_delay_timer[tid] = -1;
			}
			// update synapse delay timer
			if (synapses_delay_timer[tid] > 0) {
				synapses_delay_timer[tid]--;
			}
		}
	} // end of the simulation iteration loop

	simulation_t_end = chrono::system_clock::now();

	// save recorded data
	save_result(itest, save_all);

	auto sim_time_diff = chrono::duration_cast<chrono::milliseconds>(simulation_t_end - simulation_t_start).count();
	printf("Elapsed %li ms (measured) | T_sim = %d ms\n", sim_time_diff, T_simulation);
}

// runner
int main(int argc, char* argv[]) {
	int cms = stoi(argv[1]);
	int ees = stoi(argv[2]);
	int inh = stoi(argv[3]);
	int ped = stoi(argv[4]);
	int ht5 = stoi(argv[5]);
	int save_all = stoi(argv[6]);
	int itest = stoi(argv[7]);

	simulate(cms, ees, inh, ped, ht5, save_all, itest);

	return 0;
}