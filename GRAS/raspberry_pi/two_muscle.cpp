#define DEBUG
#include <omp.h>
#include <cstdio>
#include <cmath>
#include <utility>
#include <vector>
#include <random>
#include <chrono>
#include <string>
// for file writing
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <unistd.h>

using namespace std;

unsigned int global_id = 0;
const float SIM_STEP = 0.25;

// stuff variables
const int LEG_STEPS = 1;             // [step] number of full cycle steps
const int syn_outdegree = 20;        // synapse number outgoing from one neuron
const int neurons_in_ip = 25;        // number of neurons in interneuronal pool
const int neurons_in_moto = 25;      // motoneurons number
const int neurons_in_group = 15;     // number of neurons in a group
const int neurons_in_aff_ip = 25;    // number of neurons in interneuronal pool
const int neurons_in_afferent = 15;  // number of neurons in afferent

const unsigned short skin_stim_time = 25;
const unsigned int T_simulation = 11 * skin_stim_time * LEG_STEPS;
const unsigned int SIM_TIME_IN_STEPS = (unsigned int)(T_simulation / SIM_STEP);

class Group {
public:
	Group() = default;
	string group_name;
	unsigned int id_start{};
	unsigned int id_end{};
	unsigned short group_size{};
};

// struct for human-readable initialization of connectomes
struct SynapseMetadata {
	unsigned int pre_id;         // [id] pre neuron
	unsigned int post_id;        // [id] post neuron
	unsigned int synapse_delay;  // [step] synaptic delay of the synapse (axonal delay is included to this delay)
	short synapse_weight;        // [nS] synaptic weight. Interpreted as changing conductivity of neuron membrane

	SynapseMetadata(int pre_id, int post_id, float synapse_delay, short synapse_weight){
		this->pre_id = pre_id;
		this->post_id = post_id;
		this->synapse_delay = lround(synapse_delay * (1 / SIM_STEP) + 0.5);
		this->synapse_weight = synapse_weight;
	}
};

// struct for human-readable initialization of connectomes
struct GroupMetadata {
	Group group;
	vector<float> spike_vector;  // [ms] spike times
	#ifdef DEBUG
	unsigned short* voltage_array;        // [mV] array of membrane potential
	#endif

	explicit GroupMetadata(Group group){
		this->group = move(group);
		#ifdef DEBUG
		voltage_array = new unsigned short[SIM_TIME_IN_STEPS];
		#endif
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
	printf("Formed %s IDs [%d ... %d] = %d\n",
	       group_name.c_str(), global_id - nrns_in_group, global_id - 1, nrns_in_group);

	return group;
}

float step_to_ms(unsigned int step) { return step * SIM_STEP; }
int ms_to_step(float ms) { return (int)(ms / SIM_STEP); }

void connect_one_to_all(const Group& pre_neurons, const Group& post_neurons, float syn_delay, float weight) {
	// Seed with a real random value, if available
	random_device r;
	default_random_engine generator(r());
	normal_distribution<float> delay_distr(syn_delay, syn_delay / 5);
	normal_distribution<float> weight_distr(weight, weight / 10);

	for (unsigned int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++)
		for (unsigned int post_id = post_neurons.id_start; post_id <= post_neurons.id_end; post_id++)
			all_synapses.emplace_back(pre_id, post_id, delay_distr(generator), weight_distr(generator));

	printf("Connect %s to %s [one_to_all] (1:%d). Total: %d W=%.2f, D=%.1f\n", pre_neurons.group_name.c_str(),
		   post_neurons.group_name.c_str(), post_neurons.group_size, pre_neurons.group_size * post_neurons.group_size,
		   weight, syn_delay);
}

void connect(const Group& pre_neurons, const Group& post_neurons, float syn_delay, float syn_weight,
             int outdegree= syn_outdegree,
             bool no_distr= false) {
	// connect neurons with uniform distribution and normal distributon for syn delay and syn_weight
	random_device r;
	default_random_engine generator(r());
	uniform_int_distribution<int> id_distr(post_neurons.id_start, post_neurons.id_end);
	normal_distribution<float> delay_distr_gen(syn_delay, syn_delay / 5);
	normal_distribution<float> weight_distr_gen(syn_weight, syn_weight / 10);

	for (unsigned int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++) {
		for (int i = 0; i < outdegree; i++) {
			int rand_post_id = id_distr(generator);
			float syn_delay_distr = delay_distr_gen(generator);
			float weight_distr = weight_distr_gen(generator);

			if (syn_delay_distr <= 0.2)
				syn_delay_distr = 0.2;

			short syn_weight_distr = static_cast<short>(weight_distr);

			if (no_distr)
				all_synapses.emplace_back(pre_id, rand_post_id, syn_delay, syn_weight);
			else
				all_synapses.emplace_back(pre_id, rand_post_id, syn_delay_distr, syn_weight_distr);
		}
	}

	printf("Connect %s to %s [fixed_outdegree] (1:%d). Total: %d W=%.2f, D=%.1f\n",
	       pre_neurons.group_name.c_str(), post_neurons.group_name.c_str(),
	       outdegree, pre_neurons.group_size * outdegree, syn_weight, syn_delay);
}

void init_network() {
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

	Group MN_E = form_group("MN_E", neurons_in_moto);
	Group MN_F = form_group("MN_F", neurons_in_moto);

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
	connect(EES, E1, 1, 5000, syn_outdegree, true);
	connect(E1, E2, 1, 5000, syn_outdegree, true);
	connect(E2, E3, 1, 5000, syn_outdegree, true);
	connect(E3, E4, 1, 5000, syn_outdegree, true);
	connect(E4, E5, 1, 5000, syn_outdegree, true);

	connect_one_to_all(CV1, iIP_E, 0.5, 500);
	connect_one_to_all(CV2, iIP_E, 0.5, 500);
	connect_one_to_all(CV3, iIP_E, 0.5, 500);
	connect_one_to_all(CV4, iIP_E, 0.5, 500);
	connect_one_to_all(CV5, iIP_E, 0.5, 500);

	/// OM 1
	// input from EES group 1
	connect(E1, OM1_0, 2, 1000);
	// input from sensory
	connect_one_to_all(CV1, OM1_0, 0.5, 1900);
	connect_one_to_all(CV2, OM1_0, 0.5, 1900);
	// [inhibition]
	connect_one_to_all(CV3, OM1_3, 0.25, 5000);
	connect_one_to_all(CV4, OM1_3, 0.25, 5000);
	connect_one_to_all(CV5, OM1_3, 0.25, 5000);
	// inner connectomes
	connect(OM1_0, OM1_1, 1, 2000);
	connect(OM1_1, OM1_2_E, 1, 1500);
	connect(OM1_1, OM1_2_F, 1, 27);
	connect(OM1_1, OM1_3, 1, 400);
	connect(OM1_2_E, OM1_1, 2.5, 1500);
	connect(OM1_2_F, OM1_1, 2.5, 27);
	connect(OM1_2_E, OM1_3, 1, 400);
	connect(OM1_2_F, OM1_3, 1, 4);
	connect(OM1_3, OM1_1, 2, -1000);
	connect(OM1_3, OM1_2_E, 0.5, -1000);
	connect(OM1_3, OM1_2_F, 2, -3);
	// output to OM2
	connect(OM1_2_F, OM2_2_F, 4, 30);
	// output to IP
	connect(OM1_2_E, eIP_E, 2, 1500, neurons_in_ip);
	connect(OM1_2_F, eIP_F, 4, 5, neurons_in_ip);

	/// OM 2
	// input from EES group 2
	connect(E2, OM2_0, 2, 680);
	// input from sensory [CV]
	connect_one_to_all(CV2, OM2_0, 0.5, 1900);
	connect_one_to_all(CV3, OM2_0, 0.5, 1900);
	// [inhibition]
	connect_one_to_all(CV4, OM2_3, 1, 5000);
	connect_one_to_all(CV5, OM2_3, 1, 5000);
	// inner connectomes
	connect(OM2_0, OM2_1, 1, 2000);
	connect(OM2_1, OM2_2_E, 1, 1500);
	connect(OM2_1, OM2_2_F, 1, 27);
	connect(OM2_1, OM2_3, 1, 400);
	connect(OM2_2_E, OM2_1, 2.5, 1500);
	connect(OM2_2_F, OM2_1, 2.5, 27);
	connect(OM2_2_E, OM2_3, 1, 400);
	connect(OM2_2_F, OM2_3, 1, 4);
	connect(OM2_3, OM2_1, 2, -1000);
	connect(OM2_3, OM2_2_E, 0.5, -1000);
	connect(OM2_3, OM2_2_F, 2, -3);
	// output to OM3
	connect(OM2_2_F, OM3_2_F, 4, 30);
	// output to IP
	connect(OM2_2_E, eIP_E, 2, 1500, neurons_in_ip);
	connect(OM2_2_F, eIP_F, 4, 5, neurons_in_ip);

	/// OM 3
	// input from EES group 3
	connect(E3, OM3_0, 3, 670);
	// input from sensory [CV]
	connect_one_to_all(CV3, OM3_0, 0.5, 1900);
	connect_one_to_all(CV4, OM3_0, 0.5, 1900);
	// [inhibition]
	connect_one_to_all(CV5, OM3_3, 1, 5000);
	// inner connectomes
	connect(OM3_0, OM3_1, 1, 2000);
	connect(OM3_1, OM3_2_E, 1, 1500);
	connect(OM3_1, OM3_2_F, 1, 27);
	connect(OM3_1, OM3_3, 1, 400);
	connect(OM3_2_E, OM3_1, 2.5, 1500);
	connect(OM3_2_F, OM3_1, 2.5, 27);
	connect(OM3_2_E, OM3_3, 1, 400);
	connect(OM3_2_F, OM3_3, 1, 4);
	connect(OM3_3, OM3_1, 2, -1000);
	connect(OM3_3, OM3_2_E, 0.5, -1000);
	connect(OM3_3, OM3_2_F, 2, -3);
	// output to OM3
	connect(OM3_2_F, OM4_2_F, 4, 30);
	// output to IP
	connect(OM3_2_E, eIP_E, 2, 1500, neurons_in_ip);
	connect(OM3_2_F, eIP_F, 4, 5, neurons_in_ip);

	/// OM 4
	// input from EES group 4
	connect(E4, OM4_0, 3, 680);
	// input from sensory [CV]
	connect_one_to_all(CV4, OM4_0, 0.5, 1900);
	connect_one_to_all(CV5, OM4_0, 0.5, 1900);
	// inner connectomes
	connect(OM4_0, OM4_1, 1, 2000);
	connect(OM4_1, OM4_2_E, 1, 1500);
	connect(OM4_1, OM4_2_F, 1, 27);
	connect(OM4_1, OM4_3, 1, 400);
	connect(OM4_2_E, OM4_1, 2.5, 1500);
	connect(OM4_2_F, OM4_1, 2.5, 27);
	connect(OM4_2_E, OM4_3, 1, 400);
	connect(OM4_2_F, OM4_3, 1, 4);
	connect(OM4_3, OM4_1, 2, -1000);
	connect(OM4_3, OM4_2_E, 0.5, -1000);
	connect(OM4_3, OM4_2_F, 2, -3);
	// output to OM4
	connect(OM4_2_F, OM5_2_F, 4, 30);
	// output to IP
	connect(OM4_2_E, eIP_E, 2, 1500, neurons_in_ip);
	connect(OM4_2_F, eIP_F, 4, 5, neurons_in_ip);

	/// OM 5
	// input from EES group 5
	connect(E5, OM5_0, 3, 680);
	// input from sensory [CV]
	connect_one_to_all(CV5, OM5_0, 0.5, 1900);
	// inner connectomes
	connect(OM5_0, OM5_1, 1, 2000);
	connect(OM5_1, OM5_2_E, 1, 1500);
	connect(OM5_1, OM5_2_F, 1, 27);
	connect(OM5_1, OM5_3, 1, 400);
	connect(OM5_2_E, OM5_1, 2.5, 1500);
	connect(OM5_2_F, OM5_1, 2.5, 27);
	connect(OM5_2_E, OM5_3, 1, 400);
	connect(OM5_2_F, OM5_3, 1, 4);
	connect(OM5_3, OM5_1, 2, -1000);
	connect(OM5_3, OM5_2_E, 0.5, -1000);
	connect(OM5_3, OM5_2_F, 2, -3);
	// output to IP
	connect(OM5_2_E, eIP_E, 1, 1500, neurons_in_ip);
	connect(OM5_2_F, eIP_F, 4, 5, neurons_in_ip);

	/// reflex arc
	connect(iIP_E, eIP_F, 0.5, -10, neurons_in_ip);
	connect(iIP_F, eIP_E, 0.5, -10, neurons_in_ip);

	connect(iIP_E, OM1_2_F, 0.5, -1, neurons_in_ip);
	connect(iIP_E, OM2_2_F, 0.5, -1, neurons_in_ip);
	connect(iIP_E, OM3_2_F, 0.5, -1, neurons_in_ip);
	connect(iIP_E, OM4_2_F, 0.5, -1, neurons_in_ip);

	connect(EES, Ia_E_aff, 1, 2000);
	connect(EES, Ia_F_aff, 1, 2000);

	connect(eIP_E, MN_E, 2, 400, neurons_in_moto);
	connect(eIP_F, MN_F, 5, 8, neurons_in_moto);

	connect(iIP_E, Ia_E_pool, 1, 10, neurons_in_ip);
	connect(iIP_F, Ia_F_pool, 1, 10, neurons_in_ip);

	connect(Ia_E_pool, MN_F, 1, -4, neurons_in_ip);
	connect(Ia_E_pool, Ia_F_pool, 1, -1, neurons_in_ip);
	connect(Ia_F_pool, MN_E, 1, -4, neurons_in_ip);
	connect(Ia_F_pool, Ia_E_pool, 1, -1, neurons_in_ip);

	connect(Ia_E_aff, MN_E, 2, 2000, neurons_in_moto);
	connect(Ia_F_aff, MN_F, 2, 6, neurons_in_moto);

	connect(MN_E, R_E, 2, 1);
	connect(MN_F, R_F, 2, 1);

	connect(R_E, MN_E, 2, -5, neurons_in_moto);
	connect(R_E, R_F, 2, -10);

	connect(R_F, MN_F, 2, -5, neurons_in_moto);
	connect(R_F, R_E, 2, -10);
}

#ifdef DEBUG
void save(int test_index, GroupMetadata &metadata, const string& folder){
	if(metadata.group.group_name[0] != 'M' || metadata.group.group_name[1] != 'N')
		return;

	ofstream file;
	string file_name = "/dat/" + to_string(test_index) + "_" + metadata.group.group_name + ".dat";

	file.open(folder + file_name);
	// save voltage
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++)
		file << metadata.voltage_array[sim_iter] << " ";
	file << endl;

	// save g_exc
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++)
		file << 0 << " ";
	file << endl;

	// save g_inh
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++)
		file << 0 << " ";
	file << endl;

	// save spikes
	for (float const& value: metadata.spike_vector) {
		file << value << " ";
	}
	file.close();

	cout << "Saved to: " << folder + file_name << endl;
}

void save_result(int itest) {
	string current_path = getcwd(nullptr, 0);
	printf("Save results to: %s \n", current_path.c_str());

	for(GroupMetadata &metadata : all_groups) {
		save(itest, metadata, current_path);
	}
}
#endif

// get datasize of current variable type and its number
template <typename type>
unsigned int datasize(unsigned int size) {
	return sizeof(type) * size;
}

// fill array with current value
template <typename type>
void init_array(type *array, unsigned int size, type value) {
	for(unsigned int i = 0; i < size; i++)
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

void copy_data_to(GroupMetadata &metadata, const unsigned short* nrn_v_m, const bool *nrn_has_spike, const unsigned int sim_iter) {
	unsigned int nrn_mean_volt = 0;

	for(unsigned int tid = metadata.group.id_start; tid <= metadata.group.id_end; tid++) {
		nrn_mean_volt += nrn_v_m[tid];
		if (nrn_has_spike[tid])
			metadata.spike_vector.push_back(step_to_ms(sim_iter) + 0.25);
	}
	#ifdef DEBUG
	metadata.voltage_array[sim_iter] = 1.0f * nrn_mean_volt / metadata.group.group_size;
	#endif
}

void simulate(int itest) {
	const unsigned int neurons_number = global_id;
	const unsigned int synapses_number = static_cast<int>(all_synapses.size());
	chrono::time_point<chrono::system_clock> simulation_t_start, simulation_t_end;
	// calculate spike frequency and C0/C1 activation time in steps
	auto ees_spike_each_step = (unsigned int)(1000 / 40 / SIM_STEP);
	auto steps_activation_C0 = (unsigned int)(5 * 25 / SIM_STEP);
	auto steps_activation_C1 = (unsigned int)(6 * skin_stim_time / SIM_STEP);
	// neuron variables
	unsigned short V_m[neurons_number];      // [mV] neuron membrane potential
	unsigned short L[neurons_number];        // 
	unsigned short V_th[neurons_number];        // 
	bool nrn_has_spike[neurons_number];      // neuron state - has spike or not
	int nrn_ref_time[neurons_number];        // [step] neuron refractory time
	int nrn_ref_time_timer[neurons_number];  // [step] neuron refractory time timer

	init_array<unsigned short>(V_m, neurons_number, 28000);
	init_array<bool>(nrn_has_spike, neurons_number, false);  // by default neurons haven't spikes at start
	rand_normal_init_array<int>(nrn_ref_time, neurons_number, (int)(3 / SIM_STEP), (int)(0.4 / SIM_STEP));  // neuron ref time, aprx interval is (1.8, 4.2)
	rand_normal_init_array<unsigned short>(L, neurons_number, 500, 500 / 20);  //
	rand_normal_init_array<unsigned short>(V_th, neurons_number, 45000, 500);  //
	init_array<int>(nrn_ref_time_timer, neurons_number, 0);  // by default neurons have ref_t timers as 0

	// synapse variables
	auto *syn_delay = (int *) malloc(datasize<int>(synapses_number));
	auto *syn_delay_timer = (int *) malloc(datasize<int>(synapses_number));
	auto *syn_weight = (short *) malloc(datasize<short>(synapses_number));
	auto *syn_pre_nrn_id = (int *) malloc(datasize<int>(synapses_number));
	auto *syn_post_nrn_id = (int *) malloc(datasize<int>(synapses_number));
	init_array<int>(syn_delay_timer, synapses_number, -1);

	// fill arrays of synapses
	unsigned int syn_id = 0;
	for(SynapseMetadata metadata : all_synapses) {
		syn_pre_nrn_id[syn_id] = metadata.pre_id;
		syn_post_nrn_id[syn_id] = metadata.post_id;
		syn_delay[syn_id] = metadata.synapse_delay;
		syn_weight[syn_id] = metadata.synapse_weight;
		syn_id++;
	}
	all_synapses.clear();

	// stuff variables for controlling C0/C1 activation
	int local_iter = 0;
	bool C0_activated = false; // start from extensor
	bool C0_early_activated = false;
	unsigned int shift_time_by_step = 0;
	bool EES_activated;
	bool CV1_activated;
	bool CV2_activated;
	bool CV3_activated;
	bool CV4_activated;
	bool CV5_activated;
	unsigned int shifted_iter_time;

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

	printf("START THE MAIN SIMULATION LOOP\n");

	simulation_t_start = chrono::system_clock::now();

	// the main simulation loop
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++) {
		#ifdef DEBUG
		for (GroupMetadata &metadata : all_groups)
			copy_data_to(metadata, V_m, nrn_has_spike, sim_iter);
		#endif
		CV1_activated = false;
		CV2_activated = false;
		CV3_activated = false;
		CV4_activated = false;
		CV5_activated = false;
		EES_activated = (sim_iter % ees_spike_each_step == 0);
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
		shifted_iter_time = sim_iter - shift_time_by_step;
		// check the CV activation
		if ((begin_C_spiking[0] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[0])) CV1_activated = true;
		if ((begin_C_spiking[1] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[1])) CV2_activated = true;
		if ((begin_C_spiking[2] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[2])) CV3_activated = true;
		if ((begin_C_spiking[3] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[3])) CV4_activated = true;
		if ((begin_C_spiking[4] <= shifted_iter_time) && (shifted_iter_time < end_C_spiking[4])) CV5_activated = true;

		// update local iter (warning: can be resetted at C0/C1 activation)
		local_iter++;
		/** ================================================= **/
		/** ==================N E U R O N S================== **/
		/** ================================================= **/
		#pragma omp parallel for num_threads(4) default(shared)
		for(unsigned int tid = 0; tid < neurons_number; tid++) {
			// reset spike flag of the current neuron before calculations
			nrn_has_spike[tid] = false;
			// generating spikes for EES
			if (tid < 15 && EES_activated) nrn_has_spike[tid] = true;
			// iIP_F
			if (C0_activated && C0_early_activated && 705 <= tid && tid <= 729)
				nrn_has_spike[705 + static_cast<int>(neurons_in_ip * standard_uniform(generator))] = true;
			// skin stimulations
			if (!C0_activated) {
				if (tid == 90 && CV1_activated) nrn_has_spike[tid] = true;
				if (tid == 91 && CV2_activated) nrn_has_spike[tid] = true;
				if (tid == 92 && CV3_activated) nrn_has_spike[tid] = true;
				if (tid == 93 && CV4_activated) nrn_has_spike[tid] = true;
				if (tid == 94 && CV5_activated) nrn_has_spike[tid] = true;
			}
			// (threshold && not in refractory period) >= -55mV
			if ((V_m[tid] >= V_th[tid]) && (nrn_ref_time_timer[tid] == 0)) {
				V_m[tid] = 20000;
				nrn_has_spike[tid] = true;
				nrn_ref_time_timer[tid] = nrn_ref_time[tid];
			}
			if (V_m[tid] < 27500)
				V_m[tid] += L[tid];
			if (V_m[tid] > 28500)
				V_m[tid] -= L[tid];
			if (nrn_ref_time_timer[tid] > 0)
				nrn_ref_time_timer[tid]--;
		}
		/** ================================================= **/
		/** ==================S Y N A P S E================== **/
		/** ================================================= **/
		int post_id;
		#pragma omp parallel for num_threads(4) shared(V_m, nrn_has_spike, syn_pre_nrn_id, syn_post_nrn_id, syn_delay, syn_delay_timer, syn_weight) private(post_id)
		for(unsigned int tid = 0; tid < synapses_number; tid++) {
			// add synaptic delay if neuron has spike
			if (syn_delay_timer[tid] == -1 && nrn_has_spike[syn_pre_nrn_id[tid]])
				syn_delay_timer[tid] = syn_delay[tid];
			// if synaptic delay is zero it means the time when synapse increase I by synaptic weight
			if (syn_delay_timer[tid] == 0) {
				post_id = syn_post_nrn_id[tid];
				// post neuron ID = syn_post_nrn_id[syn_id], thread-safe (!)
				#pragma omp atomic
				V_m[post_id] += syn_weight[tid];
				if (V_m[post_id] < 20000)
					V_m[post_id] = 20000;
				if (V_m[post_id] > 50000)
					V_m[post_id] = 50000;
				// make synapse timer a "free" for next spikes
				syn_delay_timer[tid] = -1;
			}
			// update synapse delay timer
			if (syn_delay_timer[tid] > 0)
				syn_delay_timer[tid]--;
		}
	} // end of the simulation iteration loop
	simulation_t_end = chrono::system_clock::now();

	#ifdef DEBUG
	save_result(itest);
	#endif

	auto sim_time_diff = chrono::duration_cast<chrono::milliseconds>(simulation_t_end - simulation_t_start).count();
	printf("Elapsed %li ms (measured) | T_sim = %d ms\n", sim_time_diff, T_simulation);
	printf("%s x%f\n", 1.0 * T_simulation / sim_time_diff > 1? "faster" : "slower", (float)T_simulation / sim_time_diff);
}

// runner
int main(int argc, char* argv[]) {
	int itest = atoi(argv[1]);
	init_network();
	simulate(itest);

	return 0;
}
