#include <map>
#include <string>
#include <vector>
#include <iostream>

class Group {
public:
	Group() = default;
	std::string group_name;
	unsigned int id_start{};
	unsigned int id_end{};
	unsigned int group_size{};
	unsigned int time{};
};

// struct for human-readable initialization of connectomes
struct GroupMetadata {
	Group group;
	float *g_exc;                // [nS] array of excitatory conductivity
	float *g_inh;                // [nS] array of inhibition conductivity
	float *voltage_array;        // [mV] array of membrane potential
	std::vector<float> spike_vector;  // [ms] spike times

	explicit GroupMetadata(Group group) {
		this->group = std::move(group);
		voltage_array = new float[group.time];
		g_exc = new float[group.time];
		g_inh = new float[group.time];
	}
};

// common neuron's parameters
// also from https://www.cell.com/neuron/pdfExtended/S0896-6273(16)00010-6
struct Parameters {
	short *nrn_start_seg;   // [index]
	char *models;           // [str] model's names
	float *Cm;              // [uF / cm2] membrane capacitance
	float *gnabar;          // [S / cm2] the maximal fast Na+ conductance
	float *gkbar;           // [S / cm2] the maximal slow K+ conductance
	float *gl;              // [S / cm2] the maximal leak conductance
	float *Ra;              // [Ohm cm] axoplasmic resistivity
	float *diam;            // [um] soma compartment diameter
	float *length;          // [um] soma compartment length
	float *ena;             // [mV] Na+ reversal (equilibrium, Nernst) potential
	float *ek;              // [mV] K+ reversal (equilibrium, Nernst) potential
	float *el;              // [mV] Leakage reversal (equilibrium) potential
	// moto neuron's properties
	// https://senselab.med.yale.edu/modeldb/showModel.cshtml?model=189786
	// https://journals.physiology.org/doi/pdf/10.1152/jn.2002.88.4.1592
	float *gkrect;          // [S / cm2] the maximal delayed rectifier K+ conductance
	float *gcaN;            // [S / cm2] the maximal N-type Ca2+ conductance
	float *gcaL;            // [S / cm2] the maximal L-type Ca2+ conductance
	float *gcak;            // [S / cm2] the maximal Ca2+ activated K+ conductance
	// synapses' parameters
	float *E_ex;            // [mV] excitatory reversal (equilibrium) potential
	float *E_inh;           // [mV] inhibitory reversal (equilibrium) potential
	float *tau_exc;         // [ms] rise time constant of excitatory synaptic conductance
	float *tau_inh1;        // [ms] rise time constant of inhibitory synaptic conductance
	float *tau_inh2;        // [ms] decay time constant of inhibitory synaptic conductance
	int size;
};

// neuron states
struct States {
	float *Vm;              // [mV] array for three compartments volatge
	float *n;               // [0..1] compartments channel, providing the kinetic pattern of the L conductance
	float *m;               // [0..1] compartments channel, providing the kinetic pattern of the Na conductance
	float *h;               // [0..1] compartments channel, providing the kinetic pattern of the Na conductance
	float *l;               // [0..1] inward rectifier potassium (Kir) channel
	float *s;               // [0..1] nodal slow potassium channel
	float *p;               // [0..1] compartments channel, providing the kinetic pattern of the ?? conductance
	float *hc;              // [0..1] compartments channel, providing the kinetic pattern of the ?? conductance
	float *mc;              // [0..1] compartments channel, providing the kinetic pattern of the ?? conductance
	float *cai;             //
	float *I_Ca;            // [nA] Ca ionic currents
	float *NODE_A;          // the effect of this node on the parent node's equation
	float *NODE_B;          // the effect of the parent node on this node's equation
	float *NODE_D;          // diagonal element in node equation
	float *const_NODE_D;    // const diagonal element in node equation (performance)
	float *NODE_RHS;        // right hand side in node equation
	float *NODE_RINV;       // conductance uS from node to parent
	float *NODE_AREA;       // area of a node in um^2
	int size;
};

struct Neurons {
	bool *has_spike;        // spike flag for each neuron
	bool *spike_on;         // special flag to prevent fake spike detecting
	float *g_exc;           // [S] excitatory conductivity level
	float *g_inh_A;         // [S] inhibitory conductivity level
	float *g_inh_B;         // [S] inhibitory conductivity level
	float *factor;          // [const] todo
	int size;
};

struct Synapses {
	int *syn_pre_nrn;       // [id] list of pre neurons ids
	int *syn_post_nrn;      // [id] list of pre neurons ids
	float *syn_weight;      // [S] list of synaptic weights
	int *syn_delay;         // [ms * dt] list of synaptic delays in steps
	int *syn_delay_timer;   // [ms * dt] list of synaptic timers, shows how much left to send signal
	int size;   //
};