/**
See the topology https://github.com/research-team/memristive-spinal-cord/blob/master/doc/diagram/cpg_generator_FE_paper.png
Based on the NEURON repository.
*/
#include <algorithm>
#include <utility>
#include <ctime>
#include <random>
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include "test.h"
#include <stdexcept>
// for file writing
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <unistd.h>
#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )
#define PI 3.141592654f

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

using namespace std;

const double dt = 0.025;      // [ms] simulation step
const int sim_time = 100;    // [ms] simulation time
const auto SIM_TIME_IN_STEPS = (unsigned int)(sim_time / dt);  // [steps] converted time into steps

const bool EXTRACELLULAR = false;
const char GENERATOR = 'g';
const char INTER = 'i';
const char MOTO = 'm';
const char MUSCLE = 'u';

const int skin_time = 25;   // duration of layer 25 = 21 cm/s; 50 = 15 cm/s; 125 = 6 cm/s
const int step_number = 2;  // number of steps
const int cv_fr = 200;      // frequency of CV
const int ees_fr = 40;      // frequency of EES

unsigned int one_step_time = 6 * skin_time + 175;
unsigned int time_sim = 25 + one_step_time * step_number;

random_device r;
default_random_engine generator(r());

unsigned int nrns_number = 0;        // [id] global neuron id = number of neurons
unsigned int nrns_and_segs = 0;      // [id] global neuron+segs id = number of neurons with segments
//const int LEG_STEPS = 1;             // [step] number of full cycle steps

const int neurons_in_group = 50;     // number of neurons in a group
const int neurons_in_ip = 196;       // number of neurons in a group


/*
EES_stimulus = (np.arange(0, sim_time, ees_int) / dt).astype(int)
CV1_stimulus = (np.arange(skin_time * 0, skin_time * 1, random.gauss(cv_int, cv_int / 10)) / dt).astype(int)
CV2_stimulus = (np.arange(skin_time * 1, skin_time * 2, random.gauss(cv_int, cv_int / 10)) / dt).astype(int)
CV3_stimulus = (np.arange(skin_time * 2, skin_time * 3, random.gauss(cv_int, cv_int / 10)) / dt).astype(int)
CV4_stimulus = (np.arange(skin_time * 3, skin_time * 5, random.gauss(cv_int, cv_int / 10)) / dt).astype(int)
CV5_stimulus = (np.arange(skin_time * 5, skin_time * 6, random.gauss(cv_int, cv_int / 10)) / dt).astype(int)
*/

/*
# arrays for saving data
spikes = []             # saved spikes
GRAS_data = []          # saved gras data (DEBUGGING)
save_groups = []        # neurons groups that need to save
saved_voltage = []      # saved voltage
save_neuron_ids = []    # neurons id that need to save
 */

// common neuron constants
const double k = 0.01;            // synaptic coef
const double V_th = -40;          // [mV] voltage threshold
const double V_adj = -63;         // [mV] adjust voltage for -55 threshold
// moto neuron constants
const double ca0 = 2;             // initial calcium concentration
const double amA = 0.4;           // const ??? todo
const double amB = 66;            // const ??? todo
const double amC = 5;             // const ??? todo
const double bmA = 0.4;           // const ??? todo
const double bmB = 32;            // const ??? todo
const double bmC = 5;             // const ??? todo
const double R_const = 8.314472;  // [k-mole] or [joule/degC] const
const double F_const = 96485.34;  // [faraday] or [kilocoulombs] const
// muscle fiber constants
//const double g_kno = 0.01;        // [S/cm2] conductance of the todo
//const double g_kir = 0.03;        // [S/cm2] conductance of the Inwardly Rectifying Potassium K+ (Kir) channel
// Boltzman steady state curve
const double vhalfl = -98.92;     // [mV] inactivation half-potential
const double kl = 10.89;          // [mV] Stegen et al. 2012
// tau_infty
const double vhalft = 67.0828;    // [mV] fitted //100 uM sens curr 350a, Stegen et al. 2012
const double at = 0.00610779;     // [/ ms] Stegen et al. 2012
const double bt = 0.0817741;      // [/ ms] Note: typo in Stegen et al. 2012
// temperature dependence
const double q10 = 1;             // temperature scaling (sensitivity)
const double celsius = 36;        // [degC] temperature of the cell
// i_membrane [mA/cm2]
//const double e_extracellular = 0; // [mV]
//const double xraxial = 1e9;       // [MOhm/cm]

// neuron parameters
vector<unsigned int> vector_nrn_start_seg;
vector<char> vector_models;
vector<double> vector_Cm;
vector<double> vector_gnabar;
vector<double> vector_gkbar;
vector<double> vector_gl;
vector<double> vector_Ra;
vector<double> vector_diam;
vector<double> vector_length;
vector<double> vector_ena;
vector<double> vector_ek;
vector<double> vector_el;
vector<double> vector_gkrect;
vector<double> vector_gcaN;
vector<double> vector_gcaL;
vector<double> vector_gcak;
// synaptic parameters
vector<double> vector_E_ex;
vector<double> vector_E_inh;
vector<double> vector_tau_exc;
vector<double> vector_tau_inh1;
vector<double> vector_tau_inh2;
// synapses varaibels
vector<int> vector_syn_pre_nrn;       // [id] list of pre neurons ids
vector<int> vector_syn_post_nrn;      // [id] list of pre neurons ids
vector<double> vector_syn_weight;      // [S] list of synaptic weights
vector<int> vector_syn_delay;         // [ms * dt] list of synaptic delays in steps
vector<int> vector_syn_delay_timer;   // [ms * dt] list of synaptic timers, shows how much left to send signal
// results vector
vector <GroupMetadata> saving_groups; //

// form structs of neurons global ID and groups name
Group form_group(const string &group_name,
				 int nrns_in_group = neurons_in_group,
				 const char model = INTER,
				 const int segs = 1) {
	Group group = Group();
	group.group_name = group_name;     // name of a neurons group
	group.id_start = nrns_number;        // first ID in the group
	group.id_end = nrns_number + nrns_in_group - 1;  // the latest ID in the group
	group.group_size = nrns_in_group;  // size of the neurons group

	double __Cm;
	double __gnabar;
	double __gkbar;
	double __gl;
	double __Ra;
	double __ena;
	double __ek;
	double __el;
	double __diam;
	double __dx;
	double __gkrect;
	double __gcaN;
	double __gcaL;
	double __gcak;
	double __e_ex;
	double __e_inh;
	double __tau_exc;
	double __tau_inh1;
	double __tau_inh2;
	normal_distribution<double> Cm_distr(1, 0.01);
//	uniform_real_distribution<double> moto_diam_distr(45, 55);
	uniform_int_distribution<int> moto_diam_distr(45, 55);

	for (int nrn = 0; nrn < nrns_in_group; nrn++) {
		if (model == INTER) {
			__Cm = Cm_distr(generator);
			__gnabar = 0.1;
			__gkbar = 0.08;
			__gl = 0.002;
			__Ra = 100.0;
			__ena = 50.0;
			__ek = -90.0;
			__el = -70.0;
			__diam = 10.0; // random.randint(5, 15);
			__dx = __diam;
			__e_ex = 50;
			__e_inh = -80;
			__tau_exc = 0.35;
			__tau_inh1 = 0.5;
			__tau_inh2 = 3.5;
		} else if (model == MOTO) {
			__Cm = 2;
			__gnabar = 0.05;
			__gl = 0.002;
			__Ra = 200.0;
			__ena = 50.0;
			__ek = -80.0;
			__el = -70.0;
			__diam = moto_diam_distr(generator); //random.randint(45, 55);
			__dx = __diam;
			__gkrect = 0.3;
			__gcaN = 0.05;
			__gcaL = 0.0001;
			__gcak = 0.3;
			__e_ex = 50.0;
			__e_inh = -80.0;
			__tau_exc = 0.3;
			__tau_inh1 = 1.0;
			__tau_inh2 = 1.5;
			if (__diam > 50) {
				__gnabar = 0.1;
				__gcaL = 0.001;
				__gl = 0.003;
				__gkrect = 0.2;
				__gcak = 0.2;
			}
		} else if (model == MUSCLE) {
			__Cm = 3.6;
			__gnabar = 0.15;
			__gkbar = 0.03;
			__gl = 0.0002;
			__Ra = 1.1;
			__ena = 55.0;
			__ek = -80.0;
			__el = -72.0;
			__diam = 40.0;
			__dx = 3000.0;
			__e_ex = 0.0;
			__e_inh = -80.0;
			__tau_exc = 0.3;
			__tau_inh1 = 1.0;
			__tau_inh2 = 1.0;
		} else if (model == GENERATOR) {

		} else {
			throw logic_error("Choose the model");
		}
		// common properties
		vector_Cm.push_back(__Cm);
		vector_gnabar.push_back(__gnabar);
		vector_gkbar.push_back(__gkbar);
		vector_gl.push_back(__gl);
		vector_el.push_back(__el);
		vector_ena.push_back(__ena);
		vector_ek.push_back(__ek);
		vector_Ra.push_back(__Ra);
		vector_diam.push_back(__diam);
		vector_length.push_back(__dx);
		vector_gkrect.push_back(__gkrect);
		vector_gcaN.push_back(__gcaN);
		vector_gcaL.push_back(__gcaL);
		vector_gcak.push_back(__gcak);
		vector_E_ex.push_back(__e_ex);
		vector_E_inh.push_back(__e_inh);
		vector_tau_exc.push_back(__tau_exc);
		vector_tau_inh1.push_back(__tau_inh1);
		vector_tau_inh2.push_back(__tau_inh2);
		//
		vector_nrn_start_seg.push_back(nrns_and_segs);
		nrns_and_segs += (segs + 2);
		vector_models.push_back(model);
	}

	nrns_number += nrns_in_group;
	printf("Formed %s IDs [%d ... %d] = %d\n",
		group_name.c_str(), nrns_number - nrns_in_group, nrns_number - 1, nrns_in_group);

	return group;
}

__host__
unsigned int ms_to_step(double ms) { return (unsigned int) (ms / dt); }

__host__
double step_to_ms(int step) { return step * dt; }

// copy data from host to device
template<typename type>
void memcpyHtD(type *host, type *gpu, unsigned int size) {
	HANDLE_ERROR(cudaMemcpy(gpu, host, sizeof(type) * size, cudaMemcpyHostToDevice));
}

// copy data from device to host
template<typename type>
void memcpyDtH(type *gpu, type *host, unsigned int size) {
	HANDLE_ERROR(cudaMemcpy(host, gpu, size * sizeof(type), cudaMemcpyDeviceToHost));
}

template<typename type>
type* init_gpu_arr(type *cpu_var, unsigned int size = nrns_and_segs) {
	type *gpu_var;
	HANDLE_ERROR(cudaMalloc(&gpu_var, size * sizeof(type)));
	memcpyHtD<type>(cpu_var, gpu_var, size);
	return gpu_var;
}

template<typename type>
type *init_gpu_arr(vector<type> &vec) {
	type *gpu_var;
	HANDLE_ERROR(cudaMalloc(&gpu_var, sizeof(type) * vec.size()));
	memcpyHtD<type>(vec.data(), gpu_var, vec.size());
	return gpu_var;
}

template<typename type>
type* vec2arr(vector<type> &vec) {
	return vec.cpu_vector.data();
}
__device__
double Exp(double volt) {
	return (volt < -100)? 0 : exp(volt);
}

__device__
double alpham(double volt) {
	if (abs((volt + amB) / amC) < 1e-6)
		return amA * amC;
	return amA * (volt + amB) / (1.0 - Exp(-(volt + amB) / amC));
}

__device__
double betam(double volt) {
	if (abs((volt + bmB) / bmC) < 1e-6)
		return -bmA * bmC;
	return -bmA * (volt + bmB) / (1.0 - Exp((volt + bmB) / bmC));
}

__device__
double syn_current(Neurons* N, Parameters* P, int nrn, double voltage) {
	/**
	calculate synaptic current
	*/
	return N->g_exc[nrn] * (voltage - P->E_ex[nrn]) + (N->g_inh_B[nrn] - N->g_inh_A[nrn]) * (voltage - P->E_inh[nrn]);
}

__device__
double nrn_moto_current(States* S, Parameters* P, Neurons* N, int nrn, int nrn_seg_index, double voltage) {
	/**
	calculate channels current
	*/
	double iNa = P->gnabar[nrn] * pow(S->m[nrn_seg_index], 3) * S->h[nrn_seg_index] * (voltage - P->ena[nrn]);
	double iK = P->gkrect[nrn] * pow(S->n[nrn_seg_index], 4) * (voltage - P->ek[nrn]) +
                P->gcak[nrn] * pow(S->cai[nrn_seg_index], 2) / (pow(S->cai[nrn_seg_index], 2) + 0.014 * 0.014) * (voltage - P->ek[nrn]);
	double iL = P->gl[nrn] * (voltage - P->el[nrn]);
	double eCa = (1000 * R_const * 309.15 / (2 * F_const)) * log(ca0 / S->cai[nrn_seg_index]);
	S->I_Ca[nrn_seg_index] = P->gcaN[nrn] * pow(S->mc[nrn_seg_index], 2) * S->hc[nrn_seg_index] * (voltage - eCa) +
	                         P->gcaL[nrn] * S->p[nrn_seg_index] * (voltage - eCa);
	return iNa + iK + iL + S->I_Ca[nrn_seg_index];
}

__device__
double nrn_fastchannel_current(States* S, Parameters* P, Neurons* N, int nrn, int nrn_seg_index, double voltage) {
	/**
	calculate channels current
	*/
	double iNa = P->gnabar[nrn] * pow(S->m[nrn_seg_index], 3) * S->h[nrn_seg_index] * (voltage - P->ena[nrn]);
	double iK = P->gkbar[nrn] * pow(S->n[nrn_seg_index], 4) * (voltage - P->ek[nrn]);
	double iL = P->gl[nrn] * (voltage - P->el[nrn]);
	return iNa + iK + iL;
}

__device__
void recalc_synaptic(States* S, Parameters* P, Neurons* N, int nrn) {
	/**
	updating conductance(summed) of neurons' post-synaptic conenctions
	*/
	// exc synaptic conductance
	if (N->g_exc[nrn] != 0) {
		N->g_exc[nrn] -= (1.0 - exp(-dt / P->tau_exc[nrn])) * N->g_exc[nrn];
		if (N->g_exc[nrn] < 1e-5) {
			N->g_exc[nrn] = 0.0;
		}
	}
	// inh1 synaptic conductance
	if (N->g_inh_A[nrn] != 0) {
		N->g_inh_A[nrn] -= (1.0 - exp(-dt / P->tau_inh1[nrn])) * N->g_inh_A[nrn];
		if (N->g_inh_A[nrn] < 1e-5) {
			N->g_inh_A[nrn] = 0.0;
		}
	}
	// inh2 synaptic conductance
	if (N->g_inh_B[nrn] != 0) {
		N->g_inh_B[nrn] -= (1.0 - exp(-dt / P->tau_inh2[nrn])) * N->g_inh_B[nrn];
		if (N->g_inh_B[nrn] < 1e-5)
			N->g_inh_B[nrn] = 0.0;
	}
}

__device__
void syn_initial(States* S, Parameters* P, Neurons* N, int nrn) {
	/**
	initialize tau(rise / decay time, ms) and factor(const) variables
	*/
	if (P->tau_inh1[nrn] / P->tau_inh2[nrn] > 0.9999)
		P->tau_inh1[nrn] = 0.9999 * P->tau_inh2[nrn];
	if (P->tau_inh1[nrn] / P->tau_inh2[nrn] < 1e-9)
		P->tau_inh1[nrn] = P->tau_inh2[nrn] * 1e-9;
	//
	double tp = (P->tau_inh1[nrn] * P->tau_inh2[nrn]) / (P->tau_inh2[nrn] - P->tau_inh1[nrn]) *
	           log(P->tau_inh2[nrn] / P->tau_inh1[nrn]);
	N->factor[nrn] = -exp(-tp / P->tau_inh1[nrn]) + exp(-tp / P->tau_inh2[nrn]);
	N->factor[nrn] = 1.0 / N->factor[nrn];
}

__device__
void nrn_inter_initial(States* S, Parameters* P, Neurons* N, int nrn_seg_index, double V) {
	/**
	initialize channels, based on cropped evaluate_fct function
	*/
	double V_mem = V - V_adj;
	//
	double a = 0.32 * (13.0 - V_mem) / (exp((13.0 - V_mem) / 4.0) - 1.0);
	double b = 0.28 * (V_mem - 40.0) / (exp((V_mem - 40.0) / 5.0) - 1.0);
	S->m[nrn_seg_index] = a / (a + b);   // m_inf
	//
	a = 0.128 * exp((17.0 - V_mem) / 18.0);
	b = 4.0 / (1.0 + exp((40.0 - V_mem) / 5.0));
	S->h[nrn_seg_index] = a / (a + b);   // h_inf
	//
	a = 0.032 * (15.0 - V_mem) / (exp((15.0 - V_mem) / 5.0) - 1.0);
	b = 0.5 * exp((10.0 - V_mem) / 40.0);
	S->n[nrn_seg_index] = a / (a + b);   // n_inf
}

__device__
void nrn_moto_initial(States* S, Parameters* P, Neurons* N, int nrn_seg_index, double V) {
	/** initialize channels, based on cropped evaluate_fct function */
	double a = alpham(V);
	S->m[nrn_seg_index] = a / (a + betam(V));                         // m_inf
	S->h[nrn_seg_index] = 1.0 / (1.0 + Exp((V + 65.0) / 7.0));   // h_inf
	S->p[nrn_seg_index] = 1.0 / (1.0 + Exp(-(V + 55.8) / 3.7));  // p_inf
	S->n[nrn_seg_index] = 1.0 / (1.0 + Exp(-(V + 38.0) / 15.0)); // n_inf
	S->mc[nrn_seg_index] = 1.0 / (1.0 + Exp(-(V + 32.0) / 5.0)); // mc_inf
	S->hc[nrn_seg_index] = 1.0 / (1.0 + Exp((V + 50.0) / 5.0));  // hc_inf
	S->cai[nrn_seg_index] = 0.0001;
}

__device__
void nrn_muslce_initial(States* S, Parameters* P, Neurons* N, int nrn_seg_index, double V) {
	/**
	initialize channels, based on cropped evaluate_fct function
	*/
	double V_mem = V - V_adj;
	//
	double a = 0.32 * (13.0 - V_mem) / (exp((13.0 - V_mem) / 4.0) - 1.0);
	double b = 0.28 * (V_mem - 40.0) / (exp((V_mem - 40.0) / 5.0) - 1.0);
	S->m[nrn_seg_index] = a / (a + b);   // m_inf
	//
	a = 0.128 * exp((17.0 - V_mem) / 18.0);
	b = 4.0 / (1.0 + exp((40.0 - V_mem) / 5.0));
	S->h[nrn_seg_index] = a / (a + b);   // h_inf
	//
	a = 0.032 * (15.0 - V_mem) / (exp((15.0 - V_mem) / 5.0) - 1.0);
	b = 0.5 * exp((10.0 - V_mem) / 40.0);
	S->n[nrn_seg_index] = a / (a + b);   // n_inf
}

__device__
void recalc_inter_channels(States* S, Parameters* P, Neurons* N, int nrn_seg_index, double V) {
	/** calculate new states of channels (evaluate_fct) */
	// BREAKPOINT -> states -> evaluate_fct
	double V_mem = V - V_adj;
	//
	double a = 0.32 * (13.0 - V_mem) / (exp((13.0 - V_mem) / 4.0) - 1.0);
	double b = 0.28 * (V_mem - 40.0) / (exp((V_mem - 40.0) / 5.0) - 1.0);
	double tau = 1.0 / (a + b);
	double inf = a / (a + b);
	S->m[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->m[nrn_seg_index]);
	//
	a = 0.128 * exp((17.0 - V_mem) / 18.0);
	b = 4.0 / (1.0 + exp((40.0 - V_mem) / 5.0));
	tau = 1.0 / (a + b);
	inf = a / (a + b);
	S->h[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->h[nrn_seg_index]);
	//
	a = 0.032 * (15.0 - V_mem) / (exp((15.0 - V_mem) / 5.0) - 1.0);
	b = 0.5 * exp((10.0 - V_mem) / 40.0);
	tau = 1.0 / (a + b);
	inf = a / (a + b);
	// states
	S->n[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->n[nrn_seg_index]);
}

__device__
void recalc_moto_channels(States* S, Parameters* P, Neurons* N, int nrn_seg_index, double V) {
	/** calculate new states of channels (evaluate_fct) */
	// BREAKPOINT -> states -> evaluate_fct
	double a = alpham(V);
	double b = betam(V);
	// m
	double tau = 1.0 / (a + b);
	double inf = a / (a + b);
	S->m[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->m[nrn_seg_index]);
	// h
	tau = 30.0 / (Exp((V + 60.0) / 15.0) + Exp(-(V + 60.0) / 16.0));
	inf = 1.0 / (1 + Exp((V + 65.0) / 7.0));
	S->h[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->h[nrn_seg_index]);
	// DELAYED RECTIFIER POTASSIUM
	tau = 5.0 / (Exp((V + 50.0) / 40.0) + Exp(-(V + 50.0) / 50.0));
	inf = 1.0 / (1.0 + Exp(-(V + 38.0) / 15.0));
	S->n[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->n[nrn_seg_index]);
	// CALCIUM DYNAMICS N-type
	double mc_inf = 1.0 / (1.0 + Exp(-(V + 32.0) / 5.0));
	double hc_inf = 1.0 / (1.0 + Exp((V + 50.0) / 5.0));
	// CALCIUM DYNAMICS L-type
	tau = 400.0;
	inf = 1.0 / (1.0 + Exp(-(V + 55.8) / 3.7));
	S->p[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->p[nrn_seg_index]);
	// states
	S->mc[nrn_seg_index] += (1.0 - exp(-dt / 15.0)) * (mc_inf - S->mc[nrn_seg_index]);     // tau_mc = 15
	S->hc[nrn_seg_index] += (1.0 - exp(-dt / 50.0)) * (hc_inf - S->hc[nrn_seg_index]);     // tau_hc = 50
	S->cai[nrn_seg_index] += (1.0 - exp(-dt * 0.04)) * (-0.01 * S->I_Ca[nrn_seg_index] / 0.04 - S->cai[nrn_seg_index]);
}

__device__
void recalc_muslce_channels(States* S, Parameters* P, Neurons* N, int nrn_seg_index, double V) {
	/** calculate new states of channels (evaluate_fct) */
	// BREAKPOINT -> states -> evaluate_fct
	double V_mem = V - V_adj;
	//
	double a = 0.32 * (13.0 - V_mem) / (exp((13.0 - V_mem) / 4.0) - 1.0);
	double b = 0.28 * (V_mem - 40.0) / (exp((V_mem - 40.0) / 5.0) - 1.0);
	double tau = 1.0 / (a + b);
	double inf = a / (a + b);
	S->m[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->m[nrn_seg_index]);
	//
	a = 0.128 * exp((17.0 - V_mem) / 18.0);
	b = 4.0 / (1.0 + exp((40.0 - V_mem) / 5.0));
	tau = 1.0 / (a + b);
	inf = a / (a + b);
	S->h[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->h[nrn_seg_index]);
	//
	a = 0.032 * (15.0 - V_mem) / (exp((15.0 - V_mem) / 5.0) - 1.0);
	b = 0.5 * exp((10.0 - V_mem) / 40.0);
	tau = 1.0 / (a + b);
	inf = a / (a + b);
	S->n[nrn_seg_index] += (1.0 - exp(-dt / tau)) * (inf - S->n[nrn_seg_index]);
	//
	double qt = pow(q10, (celsius - 33.0) / 10.0);
	double linf = 1.0 / (1.0 + exp((V - vhalfl) / kl)); // l_steadystate
	double taul = 1.0 / (qt * (at * exp(-V / vhalft) + bt * exp(V / vhalft)));
	double alpha = 0.3 / (1.0 + exp((V + 43.0) / -5.0));
	double beta = 0.03 / (1.0 + exp((V + 80.0) / -1.0));
	double stau = 1.0 / (alpha + beta);
	double sinf = alpha / (alpha + beta);
	// states
	S->l[nrn_seg_index] += (1.0 - exp(-dt / taul)) * (linf - S->l[nrn_seg_index]);
	S->s[nrn_seg_index] += (1.0 - exp(-dt / stau)) * (sinf - S->s[nrn_seg_index]);
}

__device__
void nrn_rhs_ext(int nrn) {

}

__device__
void nrn_setup_ext(int nrn) {

}

__device__
void nrn_update_2d(int nrn) {

}

__device__
void nrn_rhs(States* S, Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	void nrn_rhs(NrnThread *_nt) combined with the first part of nrn_lhs
	calculate right hand side of
	cm*dvm/dt = -i(vm) + is(vi) + ai_j*(vi_j - vi)
	cx*dvx/dt - cm*dvm/dt = -gx*(vx - ex) + i(vm) + ax_j*(vx_j - vx)
	This is a common operation for fixed step, cvode, and daspk methods
	*/
	// init _rhs and _lhs (NODE_D) as zero
	for (int i = i1; i < i3; ++i) {
		S->NODE_RHS[i] = 0.0;
		S->NODE_D[i] = 0.0;
//		ext_rhs[i1:i3, :] = 0
	}

	// update MOD rhs, CAPS has no current [CAP MOD CAP]!
	int center_segment = i1 + ((P->models[nrn] == MUSCLE)? 2 : 1);
	// update segments except CAPs
	double V, _g, _rhs;
	for (int nrn_seg = i1 + 1; nrn_seg < i3 - 1; ++nrn_seg) {
		V = S->Vm[nrn_seg];
		// SYNAPTIC update
		if (nrn_seg == center_segment) {
			// static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type)
			_g = syn_current(N, P, nrn, V + 0.001);
			_rhs = syn_current(N, P, nrn, V);
			_g = (_g - _rhs) / 0.001;
			_g *= 1.e2 / S->NODE_AREA[nrn_seg];
			_rhs *= 1.e2 / S->NODE_AREA[nrn_seg];
			S->NODE_RHS[nrn_seg] -= _rhs;
			// static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type)
			S->NODE_D[nrn_seg] += _g;
		}
		// NEURON update
		// static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type)
		if (P->models[nrn] == INTER) {
			// muscle and inter has the same fast_channel function
			_g = nrn_fastchannel_current(S, P, N, nrn, nrn_seg, V + 0.001);
			_rhs = nrn_fastchannel_current(S, P, N, nrn, nrn_seg, V);
		} else if (P->models[nrn] == MOTO) {
			_g = nrn_moto_current(S, P, N, nrn, nrn_seg, V + 0.001);
			_rhs = nrn_moto_current(S, P, N, nrn, nrn_seg, V);
		} else if (P->models[nrn] == MUSCLE) {
			// muscle and inter has the same fast_channel function
			_g = nrn_fastchannel_current(S, P, N, nrn, nrn_seg, V + 0.001);
			_rhs = nrn_fastchannel_current(S, P, N, nrn, nrn_seg, V);
		} else {
			// todo
		}
		// save data like in NEURON (after .mod nrn_cur)
		_g = (_g - _rhs) / 0.001;
		S->NODE_RHS[nrn_seg] -= _rhs;
		// static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type)
		S->NODE_D[nrn_seg] += _g;
	} // end FOR segments
	// activsynapse_rhs()
	if (EXTRACELLULAR) {
		// Cannot have any axial terms yet so that i(vm) can be calculated from
		// i(vm)+is(vi) and is(vi) which are stored in rhs vector.
		nrn_rhs_ext(nrn);
		// nrn_rhs_ext has also computed the the internal axial current for those
		// nodes containing the extracellular mechanism
	}
	// activstim_rhs()
	// activclamp_rhs()

	// todo: always 0, because Vm0 = Vm1 = Vm2 at [CAP node CAP] model (1 section)
	double dv;
	for (int nrn_seg = i1 + 1; nrn_seg < i3; ++nrn_seg) {
		dv = S->Vm[nrn_seg - 1] - S->Vm[nrn_seg];
		// our connection coefficients are negative so
		S->NODE_RHS[nrn_seg] -= S->NODE_B[nrn_seg] * dv;
		S->NODE_RHS[nrn_seg - 1] += S->NODE_A[nrn_seg] * dv;
	}
}

__device__
void bksub(States* S, Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	void bksub(NrnThread* _nt)
	*/
	// intracellular
	S->NODE_RHS[i1] /= S->NODE_D[i1];
	//
	for (int nrn_seg = i1 + 1; nrn_seg < i3; ++nrn_seg) {
		S->NODE_RHS[nrn_seg] -= S->NODE_B[nrn_seg] * S->NODE_RHS[nrn_seg - 1];
		S->NODE_RHS[nrn_seg] /= S->NODE_D[nrn_seg];
	}
	// extracellular
	if (EXTRACELLULAR) {
	//	for j in range(nlayer):
	//	ext_rhs[i1, j] /= ext_d[i1, j]
	//	for nrn_seg in range(i1 + 1, i3):
	//	for j in range(nlayer):
	//	ext_rhs[nrn_seg, j] -= ext_b[nrn_seg, j] * ext_rhs[nrn_seg - 1, j]
	//	ext_rhs[nrn_seg, j] /= ext_d[nrn_seg, j]
	}
}

__device__
void triang(States* S, Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	void triang(NrnThread* _nt)
	*/
	// intracellular
	double ppp;
	int nrn_seg = i3 - 1;
	while (nrn_seg >= i1 + 1) {
		ppp = S->NODE_A[nrn_seg] / S->NODE_D[nrn_seg];
		S->NODE_D[nrn_seg - 1] -= ppp * S->NODE_B[nrn_seg];
		S->NODE_RHS[nrn_seg - 1] -= ppp * S->NODE_RHS[nrn_seg];
		nrn_seg--;
	}
	// extracellular
	if (EXTRACELLULAR) {
//		nrn_seg = i3 - 1
//		while nrn_seg >= i1 + 1:
//			for j in range(nlayer):
//				ppp = ext_a[nrn_seg, j] / ext_d[nrn_seg, j]
//				ext_d[nrn_seg - 1, j] -= ppp * ext_b[nrn_seg, j]
//				ext_rhs[nrn_seg - 1, j] -= ppp * ext_rhs[nrn_seg, j]
//			nrn_seg--
	}
}

__device__
void nrn_solve(States* S, Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	void nrn_solve(NrnThread* _nt)
	*/
	triang(S, P, N, nrn, i1, i3);
	bksub(S, P, N, nrn, i1, i3);
}

__device__
void setup_tree_matrix(States* S, Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/** void setup_tree_matrix(NrnThread* _nt) */
	nrn_rhs(S, P, N, nrn, i1, i3);
	// simplified nrn_lhs(nrn)
	for (int i = i1; i < i3; ++i) {
		S->NODE_D[i] += S->const_NODE_D[i];
	}
}

__device__
void update(States* S, Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/** void update(NrnThread* _nt) */
	// final voltage updating
	for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
		S->Vm[nrn_seg] += S->NODE_RHS[nrn_seg];
	}
	// save data like in NEURON (after .mod nrn_cur)
//	if DEBUG and nrn in save_neuron_ids:
//	save_data()
	// extracellular
	nrn_update_2d(nrn);
}

__device__
void nrn_deliver_events(States* S, Parameters* P, Neurons* N, int nrn) {
/** void nrn_deliver_events(NrnThread* nt) */
	// get the central segment (for detecting spikes): i1 + (2 or 1)
	int seg_update = P->nrn_start_seg[nrn] + ((P->models[nrn] == MUSCLE)? 2 : 1);
	// check if neuron has spike with special flag for avoidance multi-spike detecting
	if (!N->spike_on[nrn] && S->Vm[seg_update] > V_th) {
		N->spike_on[nrn] = true;
		N->has_spike[nrn] = true;
	} else if (S->Vm[seg_update] < V_th) {
		N->spike_on[nrn] = false;
	}
}

__device__
void nrn_fixed_step_lastpart(States* S, Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	void *nrn_fixed_step_lastpart(NrnThread *nth)
	*/
	// update neurons' synapses state
	recalc_synaptic(S, P, N, nrn);
	//  update neurons' segments state
	if (P->models[nrn] == INTER) {
		for(int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
			recalc_inter_channels(S, P, N, nrn_seg, S->Vm[nrn_seg]);
		}
	} else if (P->models[nrn] == MOTO) {
		for(int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
			recalc_moto_channels(S, P, N, nrn_seg, S->Vm[nrn_seg]);
		}
	} else if (P->models[nrn] == MUSCLE) {
		for(int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
			recalc_muslce_channels(S, P, N, nrn_seg, S->Vm[nrn_seg]);
		}
	} else {

	}
	//  spike detection for (in synapse kernel)
	nrn_deliver_events(S, P, N, nrn);
}

__device__
void nrn_area_ri(States* S, Parameters* P, Neurons* N) {
	/**
	void nrn_area_ri(Section *sec) [790] treeset.c
	area for right circular cylinders. Ri as right half of parent + left half of this
	*/
	printf("GPU: nrn_area_ri\n");
	double dx, rleft, rright;
	int i1, i3, nrn_seg, segments;
	//
	for (int nrn = 0; nrn < N->size; ++nrn) {
		if (P->models[nrn] == GENERATOR)
			continue;
		i1 = P->nrn_start_seg[nrn];
		i3 = P->nrn_start_seg[nrn + 1];
		segments = (i3 - i1 - 2);
		dx = P->length[nrn] / segments; // divide by the last index of node (or segments count)
		rright = 0;
		// todo sec->pnode needs +1 index
		for (nrn_seg = i1 + 1; nrn_seg < i1 + segments + 1; ++nrn_seg) {
			// area for right circular cylinders. Ri as right half of parent + left half of this
			S->NODE_AREA[nrn_seg] = PI * dx * P->diam[nrn];
			rleft = 1.e-2 * P->Ra[nrn] * (dx / 2.0) / (PI * pow(P->diam[nrn], 2) / 4.0);   // left half segment Megohms
			S->NODE_RINV[nrn_seg] = 1.0 / (rleft + rright); // uS
			rright = rleft;
		}
		//the first and last segments has zero length. Area is 1e2 in dimensionless units
		S->NODE_AREA[i1] = 100.0;
		nrn_seg = i1 + segments + 1; // the last segment
		S->NODE_AREA[nrn_seg] = 100.0;
		S->NODE_RINV[nrn_seg] = 1.0 / rright;
	}
}

__device__
void ext_con_coef(States* S, Parameters* P, Neurons* N) {

}

__device__
void connection_coef(States* S, Parameters* P, Neurons* N) {
	/** void connection_coef(void) treeset.c */
	printf("GPU: connection_coef\n");
	nrn_area_ri(S, P, N);
	// NODE_A is the effect of this node on the parent node's equation
	// NODE_B is the effect of the parent node on this node's equation
	int i1, i3, nrn_seg, segments;
	//
	for (int nrn =0; nrn < N->size; ++nrn) {
		if (P->models[nrn] == GENERATOR)
			continue;
		i1 = P->nrn_start_seg[nrn];
		i3 = P->nrn_start_seg[nrn + 1];
		segments = (i3 - i1 - 2);
		// first the effect of node on parent equation. Note that last nodes have area = 1.e2 in dimensionless
		// units so that last nodes have units of microsiemens
		// todo sec->pnode needs +1 index
		nrn_seg = i1 + 1;
		// sec->prop->dparam[4].val = 1, what is dparam[4].val
		S->NODE_A[nrn_seg] = -1.e2 * 1.0 * S->NODE_RINV[nrn_seg] / S->NODE_AREA[nrn_seg - 1];
		// todo sec->pnode needs +1 index
		for (nrn_seg = i1 + 1 + 1; nrn_seg < i1 + segments + 1 + 1; ++nrn_seg) {
			S->NODE_A[nrn_seg] = -1.e2 * S->NODE_RINV[nrn_seg] / S->NODE_AREA[nrn_seg - 1];
		}
		// now the effect of parent on node equation
		// todo sec->pnode needs +1 index
		for (nrn_seg = i1 + 1; nrn_seg < i1 + segments + 1 + 1; ++nrn_seg) {
			S->NODE_B[nrn_seg] = -1.e2 * S->NODE_RINV[nrn_seg] / S->NODE_AREA[nrn_seg];
		}
	}
	// for extracellular
	ext_con_coef(S, P, N);
	/// note: from LHS, this functions just recalc each time the constant NODED (!)
	/**
	void nrn_lhs(NrnThread *_nt)
	NODE_D[nrn, nd] updating is located at nrn_rhs, because _g is not the global variable
	*/
	// nt->cj = 2/dt if (secondorder) else 1/dt
	// note, the first is CAP
	// function nrn_cap_jacob(_nt, _nt->tml->ml);
	double cj = 1.0 / dt;
	double cfac = 0.001 * cj;
	for (int nrn = 0; nrn < N->size; ++nrn) {
		if (P->models[nrn] == GENERATOR)
			continue;
		i1 = P->nrn_start_seg[nrn];
		i3 = P->nrn_start_seg[nrn + 1];
		segments = (i3 - i1 - 2);
		for (nrn_seg = i1 + 1; nrn_seg < i1 + segments + 1; ++nrn_seg) {  // added + 1 for nodelist
			S->const_NODE_D[nrn_seg] += cfac * P->Cm[nrn];
		}
		// updating NODED
		for (nrn_seg = i1 + 1; nrn_seg < i3; ++nrn_seg) {
			S->const_NODE_D[nrn_seg] -= S->NODE_B[nrn_seg];
			S->const_NODE_D[nrn_seg - 1] -= S->NODE_A[nrn_seg];
		}
	}
	// extra
	// _a_matelm += NODE_A[nrn, nd]
	// _b_matelm += NODE_B[nrn, nd]
}

__global__
void initialization_kernel(States* S, Parameters* P, Neurons* N, double v_init) {
	printf("GPU: initialization_kernel\n");

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int i1, i3;

	if (tid == 0) {
		/** */
		// todo do not invoke for generators
		connection_coef(S, P, N);
		// for different models -- different init function
		for (int nrn = 0; nrn < N->size; ++nrn) {
			// do not init neuron state for generator
			if (P->models[nrn] == GENERATOR)
				continue;
			i1 = P->nrn_start_seg[nrn];
			i3 = P->nrn_start_seg[nrn + 1];
			// for each segment init the neuron model
			for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
				S->Vm[nrn_seg] = v_init;
				if (P->models[nrn] == INTER) {
					nrn_inter_initial(S, P, N, nrn_seg, v_init);
				} else if (P->models[nrn] == MOTO) {
					nrn_moto_initial(S, P, N, nrn_seg, v_init);
				} else if (P->models[nrn] == MUSCLE) {
					nrn_muslce_initial(S, P, N, nrn_seg, v_init);
				} else {

				}
			}
			// init RHS/LHS
			setup_tree_matrix(S, P, N, nrn, i1, i3);
			// init tau synapses
			syn_initial(S, P, N, nrn);
		}
	}
}

__global__
void neuron_kernel(States *S, Parameters *P, Neurons *N, int t) {
	//
	int i1, i3;
	for (int nrn = blockIdx.x * blockDim.x + threadIdx.x; nrn < N->size; nrn += blockDim.x * gridDim.x) {
		//
		N->has_spike[nrn] = false;
		// generator
		if (P->models[nrn] == GENERATOR) {
			// EES
			if (nrn == 0 && (t >= (10 / dt) && t % (int)(1000 / ees_fr / dt) == 0)) {
				N->has_spike[nrn] = true;
			}
		} else {
			i1 = P->nrn_start_seg[nrn];
			i3 = P->nrn_start_seg[nrn + 1];
			setup_tree_matrix(S, P, N, nrn, i1, i3);
			nrn_solve(S, P, N, nrn, i1, i3);
			update(S, P, N, nrn, i1, i3);
			nrn_fixed_step_lastpart(S, P, N, nrn, i1, i3);
		}
	}

}

__global__
void synapse_kernel(Neurons *N, Synapses* synapses) {
	/**
	void deliver_net_events(NrnThread* nt)
	*/
	int pre_nrn, post_id;
	double weight;
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < synapses->size; index += blockDim.x * gridDim.x) {
		pre_nrn = synapses->syn_pre_nrn[index];
		// synapse update
		if (synapses->syn_delay_timer[index] > 0) {
			synapses->syn_delay_timer[index]--;
		// if timer is over -> synapse change the conductance of the post neuron
		} else if (synapses->syn_delay_timer[index] == 0) {
			post_id = synapses->syn_post_nrn[index];
			weight = synapses->syn_weight[index];
			if (weight >= 0) {
				atomicAdd(&N->g_exc[post_id], weight);
			} else {
				atomicAdd(&N->g_inh_A[post_id], -weight * N->factor[post_id]);
				atomicAdd(&N->g_inh_B[post_id], -weight * N->factor[post_id]);
			}
			synapses->syn_delay_timer[index] = -1;
		// if pre nrn has spike and synapse is ready to send siagnal
		} else if (N->has_spike[pre_nrn] && synapses->syn_delay_timer[index] == -1) {
			synapses->syn_delay_timer[index] = synapses->syn_delay[index];
		}
	}
}

void conn_a2a(const Group &pre_neurons, const Group &post_neurons, double delay, double weight) {
	/** */
	for (int pre = pre_neurons.id_start; pre <= pre_neurons.id_end; ++pre) {
		for (int post = post_neurons.id_start; post <= post_neurons.id_end; ++post) {
			// weight = random.gauss(weight, weight / 5)
			// delay = random.gauss(delay, delay / 5)
			vector_syn_pre_nrn.push_back(pre);
			vector_syn_post_nrn.push_back(post);
			vector_syn_weight.push_back(weight);
			vector_syn_delay.push_back((int) (delay / dt));
			vector_syn_delay_timer.push_back(-1);
		}
	}
	printf("Connect all-to-all %s [%d] to %s [%d] (1:%d). Total: %d D=%.1f, W=%.2f\n",
	       pre_neurons.group_name.c_str(), pre_neurons.group_size,
	       post_neurons.group_name.c_str(), post_neurons.group_size,
	       post_neurons.group_size, pre_neurons.group_size * post_neurons.group_size, delay, weight);
}

void connect_fixed_outdegree(const Group &pre_neurons,
                             const Group &post_neurons,
                             double delay,
                             double weight,
                             int indegree = 50) {
	// pre_nrns_ids = pre_group[1]
	// post_nrns_ids = post_group[1]
	uniform_int_distribution<int> nsyn_distr(indegree - 15, indegree);
	uniform_int_distribution<int> pre_nrns_ids(pre_neurons.id_start, pre_neurons.id_end);

	int pre, nsyn = nsyn_distr(generator);
	// nsyn = random.randint(indegree - 15, indegree)
	for (int post = post_neurons.id_start; post <= post_neurons.id_end; ++post) {
		for (int i = 0; i < nsyn; ++i) {
			pre = pre_nrns_ids(generator);
			// weight = random.gauss(weight, weight / 5)
			// delay = random.gauss(delay, delay / 5)
			vector_syn_pre_nrn.push_back(pre);
			vector_syn_post_nrn.push_back(post);
			vector_syn_weight.push_back(weight);
			vector_syn_delay.push_back((int)(delay / dt));
			vector_syn_delay_timer.push_back(-1);
		}
	}
	printf("Connect indegree %s [%d] to %s [%d] (%d:1). Total: %d D=%.1f, W=%.2f\n",
	       pre_neurons.group_name.c_str(), pre_neurons.group_size,
	       post_neurons.group_name.c_str(), post_neurons.group_size,
	       indegree, post_neurons.group_size * indegree, delay, weight);
}


void file_writing(int test_index, GroupMetadata &metadata, const string &folder) {
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
	for (double const &value: metadata.spike_vector) {
		file << value << " ";
	}
	file.close();

	cout << "Saved to: " << folder + file_name << endl;
}

void save(vector<Group> groups) {
	for (Group &group : groups) {
		GroupMetadata new_meta(group, SIM_TIME_IN_STEPS);
		saving_groups.emplace_back(new_meta);
	}
}

void copy_data_to(GroupMetadata& metadata,
                  const double* Vm,
                  const double* g_exc,
                  const double* g_inh_A,
                  const double* g_inh_B,
                  const bool* has_spike,
                  const unsigned int sim_iter) {
	double nrn_mean_volt = 0;
	double nrn_mean_g_exc = 0;
	double nrn_mean_g_inh = 0;

	int center;
	for (unsigned int nrn = metadata.group.id_start; nrn <= metadata.group.id_end; ++nrn) {
		center = vector_nrn_start_seg[nrn] + ((vector_models[nrn] == MUSCLE)? 2 : 1);
		nrn_mean_volt += Vm[center];
		nrn_mean_g_exc += g_exc[nrn];
		nrn_mean_g_inh += (g_inh_B[nrn] - g_inh_A[nrn]);
		if (has_spike[nrn]) {
			metadata.spike_vector.push_back(step_to_ms(sim_iter));
		}
	}
	metadata.voltage_array[sim_iter] = nrn_mean_volt / metadata.group.group_size;
	metadata.g_exc[sim_iter] = nrn_mean_g_exc / metadata.group.group_size;
	metadata.g_inh[sim_iter] = nrn_mean_g_inh / metadata.group.group_size;
}


void save_result(int test_index) {
	string current_path = getcwd(nullptr, 0);

	printf("[Test #%d] Save results to: %s \n", test_index, current_path.c_str());

	for (GroupMetadata &metadata : saving_groups) {
		file_writing(test_index, metadata, current_path);
	}
}

template<typename type>
type* arr_segs() {
	return new type[nrns_and_segs]();
}

void init_network() {
	//
	Group EES = form_group("EES", 1, GENERATOR);
	Group E1 = form_group("E1", 1, GENERATOR);
	Group E2 = form_group("E2", 1, GENERATOR);
	Group E3 = form_group("E3", 1, GENERATOR);
	Group E4 = form_group("E4", 1, GENERATOR);
	Group E5 = form_group("E5", 1, GENERATOR);
	//
	Group CV1 = form_group("CV1", 1, GENERATOR);
	Group CV2 = form_group("CV2", 1, GENERATOR);
	Group CV3 = form_group("CV3", 1, GENERATOR);
	Group CV4 = form_group("CV4", 1, GENERATOR);
	Group CV5 = form_group("CV5", 1, GENERATOR);
	//
	Group C_0 = form_group("C_0");
	Group C_1 = form_group("C_1");
	Group V0v = form_group("V0v");
	//
	Group OM1_0E = form_group("OM1_0E");
	Group OM1_0F = form_group("OM1_0F");
	// OM1
	Group OM1_0 = form_group("OM1_0");
	Group OM1_1 = form_group("OM1_1");
	Group OM1_2_E = form_group("OM1_2_E");
	Group OM1_2_F = form_group("OM1_2_F");
	Group OM1_3 = form_group("OM1_3");
	// OM2
	Group OM2_0 = form_group("OM2_0");
	Group OM2_1 = form_group("OM2_1");
	Group OM2_2_E = form_group("OM2_2_E");
	Group OM2_2_F = form_group("OM2_2_F");
	Group OM2_3 = form_group("OM2_3");
	// OM3
	Group OM3_0 = form_group("OM3_0");
	Group OM3_1 = form_group("OM3_1");
	Group OM3_2_E = form_group("OM3_2_E");
	Group OM3_2_F = form_group("OM3_2_F");
	Group OM3_3 = form_group("OM3_3");
	// OM4
	Group OM4_0 = form_group("OM4_0");
	Group OM4_1 = form_group("OM4_1");
	Group OM4_2_E = form_group("OM4_2_E");
	Group OM4_2_F = form_group("OM4_2_F");
	Group OM4_3 = form_group("OM4_3");
	// OM5
	Group OM5_0 = form_group("OM5_0");
	Group OM5_1 = form_group("OM5_1");
	Group OM5_2_E = form_group("OM5_2_E");
	Group OM5_2_F = form_group("OM5_2_F");
	Group OM5_3 = form_group("OM5_3");
	// afferents
	Group sens_aff = form_group("sens_aff", 120);
	Group Ia_aff_E = form_group("Ia_aff_E", 120);
	Group Ia_aff_F = form_group("Ia_aff_F", 120);
	// motoneurons
	Group MN_E = form_group("MN_E", 210, MOTO);
	Group MN_F = form_group("MN_F", 180, MOTO);
	// muscle fibers
	Group muscle_E = form_group("muscle_E", 20, MUSCLE, 3); // 150 * 210
	Group muscle_F = form_group("muscle_F", 20, MUSCLE, 3); // 100 * 180
	// reflex arc E
	Group Ia_E = form_group("Ia_E", neurons_in_ip);
	Group iIP_E = form_group("iIP_E", neurons_in_ip);
	Group R_E = form_group("R_E");
	// reflex arc F
	Group Ia_F = form_group("Ia_F", neurons_in_ip);
	Group iIP_F = form_group("iIP_F", neurons_in_ip);
	Group R_F = form_group("R_F");
	// extensor IP
	Group eIP_E_1 = form_group("eIP_E_1");
	Group eIP_E_2 = form_group("eIP_E_2");
	Group eIP_E_3 = form_group("eIP_E_3");
	Group eIP_E_4 = form_group("eIP_E_4");
	Group eIP_E_5 = form_group("eIP_E_5");
	// flexor IP
	Group eIP_F_1 = form_group("eIP_F_1");
	Group eIP_F_2 = form_group("eIP_F_2");
	Group eIP_F_3 = form_group("eIP_F_3");
	Group eIP_F_4 = form_group("eIP_F_4");
	Group eIP_F_5 = form_group("eIP_F_5");

	/// generators
	// EES
	// addgener(0, ees_fr, 10000, False)
	// AFF
	// self.Iagener_E = self.addIagener(self.muscle_E, self.muscle_F, 10)
	// self.Iagener_F = self.addIagener(self.muscle_F, self.muscle_F, speed*6)
	// CV
	// self.addgener(25 + speed * layer + i * (speed * (layers + 1) + 175), random.gauss(cfr, cfr/10), (speed / c_int + 1))
	// C_0
	// self.addgener(25 + speed * 6 + i * (speed * 6 + 175), cfr, 175/c_int, False)
	// V0v
	// self.addgener(40 + speed * 6 + i * (speed * 6 + 175), cfr, 175/c_int, False)

	// gen connect
	conn_a2a(EES, Ia_aff_E, 1, 1.5);
	conn_a2a(EES, Ia_aff_E, 1, 1.5);
	conn_a2a(EES, CV1, 2, 1.5);
	// genconnect(self.Iagener_E, self.Ia_aff_E, 0.0001, 1, False, 5)
	// genconnect(self.Iagener_F, self.Ia_aff_F, 0.0001, 1, False, 5)

	connect_fixed_outdegree(Ia_aff_E, MN_E, 1.5, 1.55)
	connect_fixed_outdegree(Ia_aff_F, MN_F, 1.5, 1.5)

	connect_fixed_outdegree(MN_E, muscle_E, 2, 15.5, 45) // connectcells(self.mns_E, self.muscle_E, 15.5, 2, False, 45)
	connect_fixed_outdegree(MN_F, muscle_F, 2, 15.5, 45) // connectcells(self.mns_F, self.muscle_F, 15.5, 2, False, 45)

	/// IP
	// for layer in range(1, 4):
	//      connectcells(self.dict_IP_E[layer-1], self.dict_IP_E[layer+1], 0.45*layer, 2)
	//      connectcells(self.dict_IP_F[layer-1], self.dict_IP_F[layer+1], 0.45*layer, 2)
	// connectinsidenucleus
	connect_fixed_outdegree(EES, OM1_0, 2, 0.00075 * k * skin_time);
	conn_a2a(CV1, OM1_0, 2, 0.00048);
	// connect_fixed_outdegree(CV1, CV2, 1, 15)

	//	def connectinsidenucleus(nucleus):
	//	connectcells(nucleus, nucleus, 0.25, 0.5)

	// Layer 1
	connect_fixed_outdegree(eIP_E_1, eIP_E_1, 0.5, 0.25);
	connect_fixed_outdegree(OM1_2_E, OM1_2_E, 0.5, 0.25);
	connect_fixed_outdegree(OM1_2_F, OM1_2_F, 0.5, 0.25);
	connect_fixed_outdegree(OM1_2_E, eIP_E_1, 3, 2.85);
	connect_fixed_outdegree(eIP_E_1, MN_E, 3, 2.85);
	//
	connect_fixed_outdegree(OM1_0, OM1_1, 3, 2.95);
	connect_fixed_outdegree(OM1_1, OM1_2_E, 3, 2.85);
	connect_fixed_outdegree(OM1_2_E, OM1_1, 3, 1.95);
	connect_fixed_outdegree(OM1_2_E, OM1_3, 3, 0.0007);
	// connect_fixed_outdegree(OM1_2_F, OM2_2_F, 1.5, 2);
	connect_fixed_outdegree(OM1_1, OM1_3, 3, 0.00005);
	connect_fixed_outdegree(OM1_3, OM1_2_E, 3, -4.5);
	connect_fixed_outdegree(OM1_3, OM1_1, 3, -4.5);

	vector<Group> groups = {OM1_0, OM2_0, OM3_0};
	save(groups);
	vector_nrn_start_seg.push_back(nrns_and_segs);
}

void simulate(int test_index) {
	/**
	 *
	 */
	// init structs
	States *S = (States *)malloc(sizeof(States));
	Parameters *P = (Parameters *)malloc(sizeof(Parameters));
	Neurons *N = (Neurons *)malloc(sizeof(Neurons));
	Synapses *synapses = (Synapses *)malloc(sizeof(Synapses));

	// create neurons and their connectomes
	init_network();

	// static parameters
	P->nrn_start_seg = init_gpu_arr(vector_nrn_start_seg);
	P->models = init_gpu_arr(vector_models);
	P->Cm = init_gpu_arr(vector_Cm);
	P->gnabar = init_gpu_arr(vector_gnabar);
	P->gkbar = init_gpu_arr(vector_gkbar);
	P->gl = init_gpu_arr(vector_gl);
	P->Ra = init_gpu_arr(vector_Ra);
	P->diam = init_gpu_arr(vector_diam);
	P->length = init_gpu_arr(vector_length);
	P->ena = init_gpu_arr(vector_ena);
	P->ek = init_gpu_arr(vector_ek);
	P->el = init_gpu_arr(vector_el);
	P->gkrect = init_gpu_arr(vector_gkrect);
	P->gcaN = init_gpu_arr(vector_gcaN);
	P->gcaL = init_gpu_arr(vector_gcaL);
	P->gcak = init_gpu_arr(vector_gcak);
	P->E_ex = init_gpu_arr(vector_E_ex);
	P->E_inh = init_gpu_arr(vector_E_inh);
	P->tau_exc = init_gpu_arr(vector_tau_exc);
	P->tau_inh1 = init_gpu_arr(vector_tau_inh1);
	P->tau_inh2 = init_gpu_arr(vector_tau_inh2);
	P->size = nrns_number;

	// dynamic states of neuron (CPU arrays)
	auto *Vm = arr_segs<double>(); S->Vm = init_gpu_arr(Vm);
	auto *n = arr_segs<double>(); S->n = init_gpu_arr(n);
	auto *m = arr_segs<double>(); S->m = init_gpu_arr(m);
	auto *h = arr_segs<double>(); S->h = init_gpu_arr(h);
	auto *l = arr_segs<double>(); S->l = init_gpu_arr(l);
	auto *s = arr_segs<double>(); S->s = init_gpu_arr(s);
	auto *p = arr_segs<double>(); S->p = init_gpu_arr(p);
	auto *hc = arr_segs<double>(); S->hc = init_gpu_arr(hc);
	auto *mc = arr_segs<double>(); S->mc = init_gpu_arr(mc);
	auto *cai = arr_segs<double>(); S->cai = init_gpu_arr(cai);
	auto *I_Ca = arr_segs<double>(); S->I_Ca = init_gpu_arr(I_Ca);
	auto *NODE_A = arr_segs<double>(); S->NODE_A = init_gpu_arr(NODE_A);
	auto *NODE_B = arr_segs<double>(); S->NODE_B = init_gpu_arr(NODE_B);
	auto *NODE_D = arr_segs<double>(); S->NODE_D = init_gpu_arr(NODE_D);
	auto *const_NODE_D = arr_segs<double>(); S->const_NODE_D = init_gpu_arr(const_NODE_D);
	auto *NODE_RHS = arr_segs<double>(); S->NODE_RHS = init_gpu_arr(NODE_RHS);
	auto *NODE_RINV = arr_segs<double>(); S->NODE_RINV = init_gpu_arr(NODE_RINV);
	auto *NODE_AREA = arr_segs<double>(); S->NODE_AREA = init_gpu_arr(NODE_AREA);
	S->size = nrns_and_segs;

	// special neuron's state
	auto *has_spike = arr_segs<bool>(); N->has_spike = init_gpu_arr(has_spike);
	auto *spike_on = arr_segs<bool>(); N->spike_on = init_gpu_arr(spike_on);
	auto *g_exc = arr_segs<double>(); N->g_exc = init_gpu_arr(g_exc);
	auto *g_inh_A = arr_segs<double>(); N->g_inh_A = init_gpu_arr(g_inh_A);
	auto *g_inh_B = arr_segs<double>(); N->g_inh_B = init_gpu_arr(g_inh_B);
	auto *factor = arr_segs<double>(); N->factor = init_gpu_arr(factor);
	N->size = nrns_number;

	// synaptic parameters
	synapses->syn_pre_nrn = init_gpu_arr(vector_syn_pre_nrn);
	synapses->syn_post_nrn = init_gpu_arr(vector_syn_post_nrn);
	synapses->syn_weight = init_gpu_arr(vector_syn_weight);
	synapses->syn_delay = init_gpu_arr(vector_syn_delay);
	synapses->syn_delay_timer = init_gpu_arr(vector_syn_delay_timer);
	synapses->size = vector_syn_delay.size();

	// allocate structs to the device
	auto *dev_S = init_gpu_arr(S, 1);
	auto *dev_P = init_gpu_arr(P, 1);
	auto *dev_N = init_gpu_arr(N, 1);
	auto *dev_synapses = init_gpu_arr(synapses, 1);

	printf("Network: %d neurons (with segs: %d), %lu synapses\n", nrns_number, nrns_and_segs, vector_syn_delay.size());

	float time;
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// call initialisation kernel
	initialization_kernel<<<1, 1>>>(dev_S, dev_P, dev_N, -70.0);

	// the main simulation loop
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; ++sim_iter) {
		// deliver_net_events, synapse updating and neuron conductance changing
		synapse_kernel<<<5, 256>>>(dev_N, dev_synapses);
		// another neuron's kernel functions
		neuron_kernel<<<10, 32>>>(dev_S, dev_P, dev_N, sim_iter);
		// copy state data
		memcpyDtH(S->Vm, Vm, nrns_and_segs);
		memcpyDtH(N->g_exc, g_exc, nrns_number);
		memcpyDtH(N->g_inh_A, g_inh_A, nrns_number);
		memcpyDtH(N->g_inh_B, g_inh_B, nrns_number);
		memcpyDtH(N->has_spike, has_spike, nrns_number);

		// fill records arrays
		for (GroupMetadata& metadata : saving_groups) {
			copy_data_to(metadata, Vm, g_exc, g_inh_A, g_inh_B, has_spike, sim_iter);
		}
	}
	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));

	printf("Elapsed GPU time: %d ms\n", (int) time);
	// clock rate * 10^6 * (interface width / 8) * 2 / 10^9
	double Tbw = 12000 * pow(10, 6) * (128 / 8) * 2 / pow(10, 9);
	printf("Theoretical Bandwidth GPU (2 Ghz, 128 bit): %.2f GB/s\n", Tbw);
	/* Effective bandwidth = ( ( B r + B w ) ÷ 10 9 ) ÷ time
	Here, the effective bandwidth is in units of GB/s, Br is the number of bytes read per kernel, Bw is the number of
	bytes written per kernel, and time is given in seconds.
	For example, to compute the effective bandwidth of a 2048 x 2048 matrix copy, the following formula could be used:
	Effective bandwidth = ( ( 2048 2 × 4 × 2 ) ÷ 10 9 ) ÷ time */
	save_result(test_index);

	HANDLE_ERROR(cudaFree(S->Vm));
}

int main(int argc, char **argv) {
	int dev = 0;
	cudaDeviceProp deviceProp;
	HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s struct of array at ", argv[0]);
	printf("device %d: %s \n", dev, deviceProp.name);
	//
	HANDLE_ERROR(cudaSetDevice(dev));
	//
	simulate(0);
	// reset device
	HANDLE_ERROR(cudaDeviceReset());
}
