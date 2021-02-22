/**
See the topology https://github.com/research-team/memristive-spinal-cord/blob/master/doc/diagram/cpg_generator_FE_paper.png
Based on the NEURON repository.
*/
#include <random>
#include <vector>
#include <string>
#include "structs.h"
#include <stdexcept>
// for file writing
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <unistd.h>
#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )
#define PI 3.141592654f

using namespace std;

random_device r;
default_random_engine rand_gen(r());

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("!!! %s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

const double dt = 0.025;      // [ms] simulation step
const bool EXTRACELLULAR = false;

// global name of the models
const char GENERATOR = 'g';
const char INTER = 'i';
const char MOTO = 'm';
const char MUSCLE = 'u';
const char AFFERENTS = 'a';

const char layers = 5;      // number of OM layers (5 is default)
const int skin_time = 25;   // duration of layer 25 = 21 cm/s; 50 = 15 cm/s; 125 = 6 cm/s
const int step_number = 1;  // [step] number of full cycle steps
const int cv_fr = 200;      // frequency of CV
const int ees_fr = 40;      // frequency of EES
const int flexor_dur = 125; // flexor duration (125 or 175 ms for 4pedal)

const unsigned int one_step_time = 6 * skin_time + 125;
const unsigned int sim_time = 300; //25 + one_step_time * step_number;
const auto SIM_TIME_IN_STEPS = (unsigned int)(sim_time / dt);  // [steps] converted time into steps

unsigned int nrns_number = 0;     // [id] global neuron id = number of neurons
unsigned int nrns_and_segs = 0;   // [id] global neuron+segs id = number of neurons with segments
const int neurons_in_group = 50;  // number of neurons in a group
const int neurons_in_ip = 196;    // number of neurons in a group

// common neuron constants
const double k = 0.007;           // synaptic coef
const double V_th = 0;            // [mV] voltage threshold
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
// const double g_kno = 0.01;     // [S/cm2] conductance of the todo
// const double g_kir = 0.03;     // [S/cm2] conductance of the Inwardly Rectifying Potassium K+ (Kir) channel
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
vector<double> vector_Cm, vector_gnabar, vector_gkbar, vector_gl, vector_Ra, vector_diam, vector_length, vector_ena,
		vector_ek, vector_el, vector_gkrect, vector_gcaN, vector_gcaL, vector_gcak;
// synaptic parameters
vector<double> vector_E_ex, vector_E_inh, vector_tau_exc, vector_tau_inh1, vector_tau_inh2;
// synapses varaibels
vector<int> vector_syn_pre_nrn, vector_syn_post_nrn, vector_syn_delay, vector_syn_delay_timer;
vector<double> vector_syn_weight;
// results vector
vector <GroupMetadata> saving_groups;
// for debugging
vector <Group> all_groups;
// generators
vector<unsigned int> vec_time_end, vec_nrn_id, vec_freq_in_steps, vec_spike_each_step;

// form structs of neurons global ID and groups name
Group form_group(const string &group_name,
                 int nrns_in_group = neurons_in_group,
                 const char model = INTER,
                 const int segs = 1) {
	/**
	 *
	 */
	Group group = Group();
	group.group_name = group_name;     // name of a neurons group
	group.id_start = nrns_number;      // first ID in the group
	group.id_end = nrns_number + nrns_in_group - 1;  // the latest ID in the group
	group.group_size = nrns_in_group;  // size of the neurons group

	double Cm, gnabar, gkbar, gl, Ra, ena, ek, el, diam, dx, gkrect, gcaN, gcaL, gcak, e_ex, e_inh, tau_exc, tau_inh1, tau_inh2;
	normal_distribution<double> Cm_distr(1, 0.01);
	normal_distribution<double> moto_Cm_distr(2, 0.1);
	uniform_int_distribution<int> moto_diam_distr(45, 55);
	uniform_int_distribution<int> inter_diam_distr(5, 15);
	uniform_real_distribution<double> afferent_diam_distr(15, 35);

	for (int nrn = 0; nrn < nrns_in_group; nrn++) {
		if (model == INTER) {
			Cm = Cm_distr(rand_gen);
			gnabar = 0.1;
			gkbar = 0.08;
			gl = 0.002;
			Ra = 100.0;
			ena = 50.0;
			ek = -77.0; // -90
			el = -70.0;
			diam = inter_diam_distr(rand_gen); // 10
			dx = diam;
			e_ex = 50;
			e_inh = -80;
			tau_exc = 0.35;
			tau_inh1 = 0.5;
			tau_inh2 = 3.5;
		} else if (model == AFFERENTS) {
			Cm = 2;
			gnabar = 0.5;
			gkbar = 0.04;
			gl = 0.002;
			Ra = 200.0;
			ena = 50.0;
			ek = -90.0;
			el = -70.0;
			diam = afferent_diam_distr(rand_gen); // 10
			dx = diam;
			e_ex = 50;
			e_inh = -80;
			tau_exc = 0.35;
			tau_inh1 = 0.5;
			tau_inh2 = 3.5;
		} else if (model == MOTO) {
			Cm = moto_Cm_distr(rand_gen);
			gnabar = 0.05;
			gl = 0.002;
			Ra = 200.0;
			ena = 50.0;
			ek = -80.0;
			el = -70.0;
			diam = moto_diam_distr(rand_gen);
			dx = diam;
			gkrect = 0.3;
			gcaN = 0.05;
			gcaL = 0.0001;
			gcak = 0.3;
			e_ex = 50.0;
			e_inh = -80.0;
			tau_exc = 0.3;
			tau_inh1 = 1.0;
			tau_inh2 = 1.5;
			if (diam > 50) {
				gnabar = 0.1;
				gcaL = 0.001;
				gl = 0.003;
				gkrect = 0.2;
				gcak = 0.2;
			}
		} else if (model == MUSCLE) {
			Cm = 3.6;
			gnabar = 0.15;
			gkbar = 0.03;
			gl = 0.0002;
			Ra = 1.1;
			ena = 55.0;
			ek = -80.0;
			el = -72.0;
			diam = 40.0;
			dx = 3000.0;
			e_ex = 0.0;
			e_inh = -80.0;
			tau_exc = 0.3;
			tau_inh1 = 1.0;
			tau_inh2 = 1.0;
		} else if (model == GENERATOR) {

		} else {
			throw logic_error("Choose the model");
		}
		// common properties
		vector_Cm.push_back(Cm);
		vector_gnabar.push_back(gnabar);
		vector_gkbar.push_back(gkbar);
		vector_gl.push_back(gl);
		vector_el.push_back(el);
		vector_ena.push_back(ena);
		vector_ek.push_back(ek);
		vector_Ra.push_back(Ra);
		vector_diam.push_back(diam);
		vector_length.push_back(dx);
		vector_gkrect.push_back(gkrect);
		vector_gcaN.push_back(gcaN);
		vector_gcaL.push_back(gcaL);
		vector_gcak.push_back(gcak);
		vector_E_ex.push_back(e_ex);
		vector_E_inh.push_back(e_inh);
		vector_tau_exc.push_back(tau_exc);
		vector_tau_inh1.push_back(tau_inh1);
		vector_tau_inh2.push_back(tau_inh2);
		//
		vector_nrn_start_seg.push_back(nrns_and_segs);
		nrns_and_segs += (segs + 2);
		vector_models.push_back(model);
		// void (*foo)(int);
		// foo = &my_int_func;
	}

	nrns_number += nrns_in_group;
	printf("Formed %s IDs [%d ... %d] = %d\n",
	       group_name.c_str(), nrns_number - nrns_in_group, nrns_number - 1, nrns_in_group);

	// for debugging
	all_groups.push_back(group);

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

// init GPU array and copy data from the CPU array
template<typename type>
type* init_gpu_arr(type *cpu_var, unsigned int size = nrns_and_segs) {
	type *gpu_var;
	HANDLE_ERROR(cudaMalloc(&gpu_var, size * sizeof(type)));
	memcpyHtD<type>(cpu_var, gpu_var, size);
	return gpu_var;
}

// init GPU array and copy data from the CPU vector
template<typename type>
type *init_gpu_arr(vector<type> &vec) {
	type *gpu_var;
	HANDLE_ERROR(cudaMalloc(&gpu_var, sizeof(type) * vec.size()));
	memcpyHtD<type>(vec.data(), gpu_var, vec.size());
	return gpu_var;
}

void add_generator(Group &group, double start, double end, double freq) {
	vec_nrn_id.push_back(group.id_start);
	vec_time_end.push_back(ms_to_step(end));
	vec_freq_in_steps.push_back(ms_to_step(1000 / freq));
	vec_spike_each_step.push_back(ms_to_step(start));
	printf("start %d end %d freq %d\n", ms_to_step(start), ms_to_step(end), ms_to_step(1000 / freq));
}

// convert vector to the array
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
	 * calculate synaptic current
	 */
	return N->g_exc[nrn] * (voltage - P->E_ex[nrn]) + (N->g_inh_B[nrn] - N->g_inh_A[nrn]) * (voltage - P->E_inh[nrn]);
}

__device__
double nrn_moto_current(States* S, Parameters* P, Neurons* N, int nrn, int nrn_seg_index, double voltage) {
	/**
	 * calculate channels current
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
	 * calculate channels current
	 */
	double iNa = P->gnabar[nrn] * pow(S->m[nrn_seg_index], 3) * S->h[nrn_seg_index] * (voltage - P->ena[nrn]);
	double iK = P->gkbar[nrn] * pow(S->n[nrn_seg_index], 4) * (voltage - P->ek[nrn]);
	double iL = P->gl[nrn] * (voltage - P->el[nrn]);
	return iNa + iK + iL;
}

__device__
void recalc_synaptic(States* S, Parameters* P, Neurons* N, int nrn) {
	/**
	 * updating conductance(summed) of neurons' post-synaptic conenctions
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
	 * initialize tau(rise / decay time, ms) and factor(const) variables
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
	 * initialize channels, based on cropped evaluate_fct function
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
	/**
	 * initialize channels, based on cropped evaluate_fct function
	 */
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
	 * initialize channels, based on cropped evaluate_fct function
	 */
	double V_mem = V - V_adj;
	// m_inf
	double a = 0.32 * (13.0 - V_mem) / (exp((13.0 - V_mem) / 4.0) - 1.0);
	double b = 0.28 * (V_mem - 40.0) / (exp((V_mem - 40.0) / 5.0) - 1.0);
	S->m[nrn_seg_index] = a / (a + b);
	// h_inf
	a = 0.128 * exp((17.0 - V_mem) / 18.0);
	b = 4.0 / (1.0 + exp((40.0 - V_mem) / 5.0));
	S->h[nrn_seg_index] = a / (a + b);
	// n_inf
	a = 0.032 * (15.0 - V_mem) / (exp((15.0 - V_mem) / 5.0) - 1.0);
	b = 0.5 * exp((10.0 - V_mem) / 40.0);
	S->n[nrn_seg_index] = a / (a + b);
}

__device__
void recalc_inter_channels(States* S, Parameters* P, Neurons* N, int nrn_seg_index, double V) {
	/**
	 * calculate new states of channels (evaluate_fct)
	 */
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
	/**
	 * calculate new states of channels (evaluate_fct)
	 */
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
	/**
	 * calculate new states of channels (evaluate_fct)
	 */
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
void nrn_rhs_ext(States* S, Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	 * void nrn_rhs_ext(NrnThread* _nt)
	 */
	// init _rhs and _lhs (NODE_D) as zero
	for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
		S->EXT_RHS[nrn_seg][0] -= S->NODE_RHS[nrn_seg];
	}
	int nlayer = 2;
	int j;
	float e_extracellular;
	for (int nrn_seg = i1 + 1; nrn_seg < i3; ++nrn_seg) {
		j = nlayer - 1;
		S->EXT_RHS[nrn_seg][j] -= xg[nrn_seg - i1] * (S->EXT_RHS[nrn_seg][j] - e_extracellular);
		j = 0;
		x = xg[nrn_seg - i1] * (ext_v[nrn_seg][j] - ext_v[nrn_seg][j + 1]);
		S->EXT_RHS[nrn_seg][j] -= x;
		S->EXT_RHS[nrn_seg][j + 1] += x;
	}
}

__device__
void nrn_setup_ext(States* S, Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	double cj = 1 / dt;
	double cfac = 0.001 * cj;
	int nlayer = 2;
	// todo find the place where it is zeroed
	for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg)
		for (int j = 0; j < nlayer; ++j)
			S->EXT_D[nrn_seg][j] = 0;
	// d contains all the membrane conductances (and capacitance)
	// i.e. (cm/dt + di/dvm - dis/dvi)*[dvi] and (dis/dvi)*[dvx]
	for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
		// nde->_d only has -ELECTRODE_CURRENT contribution
		S->EXT_D[nrn_seg][0] += S->NODE_D[nrn_seg];
		// D[0] = [0 0.1442 0.1442 0.1442 0 ]
	}
	// series resistance, capacitance, and axial terms
	for (int nrn_seg = i1 + 1; nrn_seg < i3; ++nrn_seg) {
		// series resistance and capacitance to ground
		int j = 0;
		int nd = nrn_seg - i1;   // start indexing from 0
		while (true) {
			double mfac = xg[nd] + xc[nd] * cfac;
			S->EXT_D[nrn_seg][j] += mfac;
			j += 1;
			if (j == nlayer)
				break;
			S->EXT_D[nrn_seg][j] += mfac;
		}
		// axial connections
		for (j = 0; j < nlayer; ++j) {
			S->EXT_D[nrn_seg][j] -= S->EXT_B[nrn_seg][j];
			S->EXT_D[nrn_seg - 1][j] -= S->EXT_A[nrn_seg][j];
		}
	}
	// D[0] = [2e-08 1e+09 1e+09 1e+09 2e-08 ]
	// D[1] = [2e-08 2e+09 2e+09 2e+09 2e-08 ]
}

__device__
void nrn_update_2d(int nrn) {

}

__device__
void nrn_rhs(States* S, Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	 * void nrn_rhs(NrnThread *_nt) combined with the first part of nrn_lhs
	 * calculate right hand side of
	 * cm*dvm/dt = -i(vm) + is(vi) + ai_j*(vi_j - vi)
	 * cx*dvx/dt - cm*dvm/dt = -gx*(vx - ex) + i(vm) + ax_j*(vx_j - vx)
	 * This is a common operation for fixed step, cvode, and daspk methods
	 */
	// init _rhs and _lhs (NODE_D) as zero
	for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
		S->NODE_RHS[nrn_seg] = 0.0;
		// replace the process: init by 0, add Cm*frac, add A and B
		S->NODE_D[nrn_seg] = 0; // S->const_NODE_D[nrn_seg];
//		ext_rhs[i1:i3, :] = 0
	}

	// update MOD rhs, CAPS has no current [CAP MOD CAP]!
	int center_segment = i1 + ((P->models[nrn] == MUSCLE)? 2 : 1);
	// update segments except CAPs
	double V, _g, _rhs;
	for (int nrn_seg = i1 + 1; nrn_seg < i3 - 1; ++nrn_seg) {
		V = S->Vm[nrn_seg];
		// SYNAPTIC update
		// static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type)
		_g = syn_current(N, P, nrn, V + 0.001);
		_rhs = syn_current(N, P, nrn, V);
		_g = (_g - _rhs) / 0.001;
		_g *= 1.e2 / S->NODE_AREA[nrn_seg];
		_rhs *= 1.e2 / S->NODE_AREA[nrn_seg];
		S->NODE_RHS[nrn_seg] -= _rhs;
		// static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type)
		S->NODE_D[nrn_seg] += _g;

		// void nrn_cap_jacob(NrnThread* _nt, Memb_list* ml) {
		S->NODE_D[nrn_seg] += S->const_NODE_D[nrn_seg];
		// activsynapse_lhs()
		// activclamp_lhs()

		// NEURON update
		// static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type)
		if (P->models[nrn] == INTER || P->models[nrn] == AFFERENTS) {
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
			printf("\nERROR\n");
		}
		// save data like in NEURON (after .mod nrn_cur)
		_g = (_g - _rhs) / 0.001;
		S->NODE_RHS[nrn_seg] -= _rhs;
		// note that CAP has no jacob
		S->NODE_D[nrn_seg] += _g;
	} // end FOR segments

	if (EXTRACELLULAR) {
		// Cannot have any axial terms yet so that i(vm) can be calculated from
		// i(vm)+is(vi) and is(vi) which are stored in rhs vector.
		nrn_rhs_ext(nrn);
		// nrn_rhs_ext has also computed the the internal axial current for those
		// nodes containing the extracellular mechanism
	}
	// activstim_rhs()
	// activclamp_rhs()

	double dv;
	for (int nrn_seg = i1 + 1; nrn_seg < i3; ++nrn_seg) {
		dv = S->Vm[nrn_seg - 1] - S->Vm[nrn_seg];
		// our connection coefficients are negative so
		S->NODE_RHS[nrn_seg] -= S->NODE_B[nrn_seg] * dv;
		S->NODE_RHS[nrn_seg - 1] += S->NODE_A[nrn_seg] * dv;
	}
	// at this point d contains all the membrane conductances
	// now add the axial currents
	for (int nrn_seg = i1 + 1; nrn_seg < i3; ++nrn_seg) {
		S->NODE_D[nrn_seg] -= S->NODE_B[nrn_seg];
		S->NODE_D[nrn_seg - 1] -= S->NODE_A[nrn_seg];
	}
}

__device__
void bksub(States* S, Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	 * void bksub(NrnThread* _nt)
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
	 * void triang(NrnThread* _nt)
	 */
	// intracellular
	double ppp;
	for (int nrn_seg = i3 - 1; nrn_seg >= i1 + 1; --nrn_seg) {
		ppp = S->NODE_A[nrn_seg] / S->NODE_D[nrn_seg];
		S->NODE_D[nrn_seg - 1] -= ppp * S->NODE_B[nrn_seg];
		S->NODE_RHS[nrn_seg - 1] -= ppp * S->NODE_RHS[nrn_seg];
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
	 * void nrn_solve(NrnThread* _nt)
	 */
	triang(S, P, N, nrn, i1, i3);
	bksub(S, P, N, nrn, i1, i3);
}

__device__
void setup_tree_matrix(States* S, Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	 * void setup_tree_matrix(NrnThread* _nt)
	 */
	nrn_rhs(S, P, N, nrn, i1, i3);
//	if (nrn == 2) for (int ii = i1; ii < i3; ++ii) printf("ii%d\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n", ii, S->Vm[ii], S->NODE_RHS[ii], S->NODE_RINV[ii], S->NODE_A[ii], S->NODE_B[ii], S->NODE_D[ii], S->NODE_AREA[ii]);
}

__device__
void update(States* S, Parameters* P, Neurons* N, int nrn, int i1, int i3) {
	/**
	 * void update(NrnThread* _nt)
	 */
	// final voltage updating
	for (int nrn_seg = i1; nrn_seg < i3; ++nrn_seg) {
		S->Vm[nrn_seg] += S->NODE_RHS[nrn_seg];
	}
	// extracellular
	nrn_update_2d(nrn);
}

__device__
void nrn_deliver_events(States* S, Parameters* P, Neurons* N, int nrn) {
	/**
	 * void nrn_deliver_events(NrnThread* nt)
	 */
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
	 * void *nrn_fixed_step_lastpart(NrnThread *nth)
	 */
	// update neurons' synapses state
	recalc_synaptic(S, P, N, nrn);
	//  update neurons' segments state
	if (P->models[nrn] == INTER || P->models[nrn] == AFFERENTS) {
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
	 * void nrn_area_ri(Section *sec) [790] treeset.c
	 * area for right circular cylinders. Ri as right half of parent + left half of this
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
	/**
	 * void connection_coef(void) treeset.c
	 */
	printf("GPU: connection_coef\n");
	nrn_area_ri(S, P, N);
	int i1, i3, nrn_seg, segments;
	//
	for (int nrn = 0; nrn < N->size; ++nrn) {
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

	/**
	 * note: from LHS, this functions just recalc each time the constant NODED (!)
	 * void nrn_lhs(NrnThread *_nt)
	 * NODE_D[nrn, nd] updating is located at nrn_rhs, because _g is not the global variable
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
		// nrn_cap_jacob
		for (nrn_seg = i1 + 1; nrn_seg < i3 - 1; ++nrn_seg) {
			S->const_NODE_D[nrn_seg] += cfac * P->Cm[nrn];
		}
	}
	// extra
	// _a_matelm += NODE_A[nrn, nd]
	// _b_matelm += NODE_B[nrn, nd]
}

__global__
void initialization_kernel(States* S, Parameters* P, Neurons* N, double v_init) {
	/**
	 *
	 */
	if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
		int i1, i3;
		printf("GPU: initialization_kernel\n");
		//
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
				if (P->models[nrn] == INTER || P->models[nrn] == AFFERENTS) {
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
void neuron_kernel(States *S, Parameters *P, Neurons *N, Generators *G, int t) {
	/**
	 *
	 */
	int i1, i3;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//
	for (int nrn = tid; nrn < N->size; nrn += blockDim.x * gridDim.x) {
		// reset the spike state
		N->has_spike[nrn] = false;
		//
		if (P->models[nrn] != GENERATOR) {
			// calc the borders of the neuron by theirs segments
			i1 = P->nrn_start_seg[nrn];
			i3 = P->nrn_start_seg[nrn + 1];
			// re-calc currents and states based on synaptic activity
			setup_tree_matrix(S, P, N, nrn, i1, i3);
			// solve equations
			nrn_solve(S, P, N, nrn, i1, i3);
			// change voltage of the neurons based on solved equations
			update(S, P, N, nrn, i1, i3);
			// recalc conductance, update channels and deliver network events
			nrn_fixed_step_lastpart(S, P, N, nrn, i1, i3);
		}
	}
	// update generators
	if (tid == 0) {
		for (int generator = 0; generator < G->size; ++generator) {
			if (t == G->spike_each_step[generator] && t < G->time_end[generator]) {
				G->spike_each_step[generator] += G->freq_in_steps[generator];
				N->has_spike[G->nrn_id[generator]] = true;
			}
		}
	}
}

__global__
void synapse_kernel(Neurons *N, Synapses* synapses) {
	/**
	 * void deliver_net_events(NrnThread* nt)
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

void conn_generator(Group &generator, Group &post_neurons, double delay, double weight, int indegree=50) {
	/**
	 *
	 */
	if (weight > 0.01)
		weight = 0.01;
	if (weight < -0.1)
		weight = -0.1;

	uniform_int_distribution<int> nsyn_distr(indegree, indegree + 5);
	normal_distribution<double> delay_distr(delay, delay / 5);
	normal_distribution<double> weight_distr(weight, weight / 6);

//	int nsyn = nsyn_distr(rand_gen);
	printf("Connect generator %s [%d..%d] to %s [%d..%d] (1:%d). Synapses %d, D=%.1f, W=%.2f\n",
	       generator.group_name.c_str(), generator.id_start, generator.id_end,
	       post_neurons.group_name.c_str(), post_neurons.id_start, post_neurons.id_end,
	       post_neurons.group_size, generator.group_size * post_neurons.group_size, delay, weight);
	//
	int synapse_index = 0;
	int nsyn = nsyn_distr(rand_gen);
	int gen_id = generator.id_start;
	if (generator.group_size > 1) {
		printf("Generator cannot include more than 1 neuron!\n");
		exit(0);
	}
	for (int post = post_neurons.id_start; post <= post_neurons.id_end; ++post) {
		for (int i = 0; i < nsyn; ++i) {
			vector_syn_pre_nrn.push_back(gen_id);
			vector_syn_post_nrn.push_back(post);
			vector_syn_weight.push_back(weight_distr(rand_gen));
			vector_syn_delay.push_back(ms_to_step(delay_distr(rand_gen)));
			vector_syn_delay_timer.push_back(-1);
		}
	}
}

void connect_fixed_indegree(Group &pre_neurons, Group &post_neurons, double delay, double weight, int indegree=50) {
	/**
	 *
	 */
	if (vector_models[pre_neurons.id_start] == AFFERENTS && vector_models[post_neurons.id_start] == MOTO) {
		if (weight > 0.1)
			weight = 0.1;
		if (weight < -1)
			weight = -1;
	} else {
		if (vector_models[post_neurons.id_start] == MOTO) {
			if (weight > 0.05)
				weight = 0.05;
			if (weight < -0.5)
				weight = -0.5;
		} else {
			if (weight > 0.01)
				weight = 0.01;
			if (weight < -0.1)
				weight = -0.1;
		}
	}

	if (vector_models[post_neurons.id_start] == INTER) {
		printf("POST INTER ");
		weight /= 11.0;
	}

	uniform_int_distribution<int> nsyn_distr(indegree - 15, indegree);
	uniform_int_distribution<int> pre_nrns_ids(pre_neurons.id_start, pre_neurons.id_end);
	normal_distribution<double> delay_distr(delay, delay / 6);
	normal_distribution<double> weight_distr(weight, weight / 6);
	auto nsyn = nsyn_distr(rand_gen);

	printf("Connect indegree %s [%d..%d] to %s [%d..%d] (1:%d). Synapses %d, D=%.1f, W=%.2f\n",
	       pre_neurons.group_name.c_str(), pre_neurons.id_start, pre_neurons.id_end,
	       post_neurons.group_name.c_str(), post_neurons.id_start, post_neurons.id_end,
	       indegree, post_neurons.group_size * indegree, delay, weight);
	//
	int prerand = 0;
	int synapse_index = 0;
	for (int post = post_neurons.id_start; post <= post_neurons.id_end; ++post) {
		for (int i = 0; i < nsyn; ++i) {
			prerand = pre_nrns_ids(rand_gen);
			vector_syn_pre_nrn.push_back(prerand);
			vector_syn_post_nrn.push_back(post);
			if (vector_models[post] == AFFERENTS) {
				vector_syn_weight.push_back(weight);
				vector_syn_delay.push_back(ms_to_step(delay));
			} else {
				vector_syn_weight.push_back(weight_distr(rand_gen));
				vector_syn_delay.push_back(ms_to_step(delay_distr(rand_gen)));
			}
			vector_syn_delay_timer.push_back(-1);
		}
	}
}

void connectinsidenucleus(Group &nucleus) {
	connect_fixed_indegree(nucleus, nucleus, 0.5, 0.25);
}

void file_writing(int test_index, GroupMetadata &metadata, const string &folder) {
	/**
	 *
	 */
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
	for (unsigned int nrn = metadata.group.id_start; nrn <= metadata.group.id_end; ++nrn) {// nrn <= metadata.group.id_end; ++nrn) {
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
	// important: nrns_and_segs initialized at network building
	return new type[nrns_and_segs]();
}

void createmotif(Group OM0, Group OM1, Group OM2, Group OM3) {
	/**
	 * Connects motif module
	 * see https://github.com/research-team/memristive-spinal-cord/blob/master/doc/diagram/cpg_generator_FE_paper.png
	 */
	connect_fixed_indegree(OM0, OM1, 1, 2.85);
	connect_fixed_indegree(OM1, OM2, 2, 2.85);
	connect_fixed_indegree(OM2, OM1, 2, 1.95);
	connect_fixed_indegree(OM2, OM3, 2, 0.0005);
	connect_fixed_indegree(OM1, OM3, 2, 0.00005);
	connect_fixed_indegree(OM3, OM2, 2, -4.5);
	connect_fixed_indegree(OM3, OM1, 2, -4.5);
}

void createmotif_flex(Group OM0, Group OM1, Group OM2, Group OM3) {
	/**
	 * Connects motif module
	 * see https://github.com/research-team/memristive-spinal-cord/blob/master/doc/diagram/cpg_generator_FE_paper.png
	 */
	connect_fixed_indegree(OM0, OM1, 3, 2.85);
	connect_fixed_indegree(OM1, OM2, 3, 0);
	connect_fixed_indegree(OM2, OM1, 3, 0);
	connect_fixed_indegree(OM2, OM3, 3, 0.0005);
	connect_fixed_indegree(OM1, OM3, 3, 0.00005);
	connect_fixed_indegree(OM3, OM2, 2, -4.5);
	connect_fixed_indegree(OM3, OM1, 3, -4.5);
}