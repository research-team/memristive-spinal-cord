#include <map>
#include <string>
#include <vector>
#include <iostream>
#include "test.h"
#define CHECK( err ) ( HandleError( err, __FILE__, __LINE__ ) )

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

using namespace std;

// common neuron constants
float k = 0.017;           // synaptic coef
float V_th = -40;          // [mV] voltage threshold
float V_adj = -63;         // [mV] adjust voltage for -55 threshold
// moto neuron constants
float ca0 = 2;             // initial calcium concentration
float amA = 0.4;           // const ??? todo
float amB = 66;            // const ??? todo
float amC = 5;             // const ??? todo
float bmA = 0.4;           // const ??? todo
float bmB = 32;            // const ??? todo
float bmC = 5;             // const ??? todo
float R_const = 8.314472;  // [k-mole] or [joule/degC] const
float F_const = 96485.34;  // [faraday] or [kilocoulombs] const
// muscle fiber constants
float g_kno = 0.01;        // [S/cm2] conductance of the todo
float g_kir = 0.03;        // [S/cm2] conductance of the Inwardly Rectifying Potassium K+ (Kir) channel
// Boltzman steady state curve
float vhalfl = -98.92;     // [mV] inactivation half-potential
float kl = 10.89;          // [mV] Stegen et al. 2012
// tau_infty
float vhalft = 67.0828;    // [mV] fitted //100 uM sens curr 350a, Stegen et al. 2012
float at = 0.00610779;     // [/ ms] Stegen et al. 2012
float bt = 0.0817741;      // [/ ms] Note: typo in Stegen et al. 2012
// temperature dependence
float q10 = 1;             // temperature scaling (sensitivity)
float celsius = 36;        // [degC] temperature of the cell
// i_membrane [mA/cm2]
float e_extracellular = 0; // [mV]
float xraxial = 1e9;       // [MOhm/cm]

unsigned int nrns_number = 0;
unsigned int nrns_and_segs = 0;
unsigned int generators_id_end = 0;

unsigned int global_id = 0;
unsigned int SIM_TIME_IN_STEPS;
const int LEG_STEPS = 1;             // [step] number of full cycle steps
const double SIM_STEP = 0.025;        // [ms] simulation step
const int neurons_in_group = 50;     // number of neurons in a group
const int neurons_in_ip = 196;       // number of neurons in a group

vector <GroupMetadata> all_groups;

// form structs of neurons global ID and groups name
Group form_group(const string &group_name, int nrns_in_group = neurons_in_group) {
	Group group = Group();
	group.group_name = group_name;     // name of a neurons group
	group.id_start = global_id;        // first ID in the group
	group.id_end = global_id + nrns_in_group - 1;  // the latest ID in the group
	group.group_size = nrns_in_group;  // size of the neurons group
	group.time = 100000;
	all_groups.emplace_back(group);

	global_id += nrns_in_group;
	printf("Formed %s IDs [%d ... %d] = %d\n", group_name.c_str(), global_id - nrns_in_group, global_id - 1, nrns_in_group);

	return group;
}

// copy data from host to device
template<typename type>
void memcpyHtD(type *gpu, type *host, unsigned int size) {
	cudaMemcpy(gpu, host, sizeof(type) * size, cudaMemcpyHostToDevice);
}

// copy data from device to host
template<typename type>
void memcpyDtH(type *host, type *gpu, unsigned int size) {
	cudaMemcpy(host, gpu, size * sizeof(type), cudaMemcpyDeviceToHost);
}

template<typename type>
type* init_gpu_arr(type *cpu_var, int size) {
	type *gpu_var;
	cudaMalloc(&gpu_var, size * sizeof(type));
	memcpyHtD<type>(gpu_var, cpu_var, size);
	return gpu_var;
}

template<typename type>
type *init_gpu_arr(vector<type> &vec) {
	type *gpu_var;
	cudaMalloc(&gpu_var, sizeof(type) * vec.size());
	memcpyHtD<type>(gpu_var, vec.data(), vec.size());
	return gpu_var;
}

template<typename type>
type* vec2arr(vector<type> &vec) {
	return vec.cpu_vector.data();
}

__device__
void recalc_synaptic(int tid) {
	/**
	updating conductance(summed) of neurons' post-synaptic conenctions
	*/
}

__device__
void nrn_rhs(int tid) {
	/**
	void nrn_rhs(NrnThread *_nt) combined with the first part of nrn_lhs
	calculate right hand side of
	cm*dvm/dt = -i(vm) + is(vi) + ai_j*(vi_j - vi)
	cx*dvx/dt - cm*dvm/dt = -gx*(vx - ex) + i(vm) + ax_j*(vx_j - vx)
	This is a common operation for fixed step, cvode, and daspk methods
	*/

}

__device__
void bksub(int tid) {
	/**
	void bksub(NrnThread* _nt)
	*/
}

__device__
void triang(int tid) {
	/**
	void triang(NrnThread* _nt)
	*/
}

__device__
void nrn_solve(int tid) {
	/**
	void nrn_solve(NrnThread* _nt)
	*/
	triang(tid);
	bksub(tid);
}

__device__
void update(int tid) {
	/**
	void update(NrnThread* _nt)
	*/
}

__device__
void setup_tree_matrix(int tid) {
	nrn_rhs(tid);
	// simplified nrn_lhs(nrn)
	//	i1 = P.nrn_start_seg[nrn]
	//	i3 = P.nrn_start_seg[nrn + 1]
	//	S.NODE_D[i1:i3] += S.const_NODE_D[i1:i3]
}

__device__
void nrn_deliver_events(int tid) {
	/**
	void nrn_deliver_events(NrnThread* nt)
	*/
}

__device__
void nrn_fixed_step_lastpart(int tid) {
	/**
	void *nrn_fixed_step_lastpart(NrnThread *nth)
	*/
//	i1 = P.nrn_start_seg[nrn]
//	i3 = P.nrn_start_seg[nrn + 1]
//  update synapses' state
	recalc_synaptic(tid);
//  update neurons' segment state
//	if P.models[nrn] == INTER:
//	for nrn_seg in range(i1, i3):
//	recalc_inter_channels(nrn_seg, S.Vm[nrn_seg])
//	elif P.models[nrn] == MOTO:
//	for nrn_seg in range(i1, i3):
//	recalc_moto_channels(nrn_seg, S.Vm[nrn_seg])
//	elif P.models[nrn] == MUSCLE:
//	for nrn_seg in range(i1, i3):
//	recalc_muslce_channels(nrn_seg, S.Vm[nrn_seg])
//	else:
//	raise Exception("No model")
//  spike detection for
	nrn_deliver_events(tid);
}

__global__
void some_kernel(States *S, Parameters *P, int neurons_number) {
	/// STRIDE neuron update
	for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < neurons_number; tid += blockDim.x * gridDim.x) {
		S->Vm[tid] += P->Cm[tid] * P->ena[tid] + P->ek[tid];

		setup_tree_matrix(tid);
		nrn_solve(tid);
		update(tid);
		nrn_fixed_step_lastpart(tid);
	}
}

void simulate() {
	/**
	 *
	 */
	const int size = 10000;
	SIM_TIME_IN_STEPS = 100;
	// init structs
	States *dev_S, *S = (States *)malloc(sizeof(States));
	Parameters *dev_P, *P = (Parameters *)malloc(sizeof(Parameters));

	/// finitialize()
	// Allocate and fill host data
	vector<short> vector_nrn_start_seg;
	vector<char> vector_models;
	vector<float> vector_Cm;
	vector<float> vector_gnabar;
	vector<float> vector_gkbar;
	vector<float> vector_gl;
	vector<float> vector_Ra;
	vector<float> vector_diam;
	vector<float> vector_length;
	vector<float> vector_ena;
	vector<float> vector_ek;
	vector<float> vector_el;
	vector<float> vector_gkrect;
	vector<float> vector_gcaN;
	vector<float> vector_gcaL;
	vector<float> vector_gcak;
	vector<float> vector_E_ex;
	vector<float> vector_E_inh;
	vector<float> vector_tau_exc;
	vector<float> vector_tau_inh1;
	vector<float> vector_tau_inh2;

	// init neurons
	for (int i = 0; i < size; i++) {
		vector_Cm.push_back(10);
		vector_ena.push_back(5);
		vector_ek.push_back(199);
	}
	/// GPU
	// init States CPU arrays
	auto *Vm = new float[size]();
	auto *n = new float[size]();
	auto *m = new float[size]();
	auto *h = new float[size]();
	auto *l = new float[size]();
	auto *s = new float[size]();
	auto *p = new float[size]();
	auto *hc = new float[size]();
	auto *mc = new float[size]();
	auto *cai = new float[size]();
	auto *I_Ca = new float[size]();
	auto *NODE_A = new float[size]();
	auto *NODE_B = new float[size]();
	auto *NODE_D = new float[size]();
	auto *const_NODE_D = new float[size]();
	auto *NODE_RHS = new float[size]();
	auto *NODE_RINV = new float[size]();
	auto *NODE_AREA = new float[size]();
	auto *has_spike = new bool[size]();
	auto *spike_on = new bool[size]();
	auto *g_exc = new float[size]();
	auto *g_inh_A = new float[size]();
	auto *g_inh_B = new float[size]();
	auto *factor = new float[size]();

	/// GPU
	// init Parameters (malloc + memcpy) GPU arrays based on CPU vectors
	short *gpu_nrn_start_seg = init_gpu_arr(vector_nrn_start_seg);
	char *gpu_models = init_gpu_arr(vector_models);
	auto *gpu_Cm = init_gpu_arr(vector_Cm);
	float *gpu_gnabar = init_gpu_arr(vector_gnabar);
	float *gpu_gkbar = init_gpu_arr(vector_gkbar);
	float *gpu_gl = init_gpu_arr(vector_gl);
	float *gpu_Ra = init_gpu_arr(vector_Ra);
	float *gpu_diam = init_gpu_arr(vector_diam);
	float *gpu_length = init_gpu_arr(vector_length);
	float *gpu_ena = init_gpu_arr(vector_ena);
	float *gpu_ek = init_gpu_arr(vector_ek);
	float *gpu_el = init_gpu_arr(vector_el);
	float *gpu_gkrect = init_gpu_arr(vector_gkrect);
	float *gpu_gcaN = init_gpu_arr(vector_gcaN);
	float *gpu_gcaL = init_gpu_arr(vector_gcaL);
	float *gpu_gcak = init_gpu_arr(vector_gcak);
	float *gpu_E_ex = init_gpu_arr(vector_E_ex);
	float *gpu_E_inh = init_gpu_arr(vector_E_inh);
	float *gpu_tau_exc = init_gpu_arr(vector_tau_exc);
	float *gpu_tau_inh1 = init_gpu_arr(vector_tau_inh1);
	float *gpu_tau_inh2 = init_gpu_arr(vector_tau_inh2);

	// init States GPU arrays based on CPU arrays
	auto *gpu_Vm = init_gpu_arr(Vm, size);
	auto *gpu_n = init_gpu_arr(n, size);
	auto *gpu_m = init_gpu_arr(m, size);
	auto *gpu_h = init_gpu_arr(h, size);
	auto *gpu_l = init_gpu_arr(l, size);
	auto *gpu_s = init_gpu_arr(s, size);
	auto *gpu_p = init_gpu_arr(p, size);
	auto *gpu_hc = init_gpu_arr(hc, size);
	auto *gpu_mc = init_gpu_arr(mc, size);
	auto *gpu_cai = init_gpu_arr(cai, size);
	auto *gpu_I_Ca = init_gpu_arr(I_Ca, size);
	auto *gpu_NODE_A = init_gpu_arr(NODE_A, size);
	auto *gpu_NODE_B = init_gpu_arr(NODE_B, size);
	auto *gpu_NODE_D = init_gpu_arr(NODE_D, size);
	auto *gpu_const_NODE_D = init_gpu_arr(const_NODE_D, size);
	auto *gpu_NODE_RHS = init_gpu_arr(NODE_RHS, size);
	auto *gpu_NODE_RINV = init_gpu_arr(NODE_RINV, size);
	auto *gpu_NODE_AREA = init_gpu_arr(NODE_AREA, size);
	auto *gpu_has_spike = init_gpu_arr(has_spike, size);
	auto *gpu_spike_on = init_gpu_arr(spike_on, size);
	auto *gpu_g_exc = init_gpu_arr(g_exc, size);
	auto *gpu_g_inh_A = init_gpu_arr(g_inh_A, size);
	auto *gpu_g_inh_B = init_gpu_arr(g_inh_B, size);
	auto *gpu_factor = init_gpu_arr(factor, size);

	// Point to device pointer in host struct
	// states
	S->Vm = gpu_Vm;
	S->n = gpu_n;
	S->m = gpu_m;
	S->h = gpu_h;
	S->l = gpu_l;
	S->s = gpu_s;
	S->p = gpu_p;
	S->hc = gpu_hc;
	S->mc = gpu_mc;
	S->cai = gpu_cai;
	S->I_Ca = gpu_I_Ca;
	S->NODE_A = gpu_NODE_A;
	S->NODE_B = gpu_NODE_B;
	S->NODE_D = gpu_NODE_D;
	S->const_NODE_D = gpu_const_NODE_D;
	S->NODE_RHS = gpu_NODE_RHS;
	S->NODE_RINV = gpu_NODE_RINV;
	S->NODE_AREA = gpu_NODE_AREA;
	S->has_spike = gpu_has_spike;
	S->spike_on = gpu_spike_on;
	S->g_exc = gpu_g_exc;
	S->g_inh_A = gpu_g_inh_A;
	S->g_inh_B = gpu_g_inh_B;
	S->factor = gpu_factor;
	// parameters
	P->nrn_start_seg = gpu_nrn_start_seg;
	P->models = gpu_models;
	P->Cm = gpu_Cm;
	P->gnabar = gpu_gnabar;
	P->gkbar = gpu_gkbar;
	P->gl = gpu_gl;
	P->Ra = gpu_Ra;
	P->diam = gpu_diam;
	P->length = gpu_length;
	P->ena = gpu_ena;
	P->ek = gpu_ek;
	P->el = gpu_el;
	P->gkrect = gpu_gkrect;
	P->gcaN = gpu_gcaN;
	P->gcaL = gpu_gcaL;
	P->gcak = gpu_gcak;
	P->E_ex = gpu_E_ex;
	P->E_inh = gpu_E_inh;
	P->tau_exc = gpu_tau_exc;
	P->tau_inh1 = gpu_tau_inh1;
	P->tau_inh2 = gpu_tau_inh2;

	// allocate States struct to the device
	cudaMalloc((States **) &dev_S, sizeof(States));
	cudaMemcpy(dev_S, S, sizeof(States), cudaMemcpyHostToDevice);
	// allocate Parameters struct to the device
	cudaMalloc((Parameters **) &dev_P, sizeof(Parameters));
	cudaMemcpy(dev_P, P, sizeof(Parameters), cudaMemcpyHostToDevice);

	// call kernel
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; sim_iter++) {
		some_kernel<<<10000 / 256 + 1, 256>>>(dev_S, dev_P, size); // block size need to be a multiply of 256
	}

	CHECK(cudaDeviceSynchronize());

	// Copy result to host:
	cudaMemcpy(Vm, gpu_Vm, size * sizeof(*Vm), cudaMemcpyDeviceToHost);

	// Print some result
	std::cout << Vm[size-10] << std::endl;

//	CHECK(cudaFree(struct_Vm.gpu_array));
//	CHECK(cudaFree(gpu_Vm));
}

int main(int argc, char **argv) {
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s struct of array at ", argv[0]);
	printf("device %d: %s \n", dev, deviceProp.name);
	//
	CHECK(cudaSetDevice(dev));
	//
	simulate();
	// reset device
	CHECK(cudaDeviceReset());
}
