#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <ctime>
#include <stdexcept>
#include <random>

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <unistd.h>

#include <thread>
#include <chrono>

#include "Group.cpp"

#ifdef __JETBRAINS_IDE__
#define __host__
#define __shared__
#define __global__
#endif

// ============================================
//               D  A  C
// ============================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <pthread.h>
#include <iostream>
#include <math.h>

using namespace std;

#include <termios.h>

static struct termios stored_settings, new_settings;
static int peek_character = -1;

void reset_keypress(void) {
	tcsetattr(0, TCSANOW, &stored_settings);
	return;
}

void set_keypress(void) {
	tcgetattr(0, &stored_settings);

	new_settings = stored_settings;

	// disable canonical mode, and set buffer size to 1 byte
	new_settings.c_lflag &= (~ICANON);
	new_settings.c_lflag &= ~ECHO;
	new_settings.c_lflag &= ~ISIG;
	new_settings.c_cc[VTIME] = 0;
	new_settings.c_cc[VMIN] = 1;

	atexit(reset_keypress);
	tcsetattr(0, TCSANOW, &new_settings);
	return;
}

int readch() {
	char ch;
	if (peek_character != -1) {
		ch = peek_character;
		peek_character = -1;
		return ch;
	}
	read(0, &ch, 1);
	return ch;
}

#define INITGUID

#include "include/stubs.h"
#include "include/ioctl.h"
#include "include/ifc_ldev.h"
#include <errno.h>

typedef IDaqLDevice *(*CREATEFUNCPTR)(ULONG Slot);

CREATEFUNCPTR CreateInstance;

unsigned short *data1;
unsigned int *sync1;

void errorchk(bool condition, string text, char* err_text="failed") {
	cout << text << " ... ";
	if (condition) {
		cout << "ERROR (" << err_text << ")" << endl;
		cout << "FAILED !" << endl;
		reset_keypress();
		exit(0);
	} else {
		cout << "OK" << endl;
	}
}

void errorchk(bool condition, HRESULT result, string text) {
	cout << hex << text << " ... ";
	if (condition) {
		cout << "ERROR" << endl;
		cout << "FAILED !" << endl;
		reset_keypress();
		exit(0);
	} else {
		cout << hex << result << " OK" << endl;
	}
}


// ============================================
//               D  A  C
// ============================================


using namespace std;

const unsigned int syn_outdegree = 27;
const unsigned int neurons_in_ip = 196;
const unsigned int neurons_in_moto = 169;
const unsigned int neurons_in_group = 20;
const unsigned int neurons_in_afferent = 196;

const int skin_stim_time = 25;
const float INH_COEF = 1.0f;

// 6 CMS = 125 [ms]
// 15 CMS = 50 [ms]
// 21 CMS = 25 [ms]

// stuff variable
unsigned int global_id = 0;
const float T_sim = 10000;
const float SIM_STEP = 0.25;
const unsigned int sim_time_in_step = (unsigned int)(T_sim / SIM_STEP);

pthread_t thread1;

float motoneuron_voltage = 0;
bool gpu_is_run = true;

void *parallel_dac_func(void *arg) {
	ULONG mem_buffer_size = 131072;
	SLOT_PAR slot_param;
	ASYNC_PAR async_par;
	ASYNC_PAR async_par_adc;
	ADC_PAR adc_par;
	DAC_PAR dac_par;
	IDaqLDevice *device;
	PLATA_DESCR_U2 plata_descr;
	HANDLE handle;
	PVOID dll_handle;
	ULONG iresult;
	HRESULT hresult;
	LUnknown *pIUnknown;

	set_keypress();

	// load the dynamic shared object (shared library). RTLD_LAZY - perform lazy binding
	dll_handle = dlopen("/home/alex/Programs/drivers/lcomp/liblcomp.so", RTLD_LAZY);
	errorchk(!dll_handle, "open DLL", dlerror());

	// return the address where that symbol is loaded into memory
	CreateInstance = (CREATEFUNCPTR) dlsym(dll_handle, "CreateInstance");
	errorchk(dlerror() != NULL, "create instance");

	// create an object which related with a specific virtual slot (default 0)
	pIUnknown = CreateInstance(0);
	errorchk(pIUnknown == NULL, "call create instance");

	// get a pointer to the interface
	hresult = pIUnknown->QueryInterface(IID_ILDEV, (void **) &device);
	errorchk(hresult != S_OK, hresult, "query interface");

	// close an interface
	pIUnknown->Release();
	errorchk(false, "free IUnknown");

	// open an appropriate link of the board driver
	handle = device->OpenLDevice();
	errorchk(false, handle, "open device");

	// get an information of the specific virtual slot
	device->GetSlotParam(&slot_param);
	errorchk(false, "get slot parameters");

	// load a BIOS to the board
	iresult = device->LoadBios("e154");
	errorchk(iresult == 0, "load BIOS");

	// Test for board availability (always success)
	iresult = device->PlataTest();
	errorchk(iresult != 0, "plata test");

	// read an user Flash
	device->ReadPlataDescr(&plata_descr);
	errorchk(false, "read the board description");

	// allocate memory for a big ring buffer
	device->RequestBufferStream(&mem_buffer_size);
	errorchk(false, "request buffer stream");

	cout << "allocated size " << mem_buffer_size << endl;

	// fill DAC parameters
	adc_par.t1.s_Type = L_ADC_PARAM;  // тип структуры
	adc_par.t1.AutoInit = 1;          // флаг указывающий на тип сбора данных 0 - однократный 1 -циклически
	adc_par.t1.dRate = 100.0;         // частота опроса каналов в кадре (кГц)
	adc_par.t1.dKadr = 0;             // интервал между кадрами (мс), фактически определяет скоростьсбора данных;
	adc_par.t1.dScale = 0;            // масштаб работы таймера для 1250 или делителя для 1221
	adc_par.t1.AdChannel = 0;         // номер канала, выбранный для аналоговой синхронизации
	adc_par.t1.AdPorog = 0;           // пороговое значение для аналоговой синхронизации в коде АЦП
	adc_par.t1.NCh = 1;               // количество опрашиваемых в кадре каналов (для E154 макс. 16)
	adc_par.t1.Chn[0] = 1;          // массив с номерами каналов и усилением на них,
	adc_par.t1.FIFO = 4096;           // размер половины аппаратного буфера FIFO на плате
	adc_par.t1.IrqStep = 4096;        // шаг генерации прерываний
	adc_par.t1.Pages = 32;            // размер кольцевого буфера в шагах прерываний
	adc_par.t1.IrqEna = 1;            // разрешение генерации прерывания от платы (1/0);
	adc_par.t1.AdcEna = 1;            // разрешение работы AЦП (1/0)

	// 0 - нет синхронизации
	// 1 - цифровая синхронизация старта, остальные параметры синхронизации не используются
	// 2 - покадровая синхронизация, остальные параметры синхронизации не используются
	// 3 - аналоговая синхронизация старта по выбранному каналу АЦП
	adc_par.t1.SynchroType = 0;

	// 0 - аналоговая синхронизация по уровню
	// 1 - аналоговая синхронизация по переходу
	adc_par.t1.SynchroSensitivity = 0;

	// 0 - по уровню «выше» или переходу «снизу-вверх»
	// 1 - по уровню «ниже» или переходу «сверху-вниз»
	adc_par.t1.SynchroMode = 0;

	// fill inner parameter's structure of data collection values ​​from the structure ADC_PAR, DAC_PAR
	device->FillDAQparameters(&adc_par.t1);
	errorchk(false, "fill DAQ parameters");

	// setup the ADC/DAC board setting based on specific i/o parameters
	device->SetParametersStream(&adc_par.t1, &mem_buffer_size, (void **) &data1, (void **) &sync1, L_STREAM_ADC);
	errorchk(false, "set ADC parameters stream");

	// show slot parameters
	cout << "Base              : " << hex << slot_param.Base << endl;
	cout << "BaseL             : " << slot_param.BaseL << endl;
	cout << "Mem               : " << slot_param.Mem << endl;
	cout << "MemL              : " << slot_param.MemL << endl;
	cout << "Type              : " << slot_param.BoardType << endl;
	cout << "DSPType           : " << slot_param.DSPType << endl;
	cout << "Irq               : " << slot_param.Irq << endl;

	// show properties
	cout << "SerNum            : " << plata_descr.t7.SerNum << endl;
	cout << "BrdName           : " << plata_descr.t7.BrdName << endl;
	cout << "Rev               : " << plata_descr.t7.Rev << endl;
	cout << "DspType           : " << plata_descr.t7.DspType << endl;
	cout << "IsDacPresent      : " << plata_descr.t7.IsDacPresent << endl;
	cout << "Quartz            : " << dec << plata_descr.t7.Quartz << endl;

	// show ADC parameters
	cout << "Buffer size (word): " << mem_buffer_size << endl;
	cout << "Pages             : " << adc_par.t1.Pages << endl;
	cout << "IrqStep           : " << adc_par.t1.IrqStep << endl;
	cout << "FIFO              : " << adc_par.t1.FIFO << endl;
	cout << "Rate              : " << adc_par.t1.dRate << endl;

	// get an firmware version
	ULONG version = sync1[0xFF4 >> 2];
	cout << endl << "current firmware version 0x" << hex << version << dec << endl;

	// turn on an correction mode. Itself loads coefficients to the board
	device->EnableCorrection();
	errorchk(false, "enable correction mode");

	// init inner variables of the driver before starting collect a data
	device->InitStartLDevice();
	errorchk(false, "init start device");

	// start collect data from the board into the big ring buffer
	device->StartLDevice();
	errorchk(false, "START device");

	// write data to the DAC
	async_par.s_Type = L_ASYNC_DAC_OUT;
	async_par.Chn[0] = 1;  //
	async_par.Mode = 0;    // number of DAC (0/1). Setup different modes at configuration
	async_par.Data[0] = 0;
	device->IoAsync(&async_par);

	async_par_adc.s_Type = L_ASYNC_ADC_INP;
	async_par_adc.NCh = 1;
	async_par_adc.Chn[0] = 0x01;
	async_par_adc.Mode = 0;
	async_par_adc.Rate = 100;


	while (gpu_is_run) {
		async_par.Data[0] = motoneuron_voltage > 0? motoneuron_voltage / 5.0 * 0x7F : 0;
		device->IoAsync(&async_par);

		usleep(250); // std::this_thread::sleep_for(chrono::microseconds(250));

//		if(device->IoAsync(&async_par_adc) != L_SUCCESS){
//			cout << "Failed read data" << endl;
//		}
//		cout << async_par.Data[0] << " -> "<< (short)async_par_adc.Data[0] << endl;  // ADC data */
	}

	async_par.Data[0] = 0;   // data for DAC
	device->IoAsync(&async_par);

	errorchk(false, "TEST");

	// stop collecting data from the board
	device->StopLDevice();
	errorchk(false, "STOP device");

	// finishing work with the board
	device->CloseLDevice();
	errorchk(false, "close device");

	reset_keypress();

	// close an dll handle (decrements the reference count on the dynamic library handle)
	if (dll_handle)
		dlclose(dll_handle);

	// you are awesome
	cout << endl << "SUCCESS" << endl;

	return 0;


}

__host__
int ms_to_step(float ms) { return (int)(ms / SIM_STEP); }

struct SynapseMetadata{
	// struct for human-readable initialization of connectomes
	int post_id;
	int synapse_delay;
	float synapse_weight;

	SynapseMetadata() = default;
	SynapseMetadata(int post_id, float synapse_delay, float synapse_weight){
		this->post_id = post_id;
		this->synapse_delay = static_cast<int>(synapse_delay * (1 / SIM_STEP) + 0.5); // round
		this->synapse_weight = synapse_weight;
	}
};

Group form_group(string group_name, int nrns_in_group = neurons_in_group) {
	// form structs of neurons global ID and groups name
	Group group = Group();

	group.group_name = group_name;
	group.id_start = global_id;
	group.id_end = global_id + nrns_in_group - 1;
	group.group_size = nrns_in_group;

	global_id += nrns_in_group;

	printf("Formed %s IDs [%d ... %d] = %d\n",
		   group_name.c_str(), global_id - nrns_in_group, global_id - 1, nrns_in_group);

	return group;
}

// Form neuron groups
Group C1 = form_group("C1");
Group C2 = form_group("C2");
Group C3 = form_group("C3");
Group C4 = form_group("C4");
Group C5 = form_group("C5");

Group D1_1 = form_group("D1_1");
Group D1_2 = form_group("D1_2");
Group D1_3 = form_group("D1_3");
Group D1_4 = form_group("D1_4");

Group D2_1 = form_group("D2_1");
Group D2_2 = form_group("D2_2");
Group D2_3 = form_group("D2_3");
Group D2_4 = form_group("D2_4");

Group D3_1 = form_group("D3_1");
Group D3_2 = form_group("D3_2");
Group D3_3 = form_group("D3_3");
Group D3_4 = form_group("D3_4");

Group D4_1 = form_group("D4_1");
Group D4_2 = form_group("D4_2");
Group D4_3 = form_group("D4_3");
Group D4_4 = form_group("D4_4");

Group D5_1 = form_group("D5_1");
Group D5_2 = form_group("D5_2");
Group D5_3 = form_group("D5_3");
Group D5_4 = form_group("D5_4");

Group G1_1 = form_group("G1_1");
Group G1_2 = form_group("G1_2");
Group G1_3 = form_group("G1_3");

Group G2_1 = form_group("G2_1");
Group G2_2 = form_group("G2_2");
Group G2_3 = form_group("G2_3");

Group G3_1 = form_group("G3_1");
Group G3_2 = form_group("G3_2");
Group G3_3 = form_group("G3_3");

Group G4_1 = form_group("G4_1");
Group G4_2 = form_group("G4_2");
Group G4_3 = form_group("G4_3");

Group G5_1 = form_group("G5_1");
Group G5_2 = form_group("G5_2");
Group G5_3 = form_group("G5_3");

Group IP_E = form_group("IP_E", neurons_in_ip);
Group MP_E = form_group("MP_E", neurons_in_moto);
Group EES = form_group("EES");
Group Ia = form_group("Ia", neurons_in_afferent);

Group inh_group3 = form_group("inh_group3");
Group inh_group4 = form_group("inh_group4");
Group inh_group5 = form_group("inh_group5");

Group ees_group1 = form_group("ees_group1");
Group ees_group2 = form_group("ees_group2");
Group ees_group3 = form_group("ees_group3");
Group ees_group4 = form_group("ees_group4");

// Global vectors of SynapseMetadata of synapses for each neuron
vector<vector<SynapseMetadata>> metadatas(global_id, vector<SynapseMetadata>());

// Parameters (const)
const float C = 100;        // [pF] membrane capacitance
const float V_rest = -72;   // [mV] resting membrane potential
const float V_thld = -55;   // [mV] spike threshold
const float k = 0.7;          // [pA * mV-1] constant ("1/R")
const float a = 0.02;         // [ms-1] time scale of the recovery variable U_m. Higher a, the quicker recovery
const float b = 0.2;          // [pA * mV-1] sensitivity of U_m to the sub-threshold fluctuations of the V_m
const float c = -80;        // [mV] after-spike reset value of V_m
const float d = 6;          // [pA] after-spike reset value of U_m
const float V_peak = 35;    // [mV] spike cutoff value

const unsigned int steps_activation_C0 = (unsigned int)(5 * skin_stim_time / SIM_STEP);
const unsigned int steps_activation_C1 = (unsigned int)(6 * skin_stim_time / SIM_STEP);

__global__
void sim_kernel(float* old_v,
				float* old_u,
				float* nrn_current,
				int* nrn_ref_time,
				int* nrn_ref_time_timer,
				int* synapses_number,
				bool* has_spike,
				int** synapses_post_nrn_id,
				int** synapses_delay,
				int** synapses_delay_timer,
				float** synapses_weight,
				unsigned int nrn_size,
				int activated_C_,
				int shift_time_by_step,
				float* gpu_motoneuron_voltage,
//				float* voltage,
				int sim_iter) {

	// get id of the thread
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_id == 0) {
		*gpu_motoneuron_voltage = 0;
	}

	__syncthreads();


	// neuron (tid = neuron id) stride loop (0, 1024, 1, 1025 ...)
	for (int tid = thread_id; tid < nrn_size; tid += blockDim.x * gridDim.x) {
		if (activated_C_ == 0){
			if (996 <= tid && tid <= 1164) {
				atomicAdd(gpu_motoneuron_voltage, V_rest);
			}
			continue;
		}
		// C1
		if (0 <= tid && tid <= 19
			&& shift_time_by_step <= sim_iter
			&& sim_iter < (100 + shift_time_by_step)
			&& (sim_iter % 20 == 0)) {
			nrn_current[tid] = 5000; // enough for spike
		}
		// C2
		if (20 <= tid && tid <= 39
			&& (100 + shift_time_by_step) <= sim_iter
			&& sim_iter < (200 + shift_time_by_step)
			&& (sim_iter % 20 == 0)) {
			nrn_current[tid] = 5000; // enough for spike
		}
		// C3
		if (40 <= tid && tid <= 59
			&& (200 + shift_time_by_step) <= sim_iter
			&& sim_iter < (300 + shift_time_by_step)
			&& (sim_iter % 20 == 0)) {
			nrn_current[tid] = 5000;
		}
		// C4
		if (60 <= tid && tid <= 79
			&& (300 + shift_time_by_step) <= sim_iter
			&& sim_iter < (500 + shift_time_by_step)
			&& (sim_iter % 20 == 0)) {
			nrn_current[tid] = 5000;
		}
		// C5
		if (80 <= tid && tid <= 99
			&& (500 + shift_time_by_step) <= sim_iter
			&& sim_iter < (600 + shift_time_by_step)
			&& (sim_iter % 20 == 0)) {
			nrn_current[tid] = 5000;
		}

		// EES
		if (1165 <= tid && tid <= 1184 && (sim_iter % 100 == 0)) {
			nrn_current[tid] = 5000;
		}

		if (nrn_ref_time_timer[tid] > 0)
			nrn_current[tid] = 0;

		float V_old = old_v[tid];
		float U_old = old_u[tid];
		float I_current = nrn_current[tid];

		if (I_current > 10000)
			I_current = 10000;
		if (I_current < -10000)
			I_current = -10000;

		// re-calculate V_m and U_m
		float V_m = V_old + SIM_STEP * (k * (V_old - V_rest) * (V_old - V_thld) - U_old + I_current) / C;
		float U_m = U_old + SIM_STEP * a * (b * (V_old - V_rest) - U_old);

		// set bottom border of the membrane potential
		if (V_m < c)
			V_m = c;
		// set top border of the membrane potential
		if (V_m >= V_thld)
			V_m = V_peak;

//		int index = sim_iter + tid * sim_time_in_step;
//		voltage[index] = V_m;

		if (996 <= tid && tid <= 1164) {
			atomicAdd(gpu_motoneuron_voltage, V_m);
		}

		// threshold crossing (spike)
		if (V_m >= V_thld) {
			// set spike status
			has_spike[tid] = true;
			// redefine V_old and U_old
			old_v[tid] = c;
			old_u[tid] += d;
			// set the refractory period
			nrn_ref_time_timer[tid] = nrn_ref_time[tid];
		} else {
			// redefine V_old and U_old
			old_v[tid] = V_m;
			old_u[tid] = U_m;
		}

		// wait all threads
		__syncthreads();

		// pointers to current neuronID synapses_delay_timer (decrease array calls)
		int *ptr_delay_timers = synapses_delay_timer[tid];
		// synapse updating loop
		for (int syn_id = 0; syn_id < synapses_number[tid]; syn_id++) {
			// add synaptic delay if neuron has spike
			if (has_spike[tid] && ptr_delay_timers[syn_id] == -1) {
				ptr_delay_timers[syn_id] = synapses_delay[tid][syn_id];
			}
			// if synaptic delay is zero it means the time when synapse increase I by synaptic weight
			if (ptr_delay_timers[syn_id] == 0) {
				// post neuron ID = synapses_post_nrn_id[tid][syn_id], thread-safe (!)
				atomicAdd(&nrn_current[ synapses_post_nrn_id[tid][syn_id] ], synapses_weight[tid][syn_id]);
				// make synapse timer a "free" for next spikes
				ptr_delay_timers[syn_id] = -1;
			}
			// update synapse delay timer
			if (ptr_delay_timers[syn_id] > 0) {
				ptr_delay_timers[syn_id]--;
			}
		} // end synapse updating loop

		// reset spike flag of the current neuron
		has_spike[tid] = false;

		// update currents of the neuron
		if (I_current != 0) {
			// decrease current potential for positive and negative current
			if (I_current > 0) nrn_current[tid] = I_current / 2;
			if (I_current < 0) nrn_current[tid] = I_current / 1.1f;
			// avoid the near value to 0
			if (I_current > 0 && I_current <= 1) nrn_current[tid] = 0;
			if (I_current <= 0 && I_current >= -1) nrn_current[tid] = 0;
		}

		// update the refractory period timer
		if (nrn_ref_time_timer[tid] > 0)
			nrn_ref_time_timer[tid]--;
	} // end of neuron stride loop
}

void connect_fixed_outdegree(Group pre_neurons, Group post_neurons, float syn_delay, float weight, int outdegree = syn_outdegree) {
	// connect neurons with uniform distribution and normal distributon for syn delay and weight
	weight *= (100 * 0.7);
	random_device rd;
	mt19937 gen(rd());  // Initialize pseudo-random number generator

	uniform_int_distribution<int> id_distr(post_neurons.id_start, post_neurons.id_end);
	normal_distribution<float> weight_distr(weight, 2);
	normal_distribution<float> delay_distr(syn_delay, 0.1);

	for (int pre_id = pre_neurons.id_start; pre_id <= pre_neurons.id_end; pre_id++) {
		for (int i = 0; i < outdegree; i++) {
			int rand_post_id = id_distr(gen);
			float syn_delay_dist = syn_delay;   // ToDo replace after tuning : delay_distr(gen);
			float syn_weight_dist = weight;     // ToDo replace after tuning : weight_distr(gen);
			metadatas.at(pre_id).push_back(SynapseMetadata(rand_post_id, syn_delay_dist, syn_weight_dist));
		}
	}

	printf("Connect %s with %s (1:%d). W=%.2f, D=%.1f\n",
		   pre_neurons.group_name.c_str(),
		   post_neurons.group_name.c_str(),
		   post_neurons.group_size,
		   weight,
		   syn_delay);
}

void init_extensor() {
	connect_fixed_outdegree(C3, inh_group3, 0.5, 15.0);
	connect_fixed_outdegree(C4, inh_group4, 0.5, 15.0);
	connect_fixed_outdegree(C5, inh_group5, 0.5, 15.0);

	connect_fixed_outdegree(inh_group3, G1_3, 2.8, 20.0);

	connect_fixed_outdegree(inh_group4, G1_3, 1.0, 20.0);
	connect_fixed_outdegree(inh_group4, G2_3, 1.0, 20.0);

	connect_fixed_outdegree(inh_group5, G1_3, 2.0, 20.0);
	connect_fixed_outdegree(inh_group5, G2_3, 1.0, 20.0);
	connect_fixed_outdegree(inh_group5, G3_3, 1.0, 20.0);
	connect_fixed_outdegree(inh_group5, G4_3, 1.0, 20.0);

	/// D1
	// input from sensory
	connect_fixed_outdegree(C1, D1_1, 1, 0.4);
	connect_fixed_outdegree(C1, D1_4, 1, 0.4);
	connect_fixed_outdegree(C2, D1_1, 1, 0.4);
	connect_fixed_outdegree(C2, D1_4, 1, 0.4);
	// input from EES
	connect_fixed_outdegree(EES, D1_1, 2, 10); // ST value (?)
	connect_fixed_outdegree(EES, D1_4, 2, 10); // ST value (?)
	// inner connectomes
	connect_fixed_outdegree(D1_1, D1_2, 1, 1.0);
	connect_fixed_outdegree(D1_1, D1_3, 1, 10.0);
	connect_fixed_outdegree(D1_2, D1_1, 1, 7.0);
	connect_fixed_outdegree(D1_2, D1_3, 1, 10.0);
	connect_fixed_outdegree(D1_3, D1_1, 1, -10 * INH_COEF);
	connect_fixed_outdegree(D1_3, D1_2, 1, -10 * INH_COEF);
	connect_fixed_outdegree(D1_4, D1_3, 3, -20 * INH_COEF);
	// output to
	connect_fixed_outdegree(D1_3, G1_1, 3, 8);
	connect_fixed_outdegree(D1_3, ees_group1, 1.0, 60);

	// EES group connectomes
	connect_fixed_outdegree(ees_group1, ees_group2, 1.0, 20.0);

	/// D2
	// input from Sensory
	connect_fixed_outdegree(C2, D2_1, 1, 0.8);
	connect_fixed_outdegree(C2, D2_4, 1, 0.8);
	connect_fixed_outdegree(C3, D2_1, 1, 0.8);
	connect_fixed_outdegree(C3, D2_4, 1, 0.8);
	// input from Group (1)
	connect_fixed_outdegree(ees_group1, D2_1, 1.7, 0.8);
	connect_fixed_outdegree(ees_group1, D2_4, 1.7, 1.0);
	// inner connectomes
	connect_fixed_outdegree(D2_1, D2_2, 1.0, 3.0);
	connect_fixed_outdegree(D2_1, D2_3, 1.0, 10.0);
	connect_fixed_outdegree(D2_2, D2_1, 1.0, 7.0);
	connect_fixed_outdegree(D2_2, D2_3, 1.0, 20.0);
	connect_fixed_outdegree(D2_3, D2_1, 1.0, -20 * INH_COEF);
	connect_fixed_outdegree(D2_3, D2_2, 1.0, -20 * INH_COEF);
	connect_fixed_outdegree(D2_4, D2_3, 2.0, -20 * INH_COEF);
	// output to generator
	connect_fixed_outdegree(D2_3, G2_1, 1.0, 8);

	// EES group connectomes
	connect_fixed_outdegree(ees_group2, ees_group3, 1.0, 20.0);

	/// D3
	// input from Sensory
	connect_fixed_outdegree(C3, D3_1, 1, 0.5);
	connect_fixed_outdegree(C3, D3_4, 1, 0.5);
	connect_fixed_outdegree(C4, D3_1, 1, 0.5);
	connect_fixed_outdegree(C4, D3_4, 1, 0.5);
	// input from Group (2)
	connect_fixed_outdegree(ees_group2, D3_1, 1, 1.2);
	connect_fixed_outdegree(ees_group2, D3_4, 1, 1.2);
	// inner connectomes
	connect_fixed_outdegree(D3_1, D3_2, 1.0, 3.0);
	connect_fixed_outdegree(D3_1, D3_3, 1.0, 10.0);
	connect_fixed_outdegree(D3_2, D3_1, 1.0, 7.0);
	connect_fixed_outdegree(D3_2, D3_3, 1.0, 20.0);
	connect_fixed_outdegree(D3_3, D3_1, 1.0, -10 * INH_COEF);
	connect_fixed_outdegree(D3_3, D3_2, 1.0, -10 * INH_COEF);
	connect_fixed_outdegree(D3_4, D3_3, 2.0, -10 * INH_COEF);
	// output to generator
	connect_fixed_outdegree(D3_3, G3_1, 1, 25.0);
	// suppression of the generator
	connect_fixed_outdegree(D3_3, G1_3, 1.5, 30.0);

	// EES group connectomes
	connect_fixed_outdegree(ees_group3, ees_group4, 2.0, 20.0);

	/// D4
	// input from Sensory
	connect_fixed_outdegree(C4, D4_1, 1, 0.5);
	connect_fixed_outdegree(C4, D4_4, 1, 0.5);
	connect_fixed_outdegree(C5, D4_1, 1, 0.5);
	connect_fixed_outdegree(C5, D4_4, 1, 0.5);
	// input from Group (3)
	connect_fixed_outdegree(ees_group3, D4_1, 1, 1.2);
	connect_fixed_outdegree(ees_group3, D4_4, 1, 1.2);
	// inner connectomes
	connect_fixed_outdegree(D4_1, D4_2, 1.0, 3.0);
	connect_fixed_outdegree(D4_1, D4_3, 1.0, 10.0);
	connect_fixed_outdegree(D4_2, D4_1, 1.0, 7.0);
	connect_fixed_outdegree(D4_2, D4_3, 1.0, 20.0);
	connect_fixed_outdegree(D4_3, D4_1, 1.0, -20 * INH_COEF);
	connect_fixed_outdegree(D4_3, D4_2, 1.0, -20 * INH_COEF);
	connect_fixed_outdegree(D4_4, D4_3, 2.0, -20 * INH_COEF);
	// output to the generator
	connect_fixed_outdegree(D4_3, G4_1, 3.0, 20.0);
	// suppression of the generator
	connect_fixed_outdegree(D4_3, G2_3, 1.0, 30.0);

	/// D5
	// input from Sensory
	connect_fixed_outdegree(C5, D5_1, 1, 0.5);
	connect_fixed_outdegree(C5, D5_4, 1, 0.5);
	// input from Group (4)
	connect_fixed_outdegree(ees_group4, D5_1, 1.0, 1.1);
	connect_fixed_outdegree(ees_group4, D5_4, 1.0, 1.0);
	// inner connectomes
	connect_fixed_outdegree(D5_1, D5_2, 1.0, 3.0);
	connect_fixed_outdegree(D5_1, D5_3, 1.0, 15.0);
	connect_fixed_outdegree(D5_2, D5_1, 1.0, 7.0);
	connect_fixed_outdegree(D5_2, D5_3, 1.0, 20.0);
	connect_fixed_outdegree(D5_3, D5_1, 1.0, -20 * INH_COEF);
	connect_fixed_outdegree(D5_3, D5_2, 1.0, -20 * INH_COEF);
	connect_fixed_outdegree(D5_4, D5_3, 2.5, -20 * INH_COEF);
	// output to the generator
	connect_fixed_outdegree(D5_3, G5_1, 3, 8.0);
	// suppression of the genearator
	connect_fixed_outdegree(D5_3, G1_3, 1.0, 30.0);
	connect_fixed_outdegree(D5_3, G2_3, 1.0, 30.0);
	connect_fixed_outdegree(D5_3, G3_3, 1.0, 30.0);
	connect_fixed_outdegree(D5_3, G4_3, 1.0, 30.0);

	/// G1
	// inner connectomes
	connect_fixed_outdegree(G1_1, G1_2, 1.0, 10.0);
	connect_fixed_outdegree(G1_1, G1_3, 1.0, 15.0);
	connect_fixed_outdegree(G1_2, G1_1, 1.0, 10.0);
	connect_fixed_outdegree(G1_2, G1_3, 1.0, 15.0);
	connect_fixed_outdegree(G1_3, G1_1, 0.7, -70 * INH_COEF);
	connect_fixed_outdegree(G1_3, G1_2, 0.7, -70 * INH_COEF);
	// output to IP_E
	connect_fixed_outdegree(G1_1, IP_E, 3, 25.0);
	connect_fixed_outdegree(G1_1, IP_E, 3, 25.0);

	/// G2
	// inner connectomes
	connect_fixed_outdegree(G2_1, G2_2, 1.0, 10.0);
	connect_fixed_outdegree(G2_1, G2_3, 1.0, 20.0);
	connect_fixed_outdegree(G2_2, G2_1, 1.0, 10.0);
	connect_fixed_outdegree(G2_2, G2_3, 1.0, 20.0);
	connect_fixed_outdegree(G2_3, G2_1, 0.5, -30 * INH_COEF);
	connect_fixed_outdegree(G2_3, G2_2, 0.5, -30 * INH_COEF);
	// output to IP_E
	connect_fixed_outdegree(G2_1, IP_E, 1.0, 65.0);
	connect_fixed_outdegree(G2_2, IP_E, 1.0, 65.0);

	/// G3
	// inner connectomes
	connect_fixed_outdegree(G3_1, G3_2, 1.0, 14.0);
	connect_fixed_outdegree(G3_1, G3_3, 1.0, 20.0);
	connect_fixed_outdegree(G3_2, G3_1, 1.0, 12.0);
	connect_fixed_outdegree(G3_2, G3_3, 1.0, 20.0);
	connect_fixed_outdegree(G3_3, G3_1, 0.5, -30 * INH_COEF);
	connect_fixed_outdegree(G3_3, G3_2, 0.5, -30 * INH_COEF);
	// output to IP_E
	connect_fixed_outdegree(G3_1, IP_E, 2, 25.0);
	connect_fixed_outdegree(G3_1, IP_E, 2, 25.0);

	/// G4
	// inner connectomes
	connect_fixed_outdegree(G4_1, G4_2, 1.0, 10.0);
	connect_fixed_outdegree(G4_1, G4_3, 1.0, 10.0);
	connect_fixed_outdegree(G4_2, G4_1, 1.0, 5.0);
	connect_fixed_outdegree(G4_2, G4_3, 1.0, 10.0);
	connect_fixed_outdegree(G4_3, G4_1, 0.5, -30 * INH_COEF);
	connect_fixed_outdegree(G4_3, G4_2, 0.5, -30 * INH_COEF);
	// output to IP_E
	connect_fixed_outdegree(G4_1, IP_E, 1.0, 17.0);
	connect_fixed_outdegree(G4_1, IP_E, 1.0, 17.0);

	/// G5
	// inner connectomes
	connect_fixed_outdegree(G5_1, G5_2, 1.0, 7.0);
	connect_fixed_outdegree(G5_1, G5_3, 1.0, 10.0);
	connect_fixed_outdegree(G5_2, G5_1, 1.0, 7.0);
	connect_fixed_outdegree(G5_2, G5_3, 1.0, 10.0);
	connect_fixed_outdegree(G5_3, G5_1, 0.5, -30 * INH_COEF);
	connect_fixed_outdegree(G5_3, G5_2, 0.5, -30 * INH_COEF);
	// output to IP_E
	connect_fixed_outdegree(G5_1, IP_E, 2, 20.0);
	connect_fixed_outdegree(G5_1, IP_E, 2, 20.0);

	connect_fixed_outdegree(IP_E, MP_E, 1, 11);
	connect_fixed_outdegree(EES, MP_E, 2, 50);
	connect_fixed_outdegree(Ia, MP_E, 1, 1);
}

void save_result(int test_index,
				 float* voltage_recording,
				 int neurons_number) {
	// save results for each neuron (voltage/current/spikes)
	char cwd[256];
	ofstream myfile;

	getcwd(cwd, sizeof(cwd));
	printf("[Test #%d] Save results to: %s \n", test_index, cwd);

	string new_name = "/volt.dat";
	myfile.open(cwd + new_name);

	for(int nrn_id = 0; nrn_id < neurons_number; nrn_id++){
		myfile << nrn_id << " ";
		for(int sim_iter = 0; sim_iter < sim_time_in_step; sim_iter++)
			myfile << voltage_recording[sim_iter + nrn_id * sim_time_in_step] << " ";
		myfile << "\n";
	}

	myfile.close();
}

template <typename type>
void memcpyHtD(type* gpu, type* host, int size) {
	cudaMemcpy(gpu, host, sizeof(type) * size, cudaMemcpyHostToDevice);
}

template <typename type>
void memcpyDtH(type* host, type* gpu, int size) {
	cudaMemcpy(host, gpu, sizeof(type) * size, cudaMemcpyDeviceToHost);
}

template <typename type>
unsigned int datasize(int size) {
	return sizeof(type) * size;
}

template <typename type>
void init_array(type *array, int size, type value){
	for(int i = 0; i < size; i++)
		array[i] = value;
}



__host__
void simulate(int test_index) {
	int neurons_number = static_cast<int>(metadatas.size());

	float* gpu_old_v;
	float* gpu_old_u;
	int* gpu_nrn_ref_time;
	int* gpu_nrn_ref_timer;
	bool* gpu_has_spike;
	float* gpu_nrn_current;
	int* gpu_synapses_number;

//	float* gpu_voltage_recording;

	int synapses_number[neurons_number];

	float old_v[neurons_number];
	init_array<float>(old_v, neurons_number, V_rest);

	float old_u[neurons_number];
	init_array<float>(old_u, neurons_number, 0);

	int nrn_ref_time[neurons_number];
	init_array<int>(nrn_ref_time, neurons_number, ms_to_step(3.0));

	int nrn_ref_timer[neurons_number];
	init_array<int>(nrn_ref_timer, neurons_number, -1);

	bool has_spike[neurons_number];
	init_array<bool>(has_spike, neurons_number, false);

	float nrn_current[neurons_number];
	init_array<float>(nrn_current, neurons_number, 0);

//	float* voltage_recording = (float *)malloc(datasize<float *>(neurons_number * sim_time_in_step));
//	init_array<float>(voltage_recording, neurons_number * sim_time_in_step, -72);

	// init connectomes
	init_extensor();

	int **gpu_synapses_post_nrn_id, **synapses_post_nrn_id = (int **)malloc(datasize<int* >(neurons_number));
	int **gpu_synapses_delay, **synapses_delay = (int **)malloc(datasize<int* >(neurons_number));
	int **gpu_synapses_delay_timer, **synapses_delay_timer = (int **)malloc(datasize<int* >(neurons_number));
	float **gpu_synapses_weight, **synapses_weight = (float **)malloc(datasize<float* >(neurons_number));

	// fill arrays of synapses
	for(int neuron_id = 0; neuron_id < neurons_number; neuron_id++) {
		int syn_count = static_cast<int>(metadatas.at(neuron_id).size());

		int tmp_synapses_post_nrn_id[syn_count];
		int tmp_synapses_delay[syn_count];
		int tmp_synapses_delay_timer[syn_count];
		float tmp_synapses_weight[syn_count];

		int syn_id = 0;
		for(SynapseMetadata metadata : metadatas.at(neuron_id)) {
			tmp_synapses_post_nrn_id[syn_id] = metadata.post_id;
			tmp_synapses_delay[syn_id] = metadata.synapse_delay;
			tmp_synapses_delay_timer[syn_id] = -1;
			tmp_synapses_weight[syn_id] = metadata.synapse_weight;
			syn_id++;
		}

		synapses_number[neuron_id] = syn_count;

		cudaMalloc((void**)&synapses_post_nrn_id[neuron_id], datasize<int>(syn_count));
		cudaMalloc((void**)&synapses_delay[neuron_id], datasize<int>(syn_count));
		cudaMalloc((void**)&synapses_delay_timer[neuron_id], datasize<int>(syn_count));
		cudaMalloc((void**)&synapses_weight[neuron_id], datasize<float>(syn_count));

		cudaMemcpy(synapses_post_nrn_id[neuron_id], &tmp_synapses_post_nrn_id, datasize<int>(syn_count), cudaMemcpyHostToDevice);
		cudaMemcpy(synapses_delay[neuron_id], &tmp_synapses_delay, datasize<int>(syn_count), cudaMemcpyHostToDevice);
		cudaMemcpy(synapses_delay_timer[neuron_id], &tmp_synapses_delay_timer, datasize<int>(syn_count), cudaMemcpyHostToDevice);
		cudaMemcpy(synapses_weight[neuron_id], &tmp_synapses_weight, datasize<float>(syn_count), cudaMemcpyHostToDevice);
	}

	cudaMalloc((void ***)&gpu_synapses_post_nrn_id, datasize<int *>(neurons_number));
	memcpyHtD<int *>(gpu_synapses_post_nrn_id, synapses_post_nrn_id, neurons_number);

	cudaMalloc((void ***)&gpu_synapses_delay, datasize<int *>(neurons_number));
	memcpyHtD<int *>(gpu_synapses_delay, synapses_delay, neurons_number);

	cudaMalloc((void ***)&gpu_synapses_delay_timer, datasize<int *>(neurons_number));
	memcpyHtD<int *>(gpu_synapses_delay_timer, synapses_delay_timer, neurons_number);

	cudaMalloc((void ***)&gpu_synapses_weight, datasize<float *>(neurons_number));
	memcpyHtD<float *>(gpu_synapses_weight, synapses_weight, neurons_number);

	cudaMalloc(&gpu_old_v, datasize<float>(neurons_number));
	memcpyHtD<float>(gpu_old_v, old_v, neurons_number);

	cudaMalloc(&gpu_old_u, datasize<float>(neurons_number));
	memcpyHtD<float>(gpu_old_u, old_u, neurons_number);

	cudaMalloc(&gpu_has_spike, datasize<bool>(neurons_number));
	memcpyHtD<bool>(gpu_has_spike, has_spike, neurons_number);

	cudaMalloc(&gpu_nrn_ref_time, datasize<int>(neurons_number));
	memcpyHtD<int>(gpu_nrn_ref_time, nrn_ref_time, neurons_number);

	cudaMalloc(&gpu_nrn_ref_timer, datasize<int>(neurons_number));
	memcpyHtD<int>(gpu_nrn_ref_timer, nrn_ref_timer, neurons_number);

	cudaMalloc(&gpu_nrn_current, datasize<float>(neurons_number));
	memcpyHtD<float>(gpu_nrn_current, nrn_current, neurons_number);

	cudaMalloc(&gpu_synapses_number, datasize<int>(neurons_number));
	memcpyHtD<int>(gpu_synapses_number, synapses_number, neurons_number);

//	cudaMalloc(&gpu_voltage_recording, datasize<float>(neurons_number * sim_time_in_step));
//	memcpyHtD<float>(gpu_voltage_recording, voltage_recording, neurons_number * sim_time_in_step);

	int threads_per_block = 1024;
	int num_blocks = 1; //neurons_number / threads_per_block + 1;

	printf("Size of network: %i \n", neurons_number);
	printf("Start GPU with %d threads x %d blocks (Total: %d threads) \n",
		   threads_per_block, num_blocks, threads_per_block * num_blocks);

	int local_iter = 0;
	int activated_C_ = 0;
	int shift_time_by_step = 0;

	float * gpu_motoneuron_voltage;
	cudaMalloc((void**)&gpu_motoneuron_voltage, sizeof(float));
	cudaMemcpy(gpu_motoneuron_voltage, &motoneuron_voltage, sizeof(float), cudaMemcpyHostToDevice);

	chrono::time_point<chrono::system_clock> iter_t_start, iter_t_end, simulation_t_start, simulation_t_end;
	chrono::duration<double> elapsed_time_per_iter[sim_time_in_step];
	chrono::duration<double> waited_time_per_iter[sim_time_in_step];



	pthread_create(&thread1, NULL, parallel_dac_func, 0);

	simulation_t_start = chrono::system_clock::now();

	auto frame_time = std::chrono::microseconds(150);

	// GPU max T per step (4000 steps) <= 250 µm (0.25 ms)
	for (int sim_iter = 0; sim_iter < sim_time_in_step; sim_iter++) {
		// start measure time
		iter_t_start = chrono::system_clock::now();

		// if flexor C0 activated, find the end of it and change to C1
		if (activated_C_ == 0) {
			if (local_iter != 0 && local_iter % steps_activation_C0 == 0) {
				activated_C_ = 1; // change to C1
				local_iter = 0;   // reset local time iterator
				shift_time_by_step += steps_activation_C0;  // add constant 125 ms
			}
			// if extensor C1 activated, find the end of it and change to C0
		} else {
			if (local_iter != 0 && local_iter % steps_activation_C1 == 0) {
				activated_C_ = 0; // change to C0
				local_iter = 0;   // reset local time iterator
				shift_time_by_step += steps_activation_C1;  // add time equal to n_layers * 25 ms
			}
		}

		sim_kernel<<<num_blocks, threads_per_block>>>(gpu_old_v,
		                                              gpu_old_u,
		                                              gpu_nrn_current,
		                                              gpu_nrn_ref_time,
		                                              gpu_nrn_ref_timer,
		                                              gpu_synapses_number,
		                                              gpu_has_spike,
		                                              gpu_synapses_post_nrn_id,
		                                              gpu_synapses_delay,
		                                              gpu_synapses_delay_timer,
		                                              gpu_synapses_weight,
		                                              neurons_number,
		                                              activated_C_,
		                                              shift_time_by_step,
		                                              gpu_motoneuron_voltage,
//		                                              gpu_voltage_recording,
		                                              sim_iter);

		cudaMemcpy(&motoneuron_voltage, gpu_motoneuron_voltage, sizeof(float), cudaMemcpyDeviceToHost);

		local_iter++;

		// stop measure time
		iter_t_end = std::chrono::system_clock::now();

		motoneuron_voltage = (motoneuron_voltage / neurons_in_moto + 72) / 100 * 5 ;

		// save time difference
		auto elapsed = chrono::duration_cast<chrono::microseconds>(iter_t_end - iter_t_start);
//		elapsed_time_per_iter[sim_iter] = elapsed;

		if (elapsed < frame_time) {
//			waited_time_per_iter[sim_iter] = frame_time - elapsed;
			std::this_thread::sleep_for(frame_time - elapsed);
		}// else {
////			waited_time_per_iter[sim_iter] = chrono::microseconds(0);
//		}
	} // end of simulation iteration loop

	simulation_t_end = chrono::system_clock::now();

	gpu_is_run = false;

	pthread_join(thread1, NULL);

	double sum = 0;
	double wai = 0;

//	for (int i = 0; i < sim_time_in_step; i++) {
//		sum += elapsed_time_per_iter[i].count();
//		wai += waited_time_per_iter[i].count();
//		cout << fixed << i << " | " <<  (int)(elapsed_time_per_iter[i].count() * 10e5) << " µs, w=" << (int)(waited_time_per_iter[i].count() * 10e5) << endl;
//	}

	auto sim_time_diff = chrono::duration_cast<chrono::milliseconds>(simulation_t_end - simulation_t_start).count();
	printf("Elapsed %li ms (measured), used %.2f ms, waited %.2f ms, other %.2f | T_sim = %.2f ms\n",
		   sim_time_diff, sum * 1000, wai * 1000, sim_time_diff - (wai * 1000 + sum * 1000), T_sim);
	printf("%s x%f\n",
		   (double)(T_sim / sim_time_diff) > 1? "faster" : "slower", T_sim / sim_time_diff);

	// copy neurons/synapses array to the HOST
//	memcpyDtH<float>(voltage_recording, gpu_voltage_recording, neurons_number * sim_time_in_step);

	// tell the CPU to halt further processing until the CUDA kernel has finished doing its business
	cudaDeviceSynchronize();

//	save_result(test_index, voltage_recording, neurons_number);

	cudaDeviceReset();
}

int main() {
	simulate(0);

	return 0;
}