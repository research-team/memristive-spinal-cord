#include "core.cu"
#include "structs.h"

void init_network() {
	/**
	 * todo
	 */
	if (false) {
		auto e1 = form_group("e1", 1, GENERATOR);
		auto e2 = form_group("e2", 1, GENERATOR);
		auto om1 = form_group("om1", 10);
		auto moto = form_group("mns_E", 20);
		add_generator(e1, 10, 100000, 400);
		add_generator(e2, 15, 100000, 400);
		conn_generator(e1, om1, 1.1, 0.000020);
		conn_generator(e2, om1, 1.1, -0.000003);
		connect_fixed_indegree(om1, moto, 1.35, 0.5);
		save({e1, om1, moto});
		return;
		/*
		auto ees = form_group("EES", 1, GENERATOR);
		auto stim = form_group("stim", 1, GENERATOR);

		auto OM0 = form_group("OM0", 2);
		auto OM1 = form_group("OM1", 2);
		auto OM2 = form_group("OM2", 2);
		auto OM3 = form_group("OM3", 2);

		auto CV = form_group("CV", 2);
		auto E = form_group("E", 2);

		add_generator(ees, 10, 100000, 40);
		add_generator(stim, 10, 60, 200);

		conn_generator(ees, E, 1, 0.25);
		conn_generator(stim, CV, 1, 1.7);

		// testing topology coef (WITHOUT)
		connect_fixed_indegree(OM0, OM1, 2.1, 0.01);
		connect_fixed_indegree(OM1, OM2, 2.2, 0.01);
		connect_fixed_indegree(OM2, OM1, 3.2, 0.005);
		connect_fixed_indegree(OM2, OM3, 3.2, 0.0015);
		connect_fixed_indegree(OM1, OM3, 3.2, 0.00005);
		connect_fixed_indegree(OM3, OM2, 2.2, -0.1);
		connect_fixed_indegree(OM3, OM1, 3.2, -0.1);
		connect_fixed_indegree(CV, OM0, 3, 0.00035 * 1.5);
		connect_fixed_indegree(E, OM0, 3, 0.00045 * 1.5);
		save(all_groups);
	    */
		// testing topology coef (WITH)
		/*
		connect_fixed_indegree(OM0, OM1, 2.1, 2.85 / 8);
		connect_fixed_indegree(OM1, OM2, 2.2, 2.85 / 8);
		connect_fixed_indegree(OM2, OM1, 3.2, 1.95 / 8);
		connect_fixed_indegree(OM2, OM3, 3.2, 0.0015 / 8);
		connect_fixed_indegree(OM1, OM3, 3.2, 0.00005 / 8);
		connect_fixed_indegree(OM3, OM2, 2.2, -4.5 / 8);
		connect_fixed_indegree(OM3, OM1, 3.2, -4.5 / 8);
		connect_fixed_indegree(CV, OM0, 3.1, 0.00035 / 8);
		connect_fixed_indegree(E, OM0, 3.1, 0.00045 / 4);
		//	vector<Group> groups = {OM0, OM1, OM2, OM3, stim, ees};
		//	save(groups);
		*/
	}

	string name;
	vector<Group> E, CV, L0, L1, L2E, L2F, L3, IP_E, IP_F, gen_C, C_0, V0v;
	// generators
	auto ees = form_group("EES", 1, GENERATOR);
	for(int layer = 0; layer < layers + 1; ++layer) {
		name = to_string(layer + 1);
		gen_C.push_back(form_group("C" + name, 1, GENERATOR));
	}
	for(int step = 0; step < step_number; ++step) {
		name = to_string(step);
		C_0.push_back(form_group("C_0_step_" + name, 1, GENERATOR));
		V0v.push_back(form_group("V0v_step_" + name, 1, GENERATOR));
	}
	//
	auto OM1_0E = form_group("OM1_0E");
	auto OM1_0F = form_group("OM1_0F");
	// OM groups by layer
	for(int layer = 0; layer < layers; ++layer) {
		name = to_string(layer + 1);
		L0.push_back(form_group("OM" + name + "_0"));
		L1.push_back(form_group("OM" + name + "_1"));
		L2E.push_back(form_group("OM" + name + "_2E"));
		L2F.push_back(form_group("OM" + name + "_2F"));
		L3.push_back(form_group("OM" + name + "_3"));
	}
	//
	for(int layer = 0; layer < layers + 1; ++layer) {
		name = to_string(layer + 1);
		E.push_back(form_group("E" + name, 50, AFFERENTS));
		CV.push_back(form_group("CV_" + name, 50, AFFERENTS));
		// interneuronal pool
		IP_E.push_back(form_group("IP_E_" + name));
		IP_F.push_back(form_group("IP_F_" + name));
	}
	// afferents
	auto Ia_aff_E = form_group("Ia_aff_E", 120, AFFERENTS);
	auto Ia_aff_F = form_group("Ia_aff_F", 120, AFFERENTS);
	// motoneurons
	auto mns_E = form_group("mns_E", 210, MOTO);
	auto mns_F = form_group("mns_F", 180, MOTO);
	// muscle fibers
	auto muscle_E = form_group("muscle_E", 1000, MUSCLE, 3); // 150 * 210
	auto muscle_F = form_group("muscle_F", 1000, MUSCLE, 3); // 100 * 180
	// reflex arc E
	auto Ia_E = form_group("Ia_E", neurons_in_ip);
	auto iIP_E = form_group("iIP_E", neurons_in_ip);
	auto R_E = form_group("R_E");
	// reflex arc F
	auto Ia_F = form_group("Ia_F", neurons_in_ip);
	auto iIP_F = form_group("iIP_F", neurons_in_ip);
	auto R_F = form_group("R_F");

	// create EES generator
	add_generator(ees, 0, sim_time, ees_fr);
	// create CV generators (per step)
	for (int layer = 0; layer < layers + 1; ++layer) {
		for (int step_index = 0; step_index < step_number; ++step_index) {
			normal_distribution<double> freq_distr(cv_fr, cv_fr / 10);
			double start = 25 + skin_time * layer + step_index * (skin_time * (layers + 1) + flexor_dur);
			double end = start + skin_time;
			add_generator(gen_C[layer], start, end, freq_distr(rand_gen));
		}
	}
	// create C_0 and V0v generators (per step)
	for (int step_index = 0; step_index < step_number; ++step_index) {
		// freq = 200 (interval = 5ms), count = 125 / interval. Duration = count * interval = 125
		double start = 25 + skin_time * 6 + step_index * (skin_time * 6 + flexor_dur);
		double end = start + 125;
		add_generator(C_0[step_index], start, end, cv_fr);
		// V0v
		start = 40 + skin_time * 6 + step_index * (skin_time * 6 + flexor_dur);
		end = start + 75;
		add_generator(V0v[step_index], start, end, cv_fr);
	}

	// extensor
	createmotif(OM1_0E, L1[0], L2E[0], L3[0]);
	for(int layer = 1; layer < layers; ++layer)
		createmotif(L0[layer], L1[layer], L2E[layer], L3[layer]);
	// extra flexor connections
	createmotif_flex(OM1_0F, L1[0], L2E[0], L3[0]);
	for(int layer = 1; layer < layers; ++layer)
		createmotif_flex(L0[layer], L1[layer], L2F[layer], L3[layer]);

	for(int layer = 1; layer < layers; ++layer)
		connect_fixed_indegree(L2F[layer - 1], L2F[layer], 2, 1.5);
	//
	connect_fixed_indegree(E[0], OM1_0F, 3, 0.00025);
//	connect_fixed_indegree(C[step], OM1_0F, 3, 2.75);
//	for(int step = 0; step < step_number; ++step) {
//		connect_fixed_indegree(V0v[step], OM1_0F, 3, 2.75);
//	}
	// between delays via excitatory pools
	// extensor
	for(int layer = 1; layer < layers; ++layer) {
		connect_fixed_indegree(E[layer - 1], E[layer], 3, 0.75); // 4.75
	}
	// connect E (from EES)
	connect_fixed_indegree(E[0], OM1_0E, 2, 0.00044); // 0.00040 - 0.00047
	for(int layer = 1; layer < layers; ++layer) {
		connect_fixed_indegree(E[layer], L0[layer], 2, 0.00048 * 0.54); // 0.00048 * 0.4, 1.115
	}

	// E inhibitory projections (via 3rd core)
	for (int layer = 0; layer < layers - 1; ++layer) {
		if (layer >= 3) {
			for (int i = layer + 3; i < layers + 1; ++i) {
				printf("C index %d, OM%d_3 (layer > 3)\n", i, layer);
				connect_fixed_indegree(gen_C[i], L3[layer], 1, 1.95);
			}
		} else {
			for (int i = layer + 2; i < layers + 1; ++i) {
				printf("C index %d, OM%d_3 (else)\n", i, layer);
				connect_fixed_indegree(gen_C[i], L3[layer], 1, 1.95);
			}
		}
	}

	conn_generator(ees, Ia_aff_E, 1, 1.5);
	conn_generator(ees, Ia_aff_F, 1, 1.5);
	conn_generator(ees, E[0], 2, 9); // 1.5
	///conn_generator(Iagener_E, Ia_aff_E, 1, 0.0001, 5);
	///conn_generator(Iagener_F, Ia_aff_F, 1, 0.0001, 5);

	connect_fixed_indegree(Ia_aff_E, mns_E, 1.5, 5.55);
	connect_fixed_indegree(Ia_aff_F, mns_F, 1.5, 0.5);

	connect_fixed_indegree(mns_E, muscle_E, 3, 15, 45, 3); // 15.5
	connect_fixed_indegree(mns_F, muscle_F, 2, 15.5, 45);

	// IP
	for (int layer = 0; layer < layers; ++layer) {
		// Extensor
		connectinsidenucleus(IP_F[layer]);
		connectinsidenucleus(L2E[layer]);
//		connectinsidenucleus(L2F[layer]);
		connect_fixed_indegree(L2E[layer], IP_E[layer], 1.5, 0.02); // 2.5
		connect_fixed_indegree(IP_E[layer], mns_E, 2, 0.1, 50, 1); // 2.75 0.125 0.2
		if (layer > 3)
			connect_fixed_indegree(IP_E[layer], Ia_aff_E, 1, -layer * 0.0002);
		else
			connect_fixed_indegree(IP_E[layer], Ia_aff_E, 1, -0.0001);
		// Flexor
		connect_fixed_indegree(L2F[layer], IP_F[layer], 2, 2.85);
		connect_fixed_indegree(IP_F[layer], mns_F, 2, 3.75);
		connect_fixed_indegree(IP_F[layer], Ia_aff_F, 1, -0.95);
	}
	// skin inputs
	for (int layer = 0; layer < layers + 1; ++layer)
		connect_fixed_indegree(gen_C[layer], CV[layer], 2, 0.15 * k_coef * skin_time);

	// CV
	double TESTCOEF = 2.8; // 4.25
	// OM1
	connect_fixed_indegree(CV[0], OM1_0E, 2 + 1, 0.00075 * k_coef * skin_time * TESTCOEF * 0.2);
	connect_fixed_indegree(CV[1], OM1_0E, 2 + 1, 0.0005 * k_coef * skin_time * TESTCOEF * 0.35);
	// OM2
	connect_fixed_indegree(CV[0], L0[1], 3, 0.00001 * k_coef * skin_time * TESTCOEF * 0.9);
	connect_fixed_indegree(CV[1], L0[1], 3, 0.00045 * k_coef * skin_time * TESTCOEF * 0.48); // !!!
	connect_fixed_indegree(CV[2], L0[1], 2 + 0.2, 0.0004 * k_coef * skin_time * TESTCOEF * 0.5);
	// OM3
	connect_fixed_indegree(CV[0], L0[2], 3, 0.00001 * k_coef * skin_time * TESTCOEF);
	connect_fixed_indegree(CV[1], L0[2], 3, 0.00025 * k_coef * skin_time * TESTCOEF * 0.65);  // 0.7);
	connect_fixed_indegree(CV[2], L0[2], 3, 0.00035 * k_coef * skin_time * TESTCOEF * 0.65); // 0.54);
	connect_fixed_indegree(CV[3], L0[2], 3, 0.00035 * k_coef * skin_time * TESTCOEF * 0.65); // 0.51);
	connect_fixed_indegree(CV[4], L0[2], 3, 0.00035 * k_coef * skin_time * TESTCOEF * 0.8);// 0.5);
	// OM4
	connect_fixed_indegree(CV[1], L0[3], 3, 0.00002 * k_coef * skin_time * TESTCOEF * 0.9); // -
	connect_fixed_indegree(CV[2], L0[3], 3, 0.0002 * k_coef * skin_time * TESTCOEF * 0.95); // -
	connect_fixed_indegree(CV[3], L0[3], 3, 0.00035 * k_coef * skin_time * TESTCOEF * 0.6);
	connect_fixed_indegree(CV[4], L0[3], 3, 0.00035 * k_coef * skin_time * TESTCOEF * 0.6);
	connect_fixed_indegree(CV[5], L0[3], 3, 0.0001 * k_coef * skin_time * TESTCOEF * 0.5); //-
	// OM5
	connect_fixed_indegree(CV[1], L0[4], 3 + 1, 0.0001 * k_coef * skin_time * TESTCOEF * 1.6);
	connect_fixed_indegree(CV[2], L0[4], 3 + 2, 0.0001 * k_coef * skin_time * TESTCOEF * 1.6);
	connect_fixed_indegree(CV[3], L0[4], 3 - 1, 0.0001 * k_coef * skin_time * TESTCOEF * 1.3);
	connect_fixed_indegree(CV[4], L0[4], 3 - 1, 0.0001 * k_coef * skin_time * TESTCOEF * 1.3);
	connect_fixed_indegree(CV[5], L0[4], 3, 0.00035 * k_coef * skin_time * TESTCOEF * 0.5);

	// C=1 Extensor
	for (int layer = 0; layer < layers; ++layer)
		connect_fixed_indegree(IP_E[layer], iIP_E, 1, 0.001);
	//
	for (int layer = 0; layer < layers + 1; ++layer) {
		connect_fixed_indegree(CV[layer], iIP_E, 1, 1.8);
		connect_fixed_indegree(gen_C[layer], iIP_E, 1, 1.8);
	}
	connect_fixed_indegree(iIP_E, OM1_0F, 1, -1.9);

	for (int layer = 0; layer < layers; ++layer) {
		connect_fixed_indegree(iIP_E, L2F[layer], 2, -1.8);
		connect_fixed_indegree(iIP_F, L2E[layer], 2, -0.5);
	}
	//
	connect_fixed_indegree(iIP_E, Ia_aff_F, 1, -1.2);
	connect_fixed_indegree(iIP_E, mns_F, 1, -0.8);
	for (int layer = 0; layer < layers; ++layer) {
		connect_fixed_indegree(iIP_E, IP_F[layer], 1, -0.5);
		connect_fixed_indegree(IP_F[layer], iIP_F, 1, 0.0001);
		connect_fixed_indegree(iIP_F, IP_E[layer], 1, -0.8);
	}
	// C=0 Flexor
	connect_fixed_indegree(iIP_F, iIP_E, 1, -0.5);
	connect_fixed_indegree(iIP_F, Ia_aff_E, 1, -0.5);
	connect_fixed_indegree(iIP_F, mns_E, 1, -0.4);
	for(int step = 0; step < step_number; ++step) {
		connect_fixed_indegree(C_0[step], iIP_F, 1, 0.8);
	}
	// reflex arc
	connect_fixed_indegree(iIP_E, Ia_E, 1, 0.001);
	connect_fixed_indegree(Ia_aff_E, Ia_E, 1, 0.008);
	connect_fixed_indegree(mns_E, R_E, 1, 0.00015);
	connect_fixed_indegree(Ia_E, mns_F, 1, -0.08);
//	connect_fixed_indegree(R_E, mns_E, 1, -0.00015);
	connect_fixed_indegree(R_E, Ia_E, 1, -0.001);
	//
	connect_fixed_indegree(iIP_F, Ia_F, 1, 0.001);
	connect_fixed_indegree(Ia_aff_F, Ia_F, 1, 0.008);
	connect_fixed_indegree(mns_F, R_F, 1, 0.00015);
//	connect_fixed_indegree(Ia_F, mns_E, 1, -0.08);
	connect_fixed_indegree(R_F, mns_F, 1, -0.00015);
	connect_fixed_indegree(R_F, Ia_F, 1, -0.001);
	// todo C_0

	//
	connect_fixed_indegree(R_E, R_F, 1, -0.04);
	connect_fixed_indegree(R_F, R_E, 1, -0.04);
	connect_fixed_indegree(Ia_E, Ia_F, 1, -0.08);
	connect_fixed_indegree(Ia_F, Ia_E, 1, -0.08);
	connect_fixed_indegree(iIP_E, iIP_F, 1, -0.04);
	connect_fixed_indegree(iIP_F, iIP_E, 1, -0.04);

	save(all_groups);
}


void simulate(int test_index) {
	/**
	 *
	 */
	// init structs (CPU)
	States *S = (States *)malloc(sizeof(States));
	Parameters *P = (Parameters *)malloc(sizeof(Parameters));
	Neurons *N = (Neurons *)malloc(sizeof(Neurons));
	Synapses *synapses = (Synapses *)malloc(sizeof(Synapses));
	Generators *G = (Generators *)malloc(sizeof(Generators));

	// create neurons and their connectomes
	init_network();
	// note: important
	vector_nrn_start_seg.push_back(NRNS_AND_SEGS);

	// allocate generators into the GPU
	unsigned int gens_number = vec_spike_each_step.size();
	G->nrn_id = init_gpu_arr(vec_nrn_id);
	G->time_end = init_gpu_arr(vec_time_end);
	G->freq_in_steps = init_gpu_arr(vec_freq_in_steps);
	G->spike_each_step = init_gpu_arr(vec_spike_each_step);
	G->size = gens_number;

	// allocate static parameters into the GPU
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
	P->size = NRNS_NUMBER;

	// dynamic states of neuron (CPU arrays) and allocate them into the GPU
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

	int ext_size = NRNS_AND_SEGS * 2;
	auto *EXT_A = arr_segs<double>(ext_size); S->EXT_A = init_gpu_arr(EXT_A, ext_size);
	auto *EXT_B = arr_segs<double>(ext_size); S->EXT_B = init_gpu_arr(EXT_B, ext_size);
	auto *EXT_D = arr_segs<double>(ext_size); S->EXT_D = init_gpu_arr(EXT_D, ext_size);
	auto *EXT_V = arr_segs<double>(ext_size); S->EXT_V = init_gpu_arr(EXT_V, ext_size);
	auto *EXT_RHS = arr_segs<double>(ext_size); S->EXT_RHS = init_gpu_arr(EXT_RHS, ext_size);

	S->size = NRNS_AND_SEGS;
	S->ext_size = ext_size;

	// special neuron's state (CPU) and allocate them into the GPU
	auto *has_spike = arr_segs<bool>(); N->has_spike = init_gpu_arr(has_spike);
	auto *spike_on = arr_segs<bool>(); N->spike_on = init_gpu_arr(spike_on);
	auto *g_exc = arr_segs<double>(); N->g_exc = init_gpu_arr(g_exc);
	auto *g_inh_A = arr_segs<double>(); N->g_inh_A = init_gpu_arr(g_inh_A);
	auto *g_inh_B = arr_segs<double>(); N->g_inh_B = init_gpu_arr(g_inh_B);
	auto *factor = arr_segs<double>(); N->factor = init_gpu_arr(factor);
	auto *ref_time_timer = arr_segs<unsigned int>(); N->ref_time_timer = init_gpu_arr(ref_time_timer);
	auto *ref_time = arr_segs<unsigned int>();
	for (int i = 0; i < NRNS_NUMBER; ++i)
		ref_time[i] = ms_to_step(2);
	N->ref_time = init_gpu_arr(ref_time);
	N->size = NRNS_NUMBER;

	// synaptic parameters
	unsigned int synapses_number = vector_syn_delay.size();
	synapses->syn_pre_nrn = init_gpu_arr(vector_syn_pre_nrn);
	synapses->syn_post_nrn = init_gpu_arr(vector_syn_post_nrn);
	synapses->syn_weight = init_gpu_arr(vector_syn_weight);
	synapses->syn_delay = init_gpu_arr(vector_syn_delay);
	synapses->syn_delay_timer = init_gpu_arr(vector_syn_delay_timer);
	synapses->size = synapses_number;

	// allocate structs to the device
	auto *dev_S = init_gpu_arr(S, 1);
	auto *dev_P = init_gpu_arr(P, 1);
	auto *dev_N = init_gpu_arr(N, 1);
	auto *dev_G = init_gpu_arr(G, 1);
	auto *dev_synapses = init_gpu_arr(synapses, 1);

	printf("Network: %d neurons (with segs: %d), %d synapses, %d generators\n",
	       NRNS_NUMBER, NRNS_AND_SEGS, synapses_number, gens_number);

	int THREADS = 32, BLOCKS = 10;

	curandState *devStates;
	HANDLE_ERROR(cudaMalloc((void **)&devStates, NRNS_NUMBER * sizeof(curandState)));

	float time;
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// call initialisation kernel
	initialization_kernel<<<1, 1>>>(devStates, dev_S, dev_P, dev_N, -70.0);

	// the main simulation loop
	for (unsigned int sim_iter = 0; sim_iter < SIM_TIME_IN_STEPS; ++sim_iter) {
		/// KERNEL ZONE
		// deliver_net_events, synapse updating and neuron conductance changing kernel
		synapse_kernel<<<5, 256>>>(dev_N, dev_synapses);
		// updating neurons kernel
		neuron_kernel<<<BLOCKS, THREADS>>>(devStates, dev_S, dev_P, dev_N, dev_G, sim_iter);
		/// SAVE DATA ZONE
//		memcpyDtH(S->EXT_V, EXT_V, ext_size);
		memcpyDtH(S->Vm, Vm, NRNS_AND_SEGS);
		memcpyDtH(N->g_exc, g_exc, NRNS_NUMBER);
		memcpyDtH(N->g_inh_A, g_inh_A, NRNS_NUMBER);
		memcpyDtH(N->g_inh_B, g_inh_B, NRNS_NUMBER);
		memcpyDtH(N->has_spike, has_spike, NRNS_NUMBER);
		// fill records arrays
		for (GroupMetadata& metadata : saving_groups) {
			copy_data_to(metadata, Vm, g_exc, g_inh_A, g_inh_B, has_spike, sim_iter);
		}
	}
	// properly ending work with GPU
	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
	// todo optimize the code to free all GPU variables
	HANDLE_ERROR(cudaFree(S->Vm));

	// stuff info
	printf("Elapsed GPU time: %d ms\n", (int) time);
	double Tbw = 12000 * pow(10, 6) * (128 / 8) * 2 / pow(10, 9);
	printf("Theoretical Bandwidth GPU (2 Ghz, 128 bit): %.2f GB/s\n", Tbw);

	// save the data into the current folder
	save_result(test_index);
}

int main(int argc, char **argv) {
	// init the device
	int dev = 0;
	cudaDeviceProp deviceProp;
	HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s struct of array at ", argv[0]);
	printf("device %d: %s \n", dev, deviceProp.name);
	HANDLE_ERROR(cudaSetDevice(dev));
	// the main body of simulation

	fclose(fopen("file.bin", "wb"));

	simulate(0);
	// reset device
	HANDLE_ERROR(cudaDeviceReset());
}