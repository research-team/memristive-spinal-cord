#include "core.cu"
#include "structs.h"

int TEST;
double E2F_coef;
double V0v2F_coef;
double EES_test_stregth;
double QUADRU_Ia;
const char layers = 5;      // number of OM layers (5 is default)
const char CV_number = 6;
const char extra_layers = 0 + layers;
int ees_fr = 40;      // frequency of EES

void init_network() {
	/**
	 * todo
	 */
	 /* test
	auto ees = form_group("EES", 1, GENERATOR);
	auto inter = form_group("inter", 1, INTER);
	auto afferent = form_group("afferent", 1, AFFERENTS);
	auto moto = form_group("moto", 1, MOTO);

	add_generator(ees, 0, sim_time, 40);

	conn_generator(ees, inter, 10, 0.05, 1, -1);
	conn_generator(ees, afferent, 1, 0.5, 1, -1);

	connect_fixed_indegree(inter, moto, 2, 7, 1, -1);
	connect_fixed_indegree(afferent, moto, 2, 7, 1, -1);

	save({inter, afferent, moto});

	return;
	  */
	string name;
	vector<Group> E, CV, L0, L1, L2E, L2F, L3, IP_E, IP_F, gen_C, C_0, V0v;
	// generators
	auto ees = form_group("EES", 1, GENERATOR);
	auto MOTO_NOISE = form_group("MOTO_NOISE", 1, GENERATOR);
	// 
	for(int layer = 0; layer < CV_number; ++layer) {
		name = to_string(layer + 1);
		gen_C.push_back(form_group("C" + name, 1, GENERATOR));
	}
	// EXTRA layers
	for(int layer = layers; layer < extra_layers; ++layer) {
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
	// EXTRA OM groups by layer
	for(int layer = layers; layer < extra_layers; ++layer) {
		name = to_string(layer + 1);
		L0.push_back(form_group("OM" + name + "_0"));
		L1.push_back(form_group("OM" + name + "_1"));
		L2E.push_back(form_group("OM" + name + "_2E"));
		L2F.push_back(form_group("OM" + name + "_2F"));
		L3.push_back(form_group("OM" + name + "_3"));
	}

	//
	for(int layer = 0; layer < CV_number; ++layer) {
		name = to_string(layer + 1);
		E.push_back(form_group("E" + name, 50, AFFERENTS));
		CV.push_back(form_group("CV_" + name, 50, AFFERENTS));
	}

	for(int layer = 0; layer < extra_layers; ++layer) {
		name = to_string(layer + 1);
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
	auto muscle_E = form_group("muscle_E", 210 * 50, MUSCLE, 3); // 150 * 210
	auto muscle_F = form_group("muscle_F", 180 * 50, MUSCLE, 3); // 100 * 180
	// reflex arc E
	auto Ia_E = form_group("Ia_E", neurons_in_ip);
	auto iIP_E = form_group("iIP_E", neurons_in_ip);
	auto R_E = form_group("R_E");
	// reflex arc F
	auto Ia_F = form_group("Ia_F", neurons_in_ip);
	auto iIP_F = form_group("iIP_F", neurons_in_ip);
	auto R_F = form_group("R_F");

	// create EES generator
	add_generator(ees, 25, sim_time, ees_fr);

	add_generator(MOTO_NOISE, 0, sim_time, 200);

	// create CV generators (per step)
	for (int layer = 0; layer < CV_number + TEST; ++layer) {
		for (int step_index = 0; step_index < step_number; ++step_index) {
			normal_distribution<double> freq_distr(cv_fr, cv_fr / 10);
			double start = 25 + skin_time * layer + step_index * (skin_time * CV_number + 25 * slices_flexor);

			double end = start + skin_time - 3; // remove merging CV
			add_generator(gen_C[layer], start, end, freq_distr(rand_gen));
		}
		printf("step\n");
	}

	// EXTRA create CV generators (per step)
	for (int layer = layers; layer < extra_layers + TEST; ++layer) {
		for (int step_index = 0; step_index < step_number; ++step_index) {
			normal_distribution<double> freq_distr(cv_fr, cv_fr / 10);
			double start = 25 + skin_time * (layer - 4) + step_index * (skin_time * CV_number + 25 * slices_flexor);
//			double start = 25 + skin_time * (layer) + step_index * (skin_time * CV_number + 25 * slices_flexor);
			double end = start + skin_time - 3; // remove merging CV
			add_generator(gen_C[layer], start, end, freq_distr(rand_gen));
		}
		printf("step\n");
	}

	// create C_0 and V0v generators (per step)
	for (int step_index = 0; step_index < step_number; ++step_index) {
		// freq = 200 (interval = 5ms), count = 125 / interval. Duration = count * interval = 125
		double start = 25 + skin_time * slices_extensor + step_index * (skin_time * slices_extensor + 25 * slices_flexor);
		double end = start + 25 * slices_flexor - 7;
		add_generator(C_0[step_index], start, end, cv_fr);
		// V0v
		start = 20 + skin_time * slices_extensor + step_index * (skin_time * slices_extensor + 25 * slices_flexor);
		end = start + 75; // 75
		add_generator(V0v[step_index], start, end, cv_fr);
	}

	// extensor
	createmotif(OM1_0E, L1[0], L2E[0], L3[0]);
	for(int layer = 1; layer < layers; ++layer)
		createmotif(L0[layer], L1[layer], L2E[layer], L3[layer]);
	// EXTRA layers extensor
	for(int layer = layers; layer < extra_layers; ++layer)
		createmotif(L0[layer], L1[layer], L2E[layer], L3[layer]);
	
	// FLEXOR
	createmotif_flexor(OM1_0F, L1[0], L2F[0], L3[0]);
	for(int layer = 1; layer < layers; ++layer)
		createmotif_flexor(L0[layer], L1[layer], L2F[layer], L3[layer]);
	for(int layer = 1; layer < layers; ++layer)
		connect_fixed_indegree(L2F[layer - 1], L2F[layer], 3 + 1, 0.2, 50, 2);
	// EXTRA layers FLEXOR
	for(int layer = layers; layer < extra_layers; ++layer)
		connect_fixed_indegree(L2F[layer - 1], L2F[layer], 3 + 1, 0.2, 50, 2);
	
	//
	connect_fixed_indegree(E[0], OM1_0F, 3, 0.00025 * E2F_coef, 50, 3);
	for(int step = 0; step < step_number; ++step) {
		connect_fixed_indegree(V0v[step], OM1_0F, 3, 0.75 * V0v2F_coef, 50, 5);
	}
	// between delays via excitatory pools
	// extensor

	/// !!!!!
	for(int layer = 1; layer < extra_layers; ++layer) {
		connect_fixed_indegree(E[layer - 1], E[layer], 2 + 1, 0.75); // 4.75
	}
	// connect E (from EES)
	connect_fixed_indegree(E[0], OM1_0E, 2 + 0.5, 0.005 * 0.8 * E_coef, 50, 4); // 0.00040 - 0.00047
	for(int layer = 1; layer < extra_layers; ++layer) {
		connect_fixed_indegree(E[layer], L0[layer], 2 + 0.5, 0.005 * 0.8 * E_coef, 50, 4); // 0.00048 * 0.4, 1.115
	}

	// E inhibitory projections (via 3rd core)
	/*
	for (int layer = 0; layer < layers - 1; ++layer) {
		if (layer >= 3) {
			for (int i = layer + 3; i < layers + 1 + TEST; ++i) {
				printf("C index %d, OM%d_3 (layer > 3)\n", i, layer);
				connect_fixed_indegree(gen_C[i], L3[layer], 1, 1.95);
			}
		} else {
			for (int i = layer + 2; i < layers + 1 + TEST; ++i) {
				printf("C index %d, OM%d_3 (else)\n", i, layer);
				connect_fixed_indegree(gen_C[i], L3[layer], 1, 1.95);
			}
		}
	}*/
	for (int layer = 2; layer < layers + 1; ++layer) {
		if (layer > 3) {
			for (int i = 0; i < layer - 2 + TEST; ++i) {
				printf("C index %d, OM%d_3 (layer > 3)\n", i, layer);
				connect_fixed_indegree(gen_C[layer], L3[i], 1, 1.95);
			}
		} else {
			for (int i = 0; i < layer - 1 + TEST; ++i) {
				printf("C index %d, OM%d_3 (else)\n", i, layer);
				connect_fixed_indegree(gen_C[layer], L3[i], 1, 1.95);
			}
		}
	}
	// EXTRA layers
	for (int layer = layers; layer < extra_layers; ++layer) {
		printf("C index %d, OM%d_3 (else)\n", layer, layer);
		connect_fixed_indegree(gen_C[layer - 3], L3[layer], 0.1, 0.1);
	}

	conn_generator(ees, Ia_aff_E, 1, 2.5 * EES_test_stregth);
	conn_generator(ees, Ia_aff_F, 1, 2.5 * EES_test_stregth);
	conn_generator(ees, E[0], 3, 1.0 * EES_test_stregth, 50, -1); // NORMAL
	///conn_generator(Iagener_E, Ia_aff_E, 1, 0.0001, 5);
	///conn_generator(Iagener_F, Ia_aff_F, 1, 0.0001, 5);
	// TODO motifs disable weight inh
	connect_fixed_indegree(Ia_aff_E, mns_E, 1.0, 0.045 * QUADRU_Ia); // was 1.5ms 0.045
	connect_fixed_indegree(Ia_aff_F, mns_F, 2.0, 0.006);

	connect_fixed_outdegree_MUSCLE(mns_E, muscle_E, 1.2, 0.11, 45); // 2.0
	connect_fixed_outdegree_MUSCLE(mns_F, muscle_F, 1.2, 0.38, 45); // 2.0

	connect_fixed_outdegree_MUSCLE(MOTO_NOISE, mns_E, 5, 0.05, 50, 5);
	connect_fixed_outdegree_MUSCLE(MOTO_NOISE, mns_F, 5, 0.5, 50, 5);

	if (layers >= 1)
		connect_fixed_indegree(gen_C[0], Ia_aff_E, 2, -0.4, 50, 3);
	if (layers >= 2)
		connect_fixed_indegree(gen_C[1], Ia_aff_E, 2, -0.2, 50, 3);
	// if (layers >= 3)
		// connect_fixed_indegree(CV[3], Ia_aff_E, 1, -0.1, 50, 3);
	if (layers >= 4)
		connect_fixed_indegree(gen_C[4], Ia_aff_E, 2, -0.2, 50, 3);
	if (layers >= 5)
		connect_fixed_indegree(gen_C[5], Ia_aff_E, 2, -0.4, 50, 3);

	// connect_fixed_outdegree_MUSCLE(gen_C[0], mns_E, 4, -2.5, 100, 5); // 2.0
	// connect_fixed_outdegree_MUSCLE(gen_C[1], mns_E, 4, -1, 100, 5); // 2.0
	// connect_fixed_outdegree_MUSCLE(gen_C[4], mns_E, 4, -1, 100, 5); // 2.0
	// connect_fixed_outdegree_MUSCLE(gen_C[5], mns_E, 4, -2.5, 100, 5); // 2.0

	// connect_fixed_outdegree_MUSCLE(MOTO_NOISE, muscle_E, 5, 0.1, 50, 5);
	// connect_fixed_outdegree_MUSCLE(MOTO_NOISE, muscle_F, 5, 0.3, 50, 5);

	// IP
	for (int layer = 0; layer < layers; ++layer) {
		// Extensor
		connect_fixed_indegree(L2E[layer], IP_E[layer], 2, 0.005, 500, 5); // 2.5
		connect_fixed_indegree(IP_E[layer], mns_E, 2, 0.0045, 500, 5); // 0.005
		if (layer > 3)
			connect_fixed_indegree(IP_E[layer], Ia_aff_E, 1, -0.0002 *layer);
		else
			connect_fixed_indegree(IP_E[layer], Ia_aff_E, 1, -0.0001);
		// Flexor
		connect_fixed_indegree(L2F[layer], IP_F[layer], 2, 0.001, 500, 5); // 2.5
		connect_fixed_indegree(IP_F[layer], mns_F, 3, 0.004, 100, 5); // 2.75 0.125 0.2
	}

	for (int layer = layers; layer < extra_layers; ++layer) {
		// Extensor
		connect_fixed_indegree(L2E[layer], IP_E[layer], 2, 0.005, 500, 5); // 2.5
		connect_fixed_indegree(IP_E[layer], mns_E, 2, 0.0045, 500, 5); // 0.005
		// Flexor
		connect_fixed_indegree(L2F[layer], IP_F[layer], 2, 0.001, 500, 5); // 2.5
		connect_fixed_indegree(IP_F[layer], mns_F, 3, 0.004, 100, 5); // 2.75 0.125 0.2
	}

	// skin inputs
	for (int layer = 0; layer < CV_number + TEST; ++layer)
		connect_fixed_indegree(gen_C[layer], CV[layer], 2, 0.15 * cv_coef);
	// CV
	double TESTCOEF = 30.0;
	double T_coef = 1.0;
	// OM1
	connect_fixed_indegree(CV[0], OM1_0E, 2 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
	connect_fixed_indegree(CV[1], OM1_0E, 2 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
	// OM2
	if (layers >= 1) {
		connect_fixed_indegree(CV[0], L0[1], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[1], L0[1], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		if (layers >= 2) connect_fixed_indegree(CV[2], L0[1], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
	}
	// OM3
	if (layers >= 2) {
		connect_fixed_indegree(CV[0], L0[2], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[1], L0[2], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[2], L0[2], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		if (layers >= 3) connect_fixed_indegree(CV[3], L0[2], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		if (layers >= 4) connect_fixed_indegree(CV[4], L0[2], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
	}
	// OM4
	if (layers >= 3) {
		connect_fixed_indegree(CV[1], L0[3], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[2], L0[3], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[3], L0[3], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		if (layers >= 4) connect_fixed_indegree(CV[4], L0[3], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		if (layers >= 5) connect_fixed_indegree(CV[5], L0[3], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
	}
	// OM5
	if (layers >= 4) {
		connect_fixed_indegree(CV[1], L0[4], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[2], L0[4], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[3], L0[4], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[4], L0[4], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		if (layers >= 5) connect_fixed_indegree(CV[5], L0[4], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
	}

	// OM6
	if (extra_layers > 5) {
		connect_fixed_indegree(CV[1], L0[5], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[2], L0[5], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[3], L0[5], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[4], L0[5], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[5], L0[5], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
//		if (extra_layers >= 6) connect_fixed_indegree(CV[6], L0[5], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
	}

	// OM7
	if (extra_layers > 6) {
		connect_fixed_indegree(CV[1], L0[6], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[2], L0[6], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[3], L0[6], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[4], L0[6], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[5], L0[6], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
		connect_fixed_indegree(CV[6], L0[6], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
//		if (extra_layers >= 7) connect_fixed_indegree(CV[7], L0[6], 3 + T_coef, 0.00045 * cv_coef * TESTCOEF * 0.3, 50, 3);
	}

	// C=1 Extensor
	for (int layer = 0; layer < layers; ++layer)
		connect_fixed_indegree(IP_E[layer], iIP_E, 1, 0.001);

	for (int layer = 0; layer < layers + TEST; ++layer) {
		connect_fixed_indegree(CV[layer], iIP_E, 4, 1); // for what CV and Gen_C
		connect_fixed_indegree(gen_C[layer], iIP_E, 1, 1);
	}
	connect_fixed_indegree(iIP_E, OM1_0F, 0.1, -0.001);

	for (int layer = 0; layer < layers; ++layer) {
		connect_fixed_indegree(iIP_E, L2F[layer], 2, -0.4); //0.8 0.6
		connect_fixed_indegree(iIP_F, L2E[layer], 2, -0.5);
	}
	//
	connect_fixed_indegree(iIP_E, Ia_aff_F, 1, -1.2);
	connect_fixed_indegree(iIP_E, mns_F, 4, -0.0315); // 0.08 delay 0.1

	for (int layer = 0; layer < layers; ++layer) {
		connect_fixed_indegree(iIP_E, IP_F[layer], 4.5, -0.005); // 0.1
		connect_fixed_indegree(IP_F[layer], iIP_F, 1, 0.0001);
		connect_fixed_indegree(iIP_F, IP_E[layer], 1, -0.08); // -0.1
	}
	// C=0 Flexor
	connect_fixed_indegree(iIP_F, iIP_E, 1, -0.5);
	connect_fixed_indegree(iIP_F, Ia_aff_E, 1, -3.5);
	connect_fixed_indegree(iIP_F, mns_E, 1, -0.35); //0.5
	for(int step = 0; step < step_number; ++step) {
		connect_fixed_indegree(C_0[step], iIP_F, 1, 0.8);
	}
	// reflex arc
	connect_fixed_indegree(iIP_E, Ia_E, 1, 0.001);
	connect_fixed_indegree(Ia_aff_E, Ia_E, 1, 0.008);
	connect_fixed_indegree(mns_E, R_E, 1, 0.00015);
	connect_fixed_indegree(Ia_E, mns_F, 0.1, -0.002);
//	connect_fixed_indegree(R_E, mns_E, 1, -0.00015);
	connect_fixed_indegree(R_E, Ia_E, 1, -0.001);
	//
	connect_fixed_indegree(iIP_F, Ia_F, 1, 0.001);
	connect_fixed_indegree(Ia_aff_F, Ia_F, 1, 0.008);
	connect_fixed_indegree(mns_F, R_F, 1, 0.00015);
//	connect_fixed_indegree(Ia_F, mns_E, 1, -0.08);

	connect_fixed_indegree(R_F, mns_F, 0.1, -0.00015);
	connect_fixed_indegree(R_F, Ia_F, 1, -0.001);
	// todo C_0

	//
	connect_fixed_indegree(R_E, R_F, 1, -0.04);
	connect_fixed_indegree(R_F, R_E, 1, -0.04);
	connect_fixed_indegree(Ia_E, Ia_F, 1, -0.08);
	connect_fixed_indegree(Ia_F, Ia_E, 1, -0.08);
	connect_fixed_indegree(iIP_E, iIP_F, 3, -0.04); // delay 1
	connect_fixed_indegree(iIP_F, iIP_E, 1, -0.04);

	save({muscle_E, muscle_F});
//	 save(all_groups);
}


void simulate(int test_index) {
	/**
	 *
	 */
	random_device r;
	default_random_engine rand_gen(r());
	uniform_real_distribution<double> delay_distr(2, 4);

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
//	double *Vm; HANDLE_ERROR(cudaMallocHost((void**)&Vm, NRNS_AND_SEGS));
	auto *Vm = arr_init<double>(); S->Vm = init_gpu_arr(Vm);
	auto *n = arr_init<double>(); S->n = init_gpu_arr(n);
	auto *m = arr_init<double>(); S->m = init_gpu_arr(m);
	auto *h = arr_init<double>(); S->h = init_gpu_arr(h);
	auto *l = arr_init<double>(); S->l = init_gpu_arr(l);
	auto *s = arr_init<double>(); S->s = init_gpu_arr(s);
	auto *p = arr_init<double>(); S->p = init_gpu_arr(p);
	auto *hc = arr_init<double>(); S->hc = init_gpu_arr(hc);
	auto *mc = arr_init<double>(); S->mc = init_gpu_arr(mc);
	auto *cai = arr_init<double>(); S->cai = init_gpu_arr(cai);
	auto *I_Ca = arr_init<double>(); S->I_Ca = init_gpu_arr(I_Ca);
	auto *NODE_A = arr_init<double>(); S->NODE_A = init_gpu_arr(NODE_A);
	auto *NODE_B = arr_init<double>(); S->NODE_B = init_gpu_arr(NODE_B);
	auto *NODE_D = arr_init<double>(); S->NODE_D = init_gpu_arr(NODE_D);
	auto *const_NODE_D = arr_init<double>(); S->const_NODE_D = init_gpu_arr(const_NODE_D);
	auto *NODE_RHS = arr_init<double>(); S->NODE_RHS = init_gpu_arr(NODE_RHS);
	auto *NODE_RINV = arr_init<double>(); S->NODE_RINV = init_gpu_arr(NODE_RINV);
	auto *NODE_AREA = arr_init<double>(); S->NODE_AREA = init_gpu_arr(NODE_AREA);

//	int ext_size = NRNS_AND_SEGS * 2;
//	auto *EXT_A = arr_init<double>(ext_size); S->EXT_A = init_gpu_arr(EXT_A, ext_size);
//	auto *EXT_B = arr_init<double>(ext_size); S->EXT_B = init_gpu_arr(EXT_B, ext_size);
//	auto *EXT_D = arr_init<double>(ext_size); S->EXT_D = init_gpu_arr(EXT_D, ext_size);
//	auto *EXT_V = arr_init<double>(ext_size); S->EXT_V = init_gpu_arr(EXT_V, ext_size);
//	auto *EXT_RHS = arr_init<double>(ext_size); S->EXT_RHS = init_gpu_arr(EXT_RHS, ext_size);
	S->size = NRNS_AND_SEGS;
//	S->ext_size = ext_size;

	// special neuron's state (CPU) and allocate them into the GPU
	auto *tmp = arr_init<double>(NRNS_NUMBER);
	for (int i = 0; i < NRNS_NUMBER; ++i)
		tmp[i] = 0.0;

	auto *has_spike = arr_init<bool>(NRNS_NUMBER); N->has_spike = init_gpu_arr(has_spike, NRNS_NUMBER);
	auto *g_exc = arr_init<double>(NRNS_NUMBER); N->g_exc = init_gpu_arr(g_exc, NRNS_NUMBER);
	auto *g_inh_A = arr_init<double>(NRNS_NUMBER); N->g_inh_A = init_gpu_arr(g_inh_A, NRNS_NUMBER);
	auto *g_inh_B = arr_init<double>(NRNS_NUMBER); N->g_inh_B = init_gpu_arr(g_inh_B, NRNS_NUMBER);
	auto *spike_on = arr_init<bool>(NRNS_NUMBER); N->spike_on = init_gpu_arr(spike_on, NRNS_NUMBER);
	auto *factor = arr_init<double>(NRNS_NUMBER); N->factor = init_gpu_arr(factor, NRNS_NUMBER);
	auto *ref_time_timer = arr_init<unsigned int>(NRNS_NUMBER); N->ref_time_timer = init_gpu_arr(ref_time_timer, NRNS_NUMBER);
	auto *ref_time = arr_init<unsigned int>(NRNS_NUMBER);
	for (int i = 0; i < NRNS_NUMBER; ++i)
		ref_time[i] = ms_to_step(delay_distr(rand_gen));
	N->ref_time = init_gpu_arr(ref_time, NRNS_NUMBER);
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
		if (sim_iter % 1000 == 0) {
			printf("%.2f%% is done\n", 100.0 * sim_iter / SIM_TIME_IN_STEPS);
		}
		/// KERNEL ZONE
		// deliver_net_events, synapse updating and neuron conductance changing kernel
		synapse_kernel<<<5, 256>>>(dev_N, dev_synapses);
		// updating neurons kernel
		neuron_kernel<<<BLOCKS, THREADS>>>(devStates, dev_S, dev_P, dev_N, dev_G, sim_iter);
		/// SAVE DATA ZONE
		/*
		// int streamSize = NRNS_AND_SEGS / 4;
		// int streamSizeTail = NRNS_AND_SEGS % 4;
		// for (int streamIndex = 0; streamIndex < nStreams; ++streamIndex) {
		// 	int offset = streamIndex * streamSize;
		// 	if (streamIndex == nStreams - 1)
		// 		offset += streamSizeTail;
		// 	cudaMemcpyAsync(&Vm[offset], &S->Vm[offset], streamBytes, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToHost, stream[streamIndex]);
		// 	cudaMemcpyAsync(&g_exc[offset], &N->g_exc[offset], streamBytes, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToHost, stream[streamIndex]);
		// 	cudaMemcpyAsync(&g_inh_A[offset], &N->g_inh_A[offset], streamBytes, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToHost, stream[streamIndex]);
		// 	cudaMemcpyAsync(&g_inh_B[offset], &N->g_inh_B[offset], streamBytes, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToHost, stream[streamIndex]);
		// 	cudaMemcpyAsync(&has_spike[offset], &N->has_spike[offset], streamBytes, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToHost, stream[streamIndex]);
		// }*/
		memcpyDtH(S->Vm, Vm, NRNS_AND_SEGS);
		memcpyDtH(N->g_exc, g_exc, NRNS_NUMBER);
		memcpyDtH(N->g_inh_A, g_inh_A, NRNS_NUMBER);
		memcpyDtH(N->g_inh_B, g_inh_B, NRNS_NUMBER);
		memcpyDtH(N->has_spike, has_spike, NRNS_NUMBER);
		// fill records arrays
		for (GroupMetadata& metadata : saving_groups) {
			copy_data_to(metadata, Vm, tmp, g_exc, g_inh_A, g_inh_B, has_spike, sim_iter);
		}
	}
	// properly ending work with GPU
	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
	// todo optimize the code to free all GPU variables
	// HANDLE_ERROR(cudaFree(S->Vm));
	// HANDLE_ERROR( cudaFreeHost( S->Vm ) );

	// stuff info
	printf("Elapsed GPU time: %d ms\n", (int) time);
	double Tbw = 12000 * pow(10, 6) * (128 / 8) * 2 / pow(10, 9);
	printf("Theoretical Bandwidth GPU (2 Ghz, 128 bit): %.2f GB/s\n", Tbw);

	// save the data into the current folder
	save_result(test_index);
}

int main(int argc, char **argv) {
	enum modes {air, toe, plt, quadru, normal, qpz, str, s6, s13, s21};
	//
	int iter = atoi(argv[1]);
	int arg_exp = atoi(argv[2]);
	int arg_hz = atoi(argv[3]);

	modes bws, pharma, speed;

	if (arg_exp == 0) {
		printf("PLT 13.5\n");
		bws = plt;     
		pharma = normal;  
		speed = s13;
	} else if (arg_exp == 1) {
		printf("TOE 13.5\n");
		bws = toe;
		pharma = normal;
		speed = s13;
	} else if (arg_exp == 2) {
		printf("AIR 13.5\n");
		bws = air;
		pharma = normal;
		speed = s13;
	} else if (arg_exp == 3) {
		printf("QUADRU 13.5\n");
		bws = quadru;     
		pharma = normal;  
		speed = s13;    
	} else if (arg_exp == 4) {
		printf("QPZ 13.5\n");
		bws = plt; 
		pharma = qpz;
		speed = s13; 
	} else if (arg_exp == 5) {
		printf("STR 13.5\n");
		bws = plt; 
		pharma = str;
		speed = s13;  
	} else if (arg_exp == 6) {
		printf("PLT 21\n");
		bws = plt;     
		pharma = normal;
		speed = s21;
	} else if (arg_exp == 7) {
		printf("PLT 6\n");
		bws = plt;     
		pharma = normal;
		speed = s6;    
	}

	step_number = 11;
	
	TEST = 0;
	E2F_coef = 1;
	V0v2F_coef = 1;
	QUADRU_Ia = 1;
	EES_test_stregth = 1.0;
	ees_fr = arg_hz;

	// speed modes
	switch(speed) {
		case s6:
			skin_time = 125;
			break;
		case s13:
			skin_time = 50;
			break;
		case s21:
			skin_time = 25;
			break;
		default:
			exit(-1);
	}
	// BWS modes
	switch(bws) {
		case air:
			TEST = -1;
			skin_time = 25;
			cv_coef = 0.043; // 037
			E_coef = 0.05;
			slices_extensor = 5;
			slices_flexor = 4;
			E2F_coef = 0;
			V0v2F_coef = 0;
			break;
		case toe:
			TEST = -2;
			cv_coef = 0.05;
			E_coef = 0.05;
			slices_extensor = 4;
			slices_flexor = 4;
			E2F_coef = 8;
			V0v2F_coef = 0;
			break;
		case plt: //!
			QUADRU_Ia = 1.0;
			cv_coef = 0.07;			// cv_coef = 0.0615;	gut 0.08
			E_coef = 0.05;			// 	E_coef = 0.052;
			slices_extensor = 6;	// 	slices_extensor = 6;		
			slices_flexor = 5;		// 	slices_flexor = 5;	
			E2F_coef = 8;			// 	E2F_coef = 8;
			V0v2F_coef = 0.001;		// 	V0v2F_coef = 0.001;	
			break;
		case quadru:
			QUADRU_Ia = 1.0;
			cv_coef = 0.036; // 0.042
			E_coef = 0.039; // 0.045
			slices_extensor = 6;
			slices_flexor = 7;
			E2F_coef = 8;
			V0v2F_coef = 0.001;
			break;
		default:
			exit(-1);
	}
	// pharma modes
	switch(pharma) {
		case normal:
			break;
		case qpz:
			QUADRU_Ia = 1.5;
			cv_coef = 0.15; // 0.05 1ю15
			E_coef = 0.17; // 0.07  3ю0
			V0v2F_coef = 0.001;
			break;
		case str:
			QUADRU_Ia = 0.8;
			str_flag = true;
			V0v2F_coef = 0.001;
			break;
		default:
			exit(-1);
	}

	// remove
	// E_coef = 0.0025;
	// cv_coef = 0.069;
	// testing a weak EES and increased a CV strength
	// cv_coef = 0.09;
	// EES_test_stregth = 0.001;

	// one_step_time = slices_extensor * skin_time + 25 * slices_flexor;
	int fr = 1000 / ees_fr;
	one_step_time = (slices_extensor * skin_time + 25 * slices_flexor) / fr * fr;
	printf("STEP LENGTH %d\n", one_step_time);
	sim_time = 25 + one_step_time * step_number;
	SIM_TIME_IN_STEPS = (unsigned int)(sim_time / dt);  // [steps] converted time into steps

	// init the device
	int dev = 0;
	cudaDeviceProp deviceProp;
	HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
	printf("%s struct of array at device %d: %s \n", argv[0], dev, deviceProp.name);
	HANDLE_ERROR(cudaSetDevice(dev));
	
	printf("%d\n", arg_exp * 10 + iter);
	// the main body of simulation
	//simulate(arg_exp * 10 + iter);
	simulate(arg_hz);
	
	// reset device
	HANDLE_ERROR(cudaDeviceReset());
}