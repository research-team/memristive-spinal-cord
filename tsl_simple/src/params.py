num_sublevels = 6
num_spikes = 2
simulation_time = 200.


class Connections:

    sub_left_to_right = {
        'sublayer_5': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_4': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_3': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_2': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_1': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_0': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1}}

    sub_right_to_left = {
        'sublayer_5': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_4': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_3': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_2': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_1': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_0': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1}}

    hidden_sub_left_to_right = {
        'sublayer_5': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_4': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_3': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_2': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_1': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_0': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1}}

    hidden_sub_right_to_left = {
        'sublayer_5': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_4': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_3': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_2': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_1': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_0': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1}}

    right_to_right_up = {
        'sublayers_4_5' : {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayers_3_4' : {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayers_2_3' : {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayers_1_2' : {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayers_0_1' : {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},}

    right_to_hidden_right_up = {
        'sublayer_5': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_4': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_3': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_2': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_1': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_0': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1}}

    hidden_right_to_right_up = {
        'sublayer_4_5': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_3_4': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_2_3': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_1_2': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_0_1': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1}}

    hidden_left_to_left_down = {
        'sublayer_5': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_4': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_3': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_2': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_1': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_0': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1}}

    left_to_pool = {
        'sublayer_5': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_4': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_3': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_2': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_1': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1},
        'sublayer_0': {
            'weight': {'distribution': 'normal', 'mu': 10., 'sigma': 2.},
            'degree': 1}}

    pool_to_moto = {
        'weight': 1000,
        'degree': 3}


class Neurons:

    sublayers = {
        'sublayer_5': {
            'left': 20,
            'right': 20},
        'sublayer_4': {
            'left': 20,
            'right': 20},
        'sublayer_3': {
            'left': 20,
            'right': 20},
        'sublayer_2': {
            'left': 20,
            'right': 20},
        'sublayer_1': {
            'left': 20,
            'right': 20},
        'sublayer_0': {
            'left': 20,
            'right': 20}}

    hidden_sublayers = {
        'sublayer_5': {
            'left': 20,
            'right': 20},
        'sublayer_4': {
            'left': 20,
            'right': 20},
        'sublayer_3': {
            'left': 20,
            'right': 20},
        'sublayer_2': {
            'left': 20,
            'right': 20},
        'sublayer_1': {
            'left': 20,
            'right': 20},
        'sublayer_0': {
            'left': 20,
            'right': 20}}

    pool = 50

    moto = 169