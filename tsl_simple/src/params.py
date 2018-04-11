num_sublevels = 6
num_spikes = 7
simulation_time = 250.


class Params:

    pool_to_moto_conn = {
        'weight': 1000,
        'degree': 3}

    num_pool_nrns = 50

    num_moto_nrns = 169

    params = {
        'sublayer_1': {
            'num_neurons': {
                'left': 20,
                'right': 20,
                'hidden_left': 20,
                'hidden_right': 20
            },
            'connections': {
                'left_to_right': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 100.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'right_to_left': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 86.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_left_to_right': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 100.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_right_to_left': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 200.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'right_to_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 70.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'right_to_hidden_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 200.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_right_to_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 50.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_left_to_left_down': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': -300.,
                        'sigma': 2.
                    },
                    'degree': 5
                },
                'left_to_pool': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 100.,
                        'sigma': 2.
                    },
                    'degree': 3
                }
            }
        },
        'sublayer_2': {
            'num_neurons': {
                'left': 20,
                'right': 20,
                'hidden_left': 20,
                'hidden_right': 20
            },
            'connections': {
                'left_to_right': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 100.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'right_to_left': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 88.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_left_to_right': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 100.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_right_to_left': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 300.,
                        'sigma': 2.
                    },
                    'degree': 5
                },
                'right_to_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 80.,
                        'sigma': 2.
                    },
                    'degree': 2
                },
                'right_to_hidden_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 200.,
                        'sigma': 2.
                    },
                    'degree': 5
                },
                'hidden_right_to_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 40.,
                        'sigma': 2.
                    },
                    'degree': 4
                },
                'hidden_left_to_left_down': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': -500.,
                        'sigma': 2.
                    },
                    'degree': 7
                },
                'left_to_pool': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 100.,
                        'sigma': 2.
                    },
                    'degree': 3
                }
            }
        },
        'sublayer_3': {
            'num_neurons': {
                'left': 18,
                'right': 20,
                'hidden_left': 52,
                'hidden_right': 52
            },
            'connections': {
                'left_to_right': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 90.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'right_to_left': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 92.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_left_to_right': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 100.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_right_to_left': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 300.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'right_to_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 80.,
                        'sigma': 2.
                    },
                    'degree': 4
                },
                'right_to_hidden_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 200.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_right_to_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 30.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_left_to_left_down': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': -600.,
                        'sigma': 2.
                    },
                    'degree': 6
                },
                'left_to_pool': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 100.,
                        'sigma': 2.
                    },
                    'degree': 3
                }
            }
        },
        'sublayer_4': {
            'num_neurons': {
                'left': 19,
                'right': 20,
                'hidden_left': 50,
                'hidden_right': 50
            },
            'connections': {
                'left_to_right': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 100.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'right_to_left': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 84.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_left_to_right': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 100.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_right_to_left': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 200.,
                        'sigma': 2.
                    },
                    'degree': 5
                },
                'right_to_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 70.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'right_to_hidden_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 200.,
                        'sigma': 2.
                    },
                    'degree': 5
                },
                'hidden_right_to_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 30.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_left_to_left_down': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': -300.,
                        'sigma': 2.
                    },
                    'degree': 5
                },
                'left_to_pool': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 100.,
                        'sigma': 2.
                    },
                    'degree': 4
                }
            }
        },
        'sublayer_5': {
            'num_neurons': {
                'left': 20,
                'right': 20,
                'hidden_left': 50,
                'hidden_right': 50
            },
            'connections': {
                'left_to_right': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 120.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'right_to_left': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 90.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_left_to_right': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 100.,
                        'sigma': 2.
                    },
                    'degree': 4
                },
                'hidden_right_to_left': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 200.,
                        'sigma': 2.
                    },
                    'degree': 4
                },
                'right_to_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 70.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'right_to_hidden_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 200.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_right_to_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 30.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_left_to_left_down': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': -300.,
                        'sigma': 2.
                    },
                    'degree': 6
                },
                'left_to_pool': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 100.,
                        'sigma': 2.
                    },
                    'degree': 4
                }
            }
        },
        'sublayer_6': {
            'num_neurons': {
                'left': 20,
                'right': 20,
                'hidden_left': 50,
                'hidden_right': 50
            },
            'connections': {
                'left_to_right': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 100.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'right_to_left': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 90.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_left_to_right': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 100.,
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_right_to_left': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 200.,
                        'sigma': 2.
                    },
                    'degree': 4
                },
                'right_to_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 10.,
                        'sigma': 2.
                    },
                    'degree': 1
                },
                'right_to_hidden_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 200.,
                        'sigma': 2.
                    },
                    'degree': 4
                },
                'hidden_right_to_right_up': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 10.,
                        'sigma': 2.
                    },
                    'degree': 1
                },
                'hidden_left_to_left_down': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': -300.,
                        'sigma': 2.
                    },
                    'degree': 6
                },
                'left_to_pool': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': 100.,
                        'sigma': 2.
                    },
                    'degree': 5
                }
            }
        }
    }



class Connections:

    left_to_right = {
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

    right_to_left = {
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

    hidden_left_to_right = {
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

    hidden_right_to_left = {
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