num_sublevels = 2
num_spikes = 7
simulation_time = 300.
inhibition_coeff = 1.
rate = 40

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
                        'mu': 50.,#50
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_left_to_left_down': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': -300. * inhibition_coeff,
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
                        'mu': 80.,#80
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
                        'mu': 40.,#40
                        'sigma': 2.
                    },
                    'degree': 4
                },
                'hidden_left_to_left_down': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': -500. * inhibition_coeff,
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
                        'mu': 80.,#80
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
                        'mu': 30.,#30
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_left_to_left_down': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': -600. *inhibition_coeff,
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
                        'mu': 70.,#70
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
                        'mu': 30.,#30
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_left_to_left_down': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': -300. * inhibition_coeff,
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
                        'mu': 70.,#70
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
                        'mu': 30.,#30
                        'sigma': 2.
                    },
                    'degree': 3
                },
                'hidden_left_to_left_down': {
                    'weight': {
                        'distribution': 'normal',
                        'mu': -300. * inhibition_coeff,
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
                        'mu': -300. * inhibition_coeff,
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