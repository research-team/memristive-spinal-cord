import nest
from spinal_cord.afferents.afferent_fiber import DummySensoryAfferentFiber
from spinal_cord.level1 import Level1
from spinal_cord.params import Params
from spinal_cord.toolkit.multimeter import add_multimeter
from spinal_cord.toolkit.plotter import ResultsPlotter
from spinal_cord.weights import Weights
from random import shuffle


class Pool:

    params = {
        't_ref': 2.,  # Refractory period
        'V_m': -70.0,  #
        'E_L': -70.0,  #
        'E_K': -77.0,  #
        'g_L': 30.0,  #
        'g_Na': 12000.0,  #
        'g_K': 3600.0,  #
        'C_m': 134.0,  # Capacity of membrane (pF)
        'tau_syn_ex': 4.7,  # Time of excitatory action (ms)
        'tau_syn_in': 3.1  # Time of inhibitory action (ms)
    }

    def __init__(self):
        self.extens_group_name = 'pool_extens'
        self.flex_group_name = 'pool_flex'
        self.extens_suspended_name = 'suspended_extens'
        self.flex_suspended_name = 'suspended_flex'
        self.flex_suspended_nrn_id = nest.Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.params
        )
        self.extens_suspended_nrn_id = nest.Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.params
        )
        # nest.SetStatus(self.extens_suspended_nrn_id, {'E_L': -58.})
        # nest.SetStatus(self.flex_suspended_nrn_id[:13], {'E_L': -58.})
        self.flex_group_nrn_ids = nest.Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.params
        )
        self.extens_group_nrn_ids = nest.Create(
            model='hh_cond_exp_traub',
            n=20,
            params=self.params
        )
        nest.Connect(
            pre=self.extens_suspended_nrn_id,
            post=self.extens_group_nrn_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': Weights.p_extens_sus_extens_ex
            },
            conn_spec={
                'rule': 'one_to_one'
            }
        )
        nest.Connect(
            pre=self.flex_suspended_nrn_id,
            post=self.flex_group_nrn_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': Weights.p_flex_sus_flex_ex
            },
            conn_spec={
                'rule': 'one_to_one'
            }
        )
        nest.Connect(
            pre=self.flex_group_nrn_ids,
            post=self.extens_group_nrn_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': Weights.p_flex_extens_in
            },
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 15
            }
        )
        nest.Connect(
            pre=self.extens_group_nrn_ids,
            post=self.flex_group_nrn_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': Weights.p_extens_flex_in
            },
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 15
            }
        )
        nest.Connect(
            pre=self.extens_suspended_nrn_id,
            post=self.flex_group_nrn_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': Weights.p_extens_sus_flex_in
            },
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 15
            }
        )
        nest.Connect(
            pre=self.flex_suspended_nrn_id,
            post=self.extens_group_nrn_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': Weights.p_flex_sus_extens_in
            },
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 15
            }
        )
        nest.Connect(
            pre=add_multimeter(self.flex_group_name),
            post=self.flex_group_nrn_ids
        )
        nest.Connect(
            pre=add_multimeter(self.extens_group_name),
            post=self.extens_group_nrn_ids
        )
        nest.Connect(
            pre=add_multimeter(self.flex_suspended_name),
            post=self.flex_suspended_nrn_id
        )
        nest.Connect(
            pre=add_multimeter(self.extens_suspended_name),
            post=self.extens_suspended_nrn_id
        )

    def connect_sensory(self, sensory: DummySensoryAfferentFiber):
        # syn_spec = {
        #    'model': 'static_synapse',
        #    'delay': .1,
        #    'weight': 15.
        # }
        conn_spec = {
            'rule': 'fixed_indegree',
            'indegree': 3,
            'multapses': False
        }
        nest.Connect(
            pre=sensory.neuron_ids,
            post=self.flex_suspended_nrn_id,
            syn_spec={
               'model': 'static_synapse',
               'delay': .1,
               'weight': 0.
            },
            conn_spec=conn_spec
        )
        nest.Connect(
            pre=sensory.neuron_ids,
            post=self.extens_suspended_nrn_id,
            syn_spec={
               'model': 'static_synapse',
               'delay': .1,
               'weight': 6.
            },
            conn_spec=conn_spec
        )

    def connect_level1(self, level1: Level1):
        nest.Connect(
            pre=self.extens_group_nrn_ids,
            post=level1.extens_motogroup.ia_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': Weights.p_ex_ia
            },
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 25
            }
        )
        nest.Connect(
            pre=self.extens_group_nrn_ids,
            post=level1.extens_motogroup.moto_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': Weights.p_ex_moto
            },
            conn_spec={
                'rule': 'fixed_outdegree',
                'outdegree': 25
            }
        )

        nest.Connect(
            pre=self.flex_group_nrn_ids,
            post=level1.flex_motogroup.ia_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': Weights.p_fl_ia
            },
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 25
            }
        )
        nest.Connect(
            pre=self.flex_group_nrn_ids,
            post=level1.flex_motogroup.moto_ids,
            syn_spec={
                'model': 'static_synapse',
                'delay': 1.,
                'weight': Weights.p_fl_moto
            },
            conn_spec={
                'rule': 'fixed_indegree',
                'indegree': 25
            }
        )

    def plot_results(self):
        plotter = ResultsPlotter(2, 'Average "V_m" of Pool, stimulation rate: {}Hz, inhibition strength: {}%'.format(Params.rate.value, int(Params.inh_coef.value * 100)), 'pool')

        plotter.subplot(
            first_label='extensor',
            first=self.extens_group_name,
            title='Pool'
        )
        plotter.subplot(
            first_label='suspended extensor',
            first=self.extens_suspended_name,
            second_label='suspended_flexor',
            second=self.flex_suspended_name,
            title='Suspended'
        )
        plotter.save()

    def plot_slices(self, afferent: str, time=40.):
        n_slices = 7
        plotter = ResultsPlotter(n_slices, 'Average "V_m" of Pool', 'pool_slices')
        plotter.subplot_with_slices(
            slices=n_slices,
            first_label='extensor',
            first=self.extens_group_name,
            second_label='flexor',
            second=self.flex_group_name,
            third_label='stimuli',
            third=afferent,
            title='Pool'
        )
        plotter.save()


