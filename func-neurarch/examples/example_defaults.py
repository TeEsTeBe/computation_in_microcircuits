import numpy as np

from fna.networks.snn import SpikingNetwork
from fna.tasks.symbolic import SymbolicSequencer
from fna.tools.network_architect.connectivity import NESTConnector

from fna.tools.parameters import ParameterSet, extract_nestvalid_dict
from fna.decoders.extractors import set_recording_device
from fna.tools.utils.operations import copy_dict
from fna.tools.utils.system import set_kernel_defaults

import nest


def default_neuron_parameters(default_set=0):
    """
    Default single neuron parameter sets,

    :param default_set: (int) - if applicable
    :return: ParameterSet
    """
    if default_set == 0:
        # print("\nLoading default neuron parameter - iaf_psc_delta, fixed voltage threshold, fixed absolute refractory "
        #       "time")
        neuron_params = {
            'model': 'iaf_psc_delta',
            'C_m': 1.0,  # membrane capacity (pF)
            'E_L': 0.,  # resting membrane potential (mV)
            'I_e': 0.,  # external input current (pA)
            'V_m': 0.,  # membrane potential (mV)
            'V_reset': 10.,  # reset membrane potential after a spike (mV)
            'V_th': 20.,  # spike threshold (mV)
            't_ref': 2.0,  # refractory period (ms)
            'tau_m': 20.,  # membrane time constant (ms)
        }
    elif default_set == 1:
        print("\nLoading default neuron parameter - iaf_psc_exp, fixed voltage threshold, fixed absolute refractory "
              "time")
        neuron_params = {
            'model': 'iaf_psc_exp',
            'C_m': 250.0,  # membrane capacity (pF)
            'E_L': 0.0,  # resting membrane potential (mV)
            'V_reset': 0.0,  # external input current (pA)
            'V_th': 15.,  # membrane potential (mV)
            't_ref': 2.,  # reset membrane potential after a spike (mV)
            'tau_syn_ex': 2.,  # spike threshold (mV)
            'tau_syn_in': 2.,
            'tau_m': 20.
        }
    elif default_set == 2:
        print("\nLoading default neuron parameter - iaf_cond_exp, fixed voltage threshold, fixed absolute refractory "
              "time")
        neuron_params = {
            'model': 'iaf_cond_exp',
            'C_m': 250.,
            'E_L': -70.0,
            'I_e': 0.,
            'V_m': -70.0,
            'V_th': -50.0,
            'V_reset': -60.0,
            'g_L': 16.7,
            't_ref': 5.,
            'tau_minus': 20.,
            'E_ex': 0.,
            'tau_syn_ex': 5.,
            'E_in': -80.,
            'tau_syn_in': 10.
        }
    elif default_set == 3:
        print("\nLoading Default Neuron  - aeif_cond_exp, fixed voltage threshold, fixed absolute refractory "
              "time, Fast, conductance-based exponential synapses")
        neuron_params = {
            'model': 'aeif_cond_exp',
            'C_m': 250.0,
            'Delta_T': 2.0,
            'E_L': -70.,
            'E_ex': 0.0,
            'E_in': -75.0,
            'I_e': 0.,
            'V_m': -70.,
            'V_th': -50.,
            'V_reset': -60.0,
            'V_peak': 0.0,
            'a': 4.0,
            'b': 80.5,
            'g_L': 16.7,
            'g_ex': 1.0,
            'g_in': 1.0,
            't_ref': 2.0,
            'tau_minus': 20.,
            'tau_minus_triplet': 200.,
            'tau_w': 144.0,
            'tau_syn_ex': 2.,
            'tau_syn_in': 6.0
        }
    else:
        raise NotImplementedError("Default set {0} not implemented".format(str(default_set)))
    return ParameterSet(neuron_params)


def reset_kernel(resolution=0.1, np_seed=None, data_label='default_snn_extractor'):
    # Specify system and simulation parameters
    system_params = {
        'nodes': 1,
        'ppn': 16,
        'mem': 8,
        'walltime': '01-00:00:00',
        'queue': 'batch'}
    paths = {'local': {
        'data_path': '../data/',
        'jdf_template': None,
        'matplotlib_rc': None,
        'remote_directory': None,
        'queueing_system': None}}

    # initialize NEST kernel
    kernel_pars = set_kernel_defaults(resolution=resolution, run_type='local', data_label=data_label,
                                      data_paths=paths, np_seed=np_seed, **system_params)
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    # nest.set_verbosity('M_WARNING')
    nest.SetKernelStatus(extract_nestvalid_dict(kernel_pars.as_dict(), param_type='kernel'))


def default_network(N=100, record_spikes=False):
    # SNN
    # network parameters
    gamma = 0.25  # relative number of inhibitory connections
    NE = int(0.8*N)  # number of excitatory neurons (10.000 in [1])
    NI = int(gamma * NE)  # number of inhibitory neurons
    CE = 10  # indegree from excitatory neurons
    CI = int(gamma * CE)  # indegree from inhibitory neurons
    # synapse parameters
    w = 4.  # excitatory synaptic weight (mV)
    g = 5.  # relative inhibitory to excitatory synaptic weight
    d = 1.5  # synaptic transmission delay (ms)
    neuron_params = default_neuron_parameters(default_set=0)  # for convenience (full parameters can be passed)
    snn_parameters = {
        'populations': ['E', 'I'],
        'population_size': [NE, NI],
        'neurons': [neuron_params, neuron_params],
        'randomize': [
            {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})},
            {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})}]}
    # E synapses
    syn_exc = {'model': 'static_synapse', 'delay': d, 'weight': w}
    conn_exc = {'rule': 'fixed_indegree', 'indegree': CE}
    # I synapses
    syn_inh = {'model': 'static_synapse', 'delay': d, 'weight': - g * w}
    conn_inh = {'rule': 'fixed_indegree', 'indegree': CI}

    snn_synapses = {
        'connect_populations': [('E', 'E'), ('E', 'I'), ('I', 'E'), ('I', 'I')],
        'weight_matrix': [None, None, None, None],
        'conn_specs': [conn_exc, conn_inh, conn_exc, conn_inh],
        'syn_specs': [syn_exc, syn_inh, syn_exc, syn_inh]
    }

    if record_spikes:
        snn = SpikingNetwork(snn_parameters, label='Brunel BRN',
                             spike_recorders=[set_recording_device(), set_recording_device()])
    else:
        snn = SpikingNetwork(snn_parameters, label='Brunel BRN')
    snn_recurrent_connections = NESTConnector(source_network=snn, target_network=snn,
                                              connection_parameters=snn_synapses)

    return snn, snn_recurrent_connections


def default_tasks(n_tokens=10, T=100):
    sequencer = SymbolicSequencer(label='random', set_size=n_tokens)
    sequence = sequencer.generate_random_sequence(T=T)
    return sequencer, sequence
