import numpy as np
import nest


def default_network_spiking_extractor(resolution=0.1, np_seed=None):

    # Initialize NEST kernel
    from fna.tools.parameters import extract_nestvalid_dict
    from fna.tools.utils.system import set_kernel_defaults
    from fna.decoders.extractors import set_recording_device

    import nest
    import nest.raster_plot

    # Specify system and simulation parameters
    system_params = {
        'nodes': 1,
        'ppn': 16,
        'mem': 8,
        'walltime': '01-00:00:00',
        'queue': 'batch'}
    paths = {'local': {
        'data_path': './data/',
        'jdf_template': None,
        'matplotlib_rc': None,
        'remote_directory': None,
        'queueing_system': None}}

    # initialize NEST kernel
    kernel_pars = set_kernel_defaults(resolution=resolution, run_type='local', data_label='default_snn_extractor',
                                      data_paths=paths, np_seed=np_seed, **system_params)
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    # nest.set_verbosity('M_WARNING')
    nest.SetKernelStatus(extract_nestvalid_dict(kernel_pars.as_dict(), param_type='kernel'))

    # %%

    # SNN
    from fna.networks.snn import SpikingNetwork
    from fna.tools.network_architect.connectivity import NESTConnector
    from examples.example_defaults import default_neuron_parameters

    # network parameters
    gamma = 0.25  # relative number of inhibitory connections
    NE = 80  # number of excitatory neurons (10.000 in [1])
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

    snn = SpikingNetwork(snn_parameters, label='Brunel BRN',
                         spike_recorders=[set_recording_device(), set_recording_device()])
    snn_recurrent_connections = NESTConnector(source_network=snn, target_network=snn,
                                              connection_parameters=snn_synapses)

    return snn, snn_recurrent_connections