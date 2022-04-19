import numpy as np

from examples.example_defaults import default_neuron_parameters
from networks.snn import SpikingNetwork

from tools.utils.system import set_kernel_defaults, reset_nest_kernel

# Specify system and simulation parameters
resolution = 0.1
data_label = 'test_network'
system = 'local'
system_params = {
    'nodes': 1,
    'ppn': 16,
    'mem': 8,
    'walltime': '01-00:00:00',
    'queue': 'batch'}
paths = {'local': {
        'data_path': '../data/',
        'jdf_template': 	None,
        'matplotlib_rc': 	None,
        'remote_directory': None,
        'queueing_system':  None}}

# initialize NEST kernel
kernel_pars = set_kernel_defaults(run_type=system, data_label=data_label, data_paths=paths, **system_params)
reset_nest_kernel(kernel_pars)

# #####################################################################################
# ## A) Simple Brunel BRN
# #####################################################################################
# Specify network parameters
gamma = 0.25               # relative number of inhibitory connections
NE = 500                  # number of excitatory neurons (10.000 in [1])
NI = int(gamma * NE)       # number of inhibitory neurons
CE = 100                  # indegree from excitatory neurons
CI = int(gamma * CE)       # indegree from inhibitory neurons

# synapse parameters
w = 1.#0.1                    # excitatory synaptic weight (mV)
g = 5.                     # relative inhibitory to excitatory synaptic weight
d = 1.5                    # synaptic transmission delay (ms)

neuron_params = default_neuron_parameters(default_set=1) # for convenience (full parameters can be passed)

snn_parameters = {
    'populations': ['E', 'I'],
    'population_size': [NE, NI],
    'neurons': [neuron_params, neuron_params],
    'randomize': [
        {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})},
        {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})}]}

# Generate SNN instance
snn = SpikingNetwork(snn_parameters, label='Brunel BRN', topologies=None, spike_recorders=None, analog_recorders=None)

# ###################################################################################
# Connectivity
# E synapses
# syn_exc = {'model': 'static_synapse', 'delay': d, 'weight': w}
# conn_exc = {'rule': 'fixed_indegree', 'indegree': CE}
# # I synapses
# syn_inh = {'model': 'static_synapse', 'delay': d, 'weight': - g * w}
# conn_inh = {'rule': 'fixed_indegree', 'indegree': CI}
#
# snn_synapses = {
#     'connect_populations': [('E', 'E'), ('E', 'I'), ('I', 'E'), ('I', 'I')],
#     'weight_matrix': [None, None, None, None],
#     'conn_specs': [conn_exc, conn_inh, conn_exc, conn_inh],
#     'syn_specs': [syn_exc, syn_inh, syn_exc, syn_inh]
# }
# snn_recurrent_connections = NESTConnector(source_network=snn, target_network=snn, connection_parameters=snn_synapses)
# w_rec = snn_recurrent_connections.compile_weights()

# ####################################################################################
# Profiling
# ####################################################################################
# 1) Single neuron fI curves
results = snn.neuron_transferfunction(gid=100, input_type='current', input_range=(0., 800.), total_time=10000.,
                                  step=100., restore=False, display=True, save=False)

# 2) Single neuron rate transfer function
# snn.neuron_transferfunction(gid=100, input_type='rate', input_range=(0., 100000.), input_weights=(w, ),
#                             total_time=10000., step=100., restore=False, display=True, save=False)
