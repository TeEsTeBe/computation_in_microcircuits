import nest
import numpy as np

from examples.example_defaults import default_neuron_parameters
from networks.snn import SpikingNetwork
from tools.parameters import extract_nestvalid_dict
from tools.utils.system import set_kernel_defaults
from tools.network_architect.topology import set_positions
from tools.utils.operations import copy_dict
from tools.network_architect.connectivity import NESTConnector
from tools.visualization.plotting import plot_spatial_connectivity, plot_network_topology

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
nest.ResetKernel()
nest.set_verbosity('M_WARNING')
nest.SetKernelStatus(extract_nestvalid_dict(kernel_pars.as_dict(), param_type='kernel'))

# #####################################################################################
# Network Creation
# #####################################################################################
# Specify network parameters
gamma = 0.25               # relative number of inhibitory connections
NE = 8000                  # number of excitatory neurons (10.000 in [1])
NI = int(gamma * NE)       # number of inhibitory neurons
CE = 1000                  # indegree from excitatory neurons
CI = int(gamma * CE)       # indegree from inhibitory neurons

# synapse parameters
w = 0.1                    # excitatory synaptic weight (mV)
g = 5.                     # relative inhibitory to excitatory synaptic weight
d = 1.5                    # synaptic transmission delay (ms)

neuron_params = default_neuron_parameters(default_set=0) # for convenience (full parameters can be passed)

N = NE + NI

# for simplicity all other parameters are the same, only topology is added
layer_properties = {
    'center': [np.round(np.sqrt(N)) / 2., np.round(np.sqrt(N)) / 2.],
    'extent': [np.ceil(np.sqrt(N)), np.ceil(np.sqrt(N))],
    'edge_wrap': True, 'elements': neuron_params['model']}
pos = set_positions(N=N, dim=2, topology='lattice')
E_layer_properties = copy_dict(layer_properties, {'positions': pos[:int(NE)]})
I_layer_properties = copy_dict(layer_properties, {'positions': pos[int(NE):]})

topology_snn_parameters = {
    'populations': ['tpE', 'tpI'],
    'population_size': [NE, NI],
    'neurons': [neuron_params, neuron_params],
    'randomize': [
        {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})},
        {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})}],}
    # 'topologies': []}

topology_snn = SpikingNetwork(topology_snn_parameters, label='BRN with spatial topology', topologies=[
    E_layer_properties, I_layer_properties])

# #####################################################################################
# Network Connection
# #####################################################################################
# E synapses
syn_exc = {'model': 'static_synapse', 'delay': d, 'weight': w}
# I synapses
syn_inh = {'model': 'static_synapse', 'delay': d, 'weight': - g * w}

conn_dict = {'connection_type': 'divergent',
             'mask': {'circular': {'radius': 20.}},
             'kernel': {'gaussian': {'p_center': 1.0, 'sigma': 0.25}},
             'synapse_model': 'static_synapse',
             'weights': {'gaussian': {'p_center': w, 'sigma': 0.25 * w}}}
topology_snn_synapses = {
    'connect_populations': [('tpE', 'tpE'), ('tpE', 'tpI'), ('tpI', 'tpE'), ('tpI', 'tpI')],
    'weight_matrix': [None, None, None, None],
    'conn_specs': [{}, {}, {}, {}],
   'syn_specs': [syn_exc, syn_inh, syn_exc, syn_inh]
}
topology_connections = NESTConnector(source_network=topology_snn, target_network=topology_snn,
                                     connection_parameters=topology_snn_synapses, topology=[conn_dict, conn_dict,
                                                                                               conn_dict, conn_dict])
w_rec = topology_connections.compile_weights()


# ############ plots
import matplotlib.pyplot as pl
# from tools.visualization.helper import plot_kernel, plot_mask

fig, ax = pl.subplots()
plot_network_topology(topology_snn, ax=ax, display=False)
plot_spatial_connectivity(topology_snn, kernel=conn_dict['kernel'], mask=conn_dict['mask'], ax=ax)
# plot_mask(topology_snn.populations[0].gids[10], mask=conn_dict['mask'], ax=ax, color='k')
# plot_kernel(topology_snn.populations[0].gids[10], kernel=conn_dict['kernel'], ax=ax, color='k')
pl.show()