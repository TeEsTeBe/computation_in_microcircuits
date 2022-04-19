import sys
import numpy as np

from examples.example_defaults import default_neuron_parameters
from fna.networks.snn import SpikingNetwork
from fna.decoders.extractors import set_recording_device
from fna.tools.utils.system import set_kernel_defaults, reset_nest_kernel
from fna.tools.network_architect.topology import set_positions
from fna.tools.utils.operations import copy_dict

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

# Specify network parameters
gamma = 0.25  # relative number of inhibitory connections
NE = 5000  # number of excitatory neurons (10.000 in [1])
NI = int(gamma * NE)  # number of inhibitory neurons
CE = 1000  # indegree from excitatory neurons
CI = int(gamma * CE)  # indegree from inhibitory neurons

# synapse parameters
w = 0.1  # excitatory synaptic weight (mV)
g = 5.  # relative inhibitory to excitatory synaptic weight
d = 1.5  # synaptic transmission delay (ms)

neuron_params = default_neuron_parameters(default_set=0)  # for convenience (full parameters can be passed)


def test_simple_brunel():
    reset_nest_kernel(kernel_pars)
    # #####################################################################################
    # ## A) Simple Brunel BRN
    # #####################################################################################
    snn_parameters = {
        'populations': ['E', 'I'],
        'population_size': [NE, NI],
        'neurons': [neuron_params, neuron_params],
        'randomize': [
            {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})},
            {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})}]}

    spike_recorder = set_recording_device(start=0., stop=sys.float_info.max, resolution=resolution, record_to='memory',
                                          device_type='spike_detector')
    spike_recorders = [spike_recorder for _ in snn_parameters['populations']]

    # Generate SNN instance
    snn = SpikingNetwork(snn_parameters, label='Brunel BRN', topologies=None, spike_recorders=spike_recorders,
                         analog_recorders=None)

def test_nested_populations():
    reset_nest_kernel(kernel_pars)
    # #####################################################################################
    # ## B) SNN with nested populations
    # #####################################################################################
    nested_snn_parameters = {
        'populations': ['E1', 'I1', 'E2', 'I2'],
        'merged_populations': [['E1', 'I1'], ['E2', 'I2']],
        'population_size': [NE, NI, NE, NI],
        'neurons': [neuron_params, neuron_params, neuron_params, neuron_params],
        'randomize': [
            {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})},
            {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})},
            {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})},
            {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})}]}
    nested_snn = SpikingNetwork(nested_snn_parameters, label='Nested BRN')

def test_topology():
    reset_nest_kernel(kernel_pars)
    # #####################################################################################
    # ## C) SNN with topology
    # #####################################################################################
    N = NE + NI

    # for simplicity all other parameters are the same, only topology is added
    layer_properties = {
        'center': [np.round(np.sqrt(N)) / 2., np.round(np.sqrt(N)) / 2.],
        'extent': [np.ceil(np.sqrt(N)), np.ceil(np.sqrt(N))],
        'edge_wrap': True, 'elements': neuron_params['model']}
    pos = set_positions(N=N, dim=2, topology='random')
    E_layer_properties = copy_dict(layer_properties, {'positions': pos[:int(NE)]})
    I_layer_properties = copy_dict(layer_properties, {'positions': pos[int(NE):]})


    topology_snn_parameters = {
        'populations': ['tpE', 'tpI'],
        'population_size': [NE, NI],
        'neurons': [neuron_params, neuron_params],
        'randomize': [
            {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})},
            {'V_m': (np.random.uniform, {'low': neuron_params['E_L'], 'high': neuron_params['V_th']})}]}

    topology_snn = SpikingNetwork(topology_snn_parameters, label='BRN with spatial topology', topologies=[
        E_layer_properties, I_layer_properties])

    # import matplotlib.pyplot as pl
    # fig, ax = pl.subplots()
    # plot_network_topology(topology_snn, ax=ax, display=False)
