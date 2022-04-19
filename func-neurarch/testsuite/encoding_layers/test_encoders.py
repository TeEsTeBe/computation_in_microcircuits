import numpy as np
import nest.raster_plot

from fna.tasks.symbolic import SymbolicSequencer
from fna.tasks.symbolic.embeddings import VectorEmbeddings
from fna.tools.parameters import extract_nestvalid_dict
from fna.tools.utils.system import set_kernel_defaults
from fna.decoders.extractors import set_recording_device

from examples.example_defaults import default_neuron_parameters


def test():
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

    # ###############################
    # random input sequence
    vocabulary_size = 3
    T = 10
    plot_nStrings = 3

    sequencer = SymbolicSequencer(label='random sequence', set_size=vocabulary_size)
    seq = sequencer.generate_random_sequence(T=T)

    sample_sequences = sequencer.draw_subsequences(n_subseq=1, seq=seq, length_range=(5, 15)) # to plot

    # ################################
    # sequence embedding
    # ######
    one_hot = VectorEmbeddings(vocabulary=sequencer.tokens).one_hot()
    binary_codeword = VectorEmbeddings(vocabulary=sequencer.tokens).binary_codeword(dim=100, density=0.3)
    scalar = VectorEmbeddings(vocabulary=sequencer.tokens).real_valued(dim=1, distribution=np.random.uniform, parameters={
        'low': -1., 'high': 1.})
    random_vector = VectorEmbeddings(vocabulary=sequencer.tokens).real_valued(dim=100, distribution=np.random.normal,
                                                                        parameters={'loc': 0., 'scale': 0.5})
    # #######
    enc = one_hot # random_vector #binary_codeword

    # ################################
    # input from embedding
    # ######
    # a) continuous signal
    signal_pars = {
        'duration': 20., # single values or rvs
        'amplitude': 5., # single value, list of dim, or rvs
        'kernel': ('box', {}), # (kernel_label, {parameters}).. see documentation
        'dt': 0.1 # dt
    }
    sig = enc.unfold(to_signal=True, **signal_pars)
    stim_seq, time_info = sig.draw_stimulus_sequence(seq, onset_time=0.1, continuous=False, intervals=None)

    # ###############
    # b) Frozen noise
    # spk_pattern = DynamicEmbeddings(vocabulary=sequencer.tokens).frozen_noise(n_processes=100, pattern_duration=200., rate=10.,
    #                                                                     resolution=0.1, jitter=None, rng=None)
    # stim_seq = spk_pattern.draw_stimulus_sequence(seq, onset_time=0.1, continuous=False, intervals=None)

    # ###############
    # c) unfold to spikes
    # spk_encoding_pars = {
    #     'duration': 150.,
    #     'rate_scale': 100.,
    #     'dt': 0.1,
    #     'jitter': None
    # }
    # spk_vec = DynamicEmbeddings(vocabulary=sequencer.tokens).unfold_discrete_set(stimulus_set=enc.stimulus_set,
    #                                                                    to_spikes=True, to_signal=False, **spk_encoding_pars)
    # stim_seq = spk_vec.draw_stimulus_sequence(seq, onset_time=0.1, continuous=True, intervals=100.)

    # ###
    embedding = sig # spk_vec

    # ################################
    # create encoding layer
    # ################################
    from fna.encoders import NESTEncoder, InputMapper

    # 1) no encoder population, stimulus delivered by generators
    # enc_layer = NESTEncoder('spike_generator', stim_seq=stim_seq, label='spike-pattern-input',
    #                         dim=embedding.embedding_dimensions)
    enc_layer = NESTEncoder('step_current_generator', stim_seq=stim_seq, label='spike-pattern-input',
                            input_resolution=sig.dt, dim=embedding.embedding_dimensions)

    # ################################
    # Create target snn
    # ######
    from fna.networks.snn import SpikingNetwork
    from fna.tools.network_architect.connectivity import NESTConnector

    # network parameters
    gamma = 0.25               # relative number of inhibitory connections
    NE = 500                  # number of excitatory neurons (10.000 in [1])
    NI = int(gamma * NE)       # number of inhibitory neurons
    CE = 1000                  # indegree from excitatory neurons
    CI = int(gamma * CE)       # indegree from inhibitory neurons
    # synapse parameters
    w = 0.1                    # excitatory synaptic weight (mV)
    g = 5.                     # relative inhibitory to excitatory synaptic weight
    d = 1.5                    # synaptic transmission delay (ms)
    neuron_params = default_neuron_parameters(default_set=0) # for convenience (full parameters can be passed)
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

    snn = SpikingNetwork(snn_parameters, label='Brunel BRN', spike_recorders=[set_recording_device(), set_recording_device()])
    snn_recurrent_connections = NESTConnector(source_network=snn, target_network=snn, connection_parameters=snn_synapses)

    # ################################
    # connect encoding layer
    # ######

    # input synapses
    input_syn = {'model': 'static_synapse', 'delay': 0.1, 'weight': 1.}
    input_conn = {'rule': 'all_to_all'}

    input_synapses = {
        'connect_populations': [('E', 'spike-pattern-input'), ('I', 'spike-pattern-input'),],
        'weight_matrix': [None, None],
        'conn_specs': [input_conn, input_conn],
        'syn_specs': [input_syn, input_syn]
    }
    in_to_snn_connections = InputMapper(source=enc_layer, target=snn, parameters=input_synapses)
    # w_in = in_to_snn_connections.compile_weights()



    nest.Simulate(1000.)

    # inp = stim_seq.time_slice(t_start=0., t_stop=1000.)
    # inp.raster_plot(with_rate=True)
    # nest.raster_plot.from_device(snn.device_gids[0][0])
