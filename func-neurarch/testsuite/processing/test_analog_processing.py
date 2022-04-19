#import tensorflow as tf
import nest
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from fna.tools.parameters import ParameterSet
from fna.tasks.symbolic import SymbolicSequencer
from fna.tasks.symbolic.embeddings import VectorEmbeddings
from fna.networks.rnn.helper import prepare_analog_batch

# ######################################################################################################################
# Continuous integration task
# ######################################################################################################################
data_label = 'analog_processing'
# system parameters
resolution = 1.
net_type = 'SNN'

# task parameters
n_epochs = 2
n_batches = 5
batch_size = 20

sequencer = SymbolicSequencer(label='random sequence', alphabet=['A', 'B'], eos='#')
embedding = VectorEmbeddings(vocabulary=sequencer.tokens).one_hot()
embedding.stimulus_set.update({'#': np.zeros_like(embedding.stimulus_set['A']),
                               'B': -1*embedding.stimulus_set['B']})

# inputs
signal_pars = {
    'duration': 10., # [ms]
    'amplitude': 1., # [max. rate]
    'kernel': ('box', {}),# (kernel, {kernel_pars})
    'dt': resolution # [ms]
}
sig = embedding.unfold(to_signal=True, **signal_pars)
sequencer.generate_stringset(set_length=batch_size, length_range=(2, 2), verbose=False)
batch_seq = sequencer.generate_sequence()
sequencer.string_set = []
batch_time = (signal_pars['duration'] * len(batch_seq)) / resolution


def tst_ANN():
    # ######################################################################################################################
    # 1) VanillaRNN
    # ######################################################################################################################
    from fna.networks.ann import ArtificialNeuralNetwork

    net_params = {
        'N': 500,
        'cell_type': 'LSTM',
        'learning_rate': 2e-2
    }
    rnn = ArtificialNeuralNetwork(label='LSTM', network_parameters=net_params, input_dim=1, output_dim=1)

    # train
    train_batch = prepare_analog_batch(rnn.simulator, n_batches=n_batches, batch_size=batch_size, sequencer=sequencer,
                                   discrete_embedding=embedding, as_tensor=False)

    train_results = rnn.train(train_batch, n_epochs=n_epochs, verbose=True, save=False)


def test_rRNN():
    # ######################################################################################################################
    # 3) rRNN
    # ######################################################################################################################
    from fna.networks.rnn import ReservoirRecurrentNetwork

    N = 100
    taus = np.ones(N) * 100.
    noise = np.sqrt(2*(resolution/taus)) * 0.5

    net_params = {
        'N': N,
        'transfer_fcn': 'sigmoid',
        'EI_ratio': 0,
        'tau_x': taus,
        'tau_r': np.ones_like(taus)*resolution,
        'noise': noise.astype(np.float32),
        'initial_state': (np.random.uniform, {'low': 0., 'high': 1.}),
        'learning_rate': 1e-2
    }
    connection_params = {
        'w_in': {
            'density': 1.,
            'distribution': (np.random.uniform, {'low': -1., 'high': 1.})
        },
        'w_rec': {
            'ei_balance': 4.,
            'density': .1,
            'distribution': (np.random.gamma, {'shape': 0.2, 'scale': 1.})
        },
        'w_out': {
            'density': 1.,
            'distribution': (np.random.gamma, {'shape': 0.1, 'scale': 1.})
        }
    }

    extractor_parameters = {
        'r_@offset': {
            'population': None,
            'variable': 'active',
            'sampling_times': ['stim_offset'],
            'save': True}}
    decoding_parameters = {
        'SGD-Regression': {
            'algorithm': "sgd-reg",
            'extractor': 'r_@offset',
            'save': True
        },}

    rnn = ReservoirRecurrentNetwork(label='rRNN-random-amorphous', network_parameters=net_params,
                                    connection_parameters=connection_params, input_dim=1,
                                    extractors=extractor_parameters, decoders=decoding_parameters,
                                    output_dim=1, dt=resolution)
    # Train set
    train_batch = prepare_analog_batch(rnn.simulator, n_batches=n_batches, batch_size=batch_size,
                                       batch_time=int(batch_time), sequencer=sequencer, continuous_embedding=sig,
                                       as_tensor=True)

    train_results = rnn.train(data_batch=train_batch, n_epochs=n_epochs, symbolic=False, save=False)


def test_SNN():
    # ######################################################################################################################
    # 3) SNN
    # ######################################################################################################################
    from examples import example_defaults
    from fna.encoders import NESTEncoder, InputMapper

    example_defaults.reset_kernel(resolution=0.1)
    rnn, rnn_recurrent_connections = example_defaults.default_network(N=100, record_spikes=True)

    encoder = NESTEncoder('inhomogeneous_poisson_generator', label='poisson-input', dim=2)
    # input synapses
    input_synapses = {
        'connect_populations': [('E', encoder.name), ('I', encoder.name), ],
        'weight_matrix': [None, None],
        'conn_specs': [{'rule': 'all_to_all'}, {'rule': 'all_to_all'}],
        'syn_specs': [{'model': 'static_synapse', 'delay': 0.1, 'weight': 1.},
                      {'model': 'static_synapse', 'delay': 0.1, 'weight': 1.}]
    }
    in_to_rnn_connections = InputMapper(source=encoder, target=rnn, parameters=input_synapses)

    # decoder
    extractor_parameters = {
        'E_Vm_@offset': {
            'population': 'E',
            'variable': 'V_m',
            'sampling_times': ['stim_offset'],
            'save': True}}
    decoding_parameters = {
        'readout_E_force': {'algorithm': 'force', 'extractor': 'E_Vm_@offset', 'save': True},
        'SGD-Regression': {'algorithm': "sgd-reg", 'extractor': 'E_Vm_@offset', 'save': True},
    }

    # create and connect extractors and decoders
    rnn.connect_state_extractors(extractor_parameters, encoder=encoder, input_mapper=in_to_rnn_connections,
                                 stim_duration=signal_pars['duration'], stim_isi=None, stim_onset=rnn.next_onset_time(),
                                 to_memory=True)
    rnn.create_decoder(decoding_parameters)
    total_delay = in_to_rnn_connections.total_delay + rnn.state_extractor.max_extractor_delay

    # train
    train_batch = prepare_analog_batch(rnn.simulator, n_batches=n_batches, batch_size=batch_size,
                                       batch_time=int(batch_time), sequencer=sequencer, continuous_embedding=sig,
                                       as_tensor=True)
    train_results = rnn.train(train_batch, n_epochs, sig, encoder, rnn.next_onset_time(),
                              total_delay=total_delay, symbolic=False, continuous=True, verbose=True, save=False)