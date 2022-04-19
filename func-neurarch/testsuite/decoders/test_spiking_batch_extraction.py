import numpy as np

from fna.encoders import NESTEncoder, InputMapper
from fna.tasks.symbolic.embeddings import VectorEmbeddings
from fna.tasks.symbolic.sequences import SymbolicSequencer
from fna.tools.utils import data_handling as io

from examples import example_defaults

data_path = '../data'  # root path of the project's data folder
data_label = 'batch_tests'  # global label of the experiment (root folder's name within the data path)
instance_label = 'spiking_extraction'


def run_test(stim_interval=None, transient_set=False, variable=False):
    io.set_storage_locations(data_path, data_label, instance_label)

    # Create target network
    resolution = 0.1
    example_defaults.reset_kernel(resolution=resolution, np_seed=123456)
    snn, snn_recurrent_connections = example_defaults.default_network()

    # Create the input (sequence, encoding)
    rng = np.random.default_rng(112233)
    set_size = 2

    random_sequencer = SymbolicSequencer(label='random', set_size=set_size, rng=rng)
    sequencer = random_sequencer
    binary_codeword = VectorEmbeddings(vocabulary=sequencer.tokens, rng=rng).binary_codeword(dim=20, density=0.5)
    emb = binary_codeword

    # ################################
    # input from embedding
    # unfold embedding to continuous signal
    signal_pars = {
        'duration': 50., # single values or rvs
        'amplitude': 250., # single value, list of dim, or rvs
        'kernel': ('box', {}), # (kernel_label, {parameters}).. see documentation
        'dt': 0.1 # dt
    }
    embedded_signal = emb.unfold(to_signal=True, **signal_pars)
    stim_onset = 0.1

    # inhomogeneous Poisson generator
    # create it without a stim_seq initially, update state later on during each batch separately
    encoder = NESTEncoder('inhomogeneous_poisson_generator', label='poisson-input',
                          dim=emb.embedding_dimensions)

    # input synapses
    input_syn = {'model': 'static_synapse', 'delay': 0.1, 'weight': 3.}
    input_conn = {'rule': 'all_to_all'}

    input_synapses = {
        'connect_populations': [('E', encoder.name), ('I', encoder.name), ],
        'weight_matrix': [None, None],
        'conn_specs': [input_conn, input_conn],
        'syn_specs': [input_syn, input_syn]
    }
    in_to_snn_connections = InputMapper(source=encoder, target=snn, parameters=input_synapses)

    variable_signal = (stim_interval and not isinstance(stim_interval, float)) or variable
    if not variable_signal:
        extractor_parameters = {
            'ex_E_Vm_@10kHz+isi': {
                'population': 'E',
                'variable': 'V_m',
                'sampling_rate': 10000.,
                'sample_isi': True
            },
            'ex_E_Vm_@100Hz+isi': {
                'population': 'E',
                'variable': 'V_m',
                'sampling_rate': 100.,
                'sample_isi': True,
                'save': False
            },
            'ex_E_spikes_@100Hz': {
                'population': 'E',
                'variable': 'spikes',
                'filter_time': 20.,
                'sampling_rate': 100.,
                'sample_isi': False
            },
            'ex_E_spikes_@100+avg': {
                'population': 'E',
                'variable': 'spikes',
                'average_states': True,
                'filter_time': 20.,
                'sampling_rate': 100.,
                'sample_isi': False
            },
            'ex_E_spikes_@offset': {
                'population': 'E',
                'variable': 'spikes',
                'filter_time': 20.,
                'sampling_times': ['stim_offset']
            },
        }
    else:
        extractor_parameters = {
            'ex_E_Vm_@10kHz+isi': {
                'population': 'E',
                'variable': 'V_m',
                'sampling_rate': 10000.,
                'sample_isi': True
            },
            'ex_E_spikes_@10kHz': {
                'population': 'E',
                'variable': 'spikes',
                'filter_time': 20.,
                'sampling_rate': 10000.,
                'sample_isi': False
            },
            'ex_E_spikes_@10kHz+avg': {
                'population': 'E',
                'variable': 'spikes',
                'average_states': True,
                'filter_time': 20.,
                'sampling_rate': 10000.,
                'sample_isi': False
            },
            'ex_E_spikes_@offset': {
                'population': 'E',
                'variable': 'spikes',
                'filter_time': 20.,
                'sampling_times': ['stim_offset']
            },
        }

    decoding_parameters = {
        'readout_E_ridge': {
            'algorithm': 'ridge',
            'extractor': 'ex_E_Vm_@100Hz+isi',  # state (extractor) matrix label
            'save': False  # whether to store intermediate weights trained after each batch
        },
        'readout_E_spikes_ridge': {
            'algorithm': 'ridge',
            'extractor': 'ex_E_spikes_@offset',  # state (extractor) matrix label
        }
    }

    ####################################################################################################################
    #################### BATCH STUFF
    ####################################################################################################################
    n_epochs = 1
    n_batches = 2
    batch_size = 3
    # continuous = False
    continuous = True

    ####################################################################################################################
    # Transient set
    if transient_set:
        batch_sequence = random_sequencer.generate_random_sequence(T=2)
        batch_stim_seq = embedded_signal.draw_stimulus_sequence(batch_sequence, onset_time=stim_onset,
                                                                continuous=continuous, intervals=stim_interval)
        snn.process_batch('transient', encoder, batch_stim_seq)
        stim_onset = snn.next_onset_time()

    ####################################################################################################################
    # Training
    if variable_signal:
        snn.connect_state_extractors(extractor_parameters, encoder=encoder, input_mapper=in_to_snn_connections,
                                     variable_signal=True, stim_onset=stim_onset, to_memory=True)
    else:
        snn.connect_state_extractors(extractor_parameters, encoder=encoder, input_mapper=in_to_snn_connections,
                                     stim_duration=signal_pars['duration'], stim_isi=stim_interval,
                                     stim_onset=stim_onset, to_memory=True)

    snn.create_decoder(decoding_parameters, rng=rng)
    total_delay = in_to_snn_connections.total_delay + snn.state_extractor.max_extractor_delay

    # batch_labels = ['{}_train_batch={}'.format(data_label, batch + 1) for batch in range(n_batches)]
    batch_seq = [random_sequencer.generate_random_sequence(T=batch_size, verbose=False) for _ in range(n_batches)]
    batch_targets = [random_sequencer.generate_default_outputs(batch_sequence, max_memory=0, max_prediction=0,
        max_chunk=0, chunk_memory=False, chunk_prediction=False) for batch_sequence in batch_seq]

    # train
    data_batch = {'inputs': batch_seq, 'decoder_outputs': batch_targets}
    snn.train(data_batch, n_epochs, embedded_signal, encoder, stim_onset, intervals=stim_interval,
              total_delay=total_delay, symbolic=True, continuous=continuous, verbose=True, save=False)

    ####################################################################################################################
    # Prediction / Testing
    test_size = 2
    test_label = 'fna_test_set'
    test_sequence = sequencer.generate_random_sequence(T=test_size)
    target_outputs = sequencer.generate_default_outputs(test_sequence, max_memory=0, max_chunk=0, max_prediction=0,
                                                        chunk_memory=False, chunk_prediction=False)

    data_test = {'inputs': [test_sequence], 'decoder_outputs': [target_outputs]}
    test_results = snn.test(data_test, embedded_signal, encoder, snn.next_onset_time(), total_delay=total_delay,
                            intervals=stim_interval, symbolic=True, continuous=continuous, output_parsing="k-WTA")

    return snn


def _validate_results(snn, target_results):
    # before the validation times considered the sim_res delay of 0.1 as an extractor delay, hence they were shifted
    # 0.1 ms to the left. Here we compensate for this..
    sim_res_delay = 0.0

    for label, targets in target_results.items():
        print("Testing extractor [{}]...".format(label))
        state_matrix_obj = snn.state_extractor.get_state_matrix(label)
        assert targets['shape'] == state_matrix_obj.matrix.shape
        assert targets['n_samples'] == len(state_matrix_obj.sampled_times[0])
        assert np.isclose(targets['times_range'][0] + sim_res_delay, min(state_matrix_obj.sampled_times[0]))
        assert np.isclose(targets['times_range'][1] + sim_res_delay, max(state_matrix_obj.sampled_times[0]))
        print("\tTest passed")


def test1():
    snn = run_test(None, False)
    target_results = {
        'ex_E_Vm_@10kHz+isi': {
            'shape': (80, 1000),
            'n_samples': 1000,
            'times_range': (300.6, 400.5)
        },
        'ex_E_Vm_@100Hz+isi': {
            'shape': (80, 10),
            'n_samples': 10,
            'times_range': (310.5, 400.5)
        },
        'ex_E_spikes_@100Hz': {
            'shape': (80, 10),
            'n_samples': 10,
            'times_range': (310.5, 400.5)
        },
        'ex_E_spikes_@100+avg': {
            'shape': (80, 2),
            'n_samples': 2,
            'times_range': (350.5, 400.5)
        },
        'ex_E_spikes_@offset': {
            'shape': (80, 2),
            'n_samples': 2,
            'times_range': (350.5, 400.5)
        },
    }
    _validate_results(snn, target_results)


def test2():
    snn = run_test(10., False)
    target_results = {
        'ex_E_Vm_@10kHz+isi': {
            'shape': (80, 1200),
            'n_samples': 1200,
            'times_range': (360.6, 480.5)
        },
        'ex_E_Vm_@100Hz+isi': {
            'shape': (80, 12),
            'n_samples': 12,
            'times_range': (370.5, 480.5)
        },
        'ex_E_spikes_@100Hz': {
            'shape': (80, 10),
            'n_samples': 10,
            'times_range': (370.5, 470.5)
        },
        'ex_E_spikes_@100+avg': {
            'shape': (80, 2),
            'n_samples': 2,
            'times_range': (410.5, 470.5)
        },
        'ex_E_spikes_@offset': {
            'shape': (80, 2),
            'n_samples': 2,
            'times_range': (410.5, 470.5)
        },
    }
    _validate_results(snn, target_results)


def test3():
    snn = run_test(10., True)
    target_results = {
        'ex_E_Vm_@10kHz+isi': {
            'shape': (80, 1200),
            'n_samples': 1200,
            'times_range': (480.6, 600.5)
        },
        'ex_E_Vm_@100Hz+isi': {
            'shape': (80, 12),
            'n_samples': 12,
            'times_range': (490.5, 600.5)
        },
        'ex_E_spikes_@100Hz': {
            'shape': (80, 10),
            'n_samples': 10,
            'times_range': (490.5, 590.5)
        },
        'ex_E_spikes_@100+avg': {
            'shape': (80, 2),
            'n_samples': 2,
            'times_range': (530.5, 590.5)
        },
        'ex_E_spikes_@offset': {
            'shape': (80, 2),
            'n_samples': 2,
            'times_range': (530.5, 590.5)
        },
    }
    _validate_results(snn, target_results)


######################
# VARIABLE ISI
def test4():
    snn = run_test(None, False, variable=True)
    target_results = {
        'ex_E_Vm_@10kHz+isi': {
            'shape': (80, 1000),
            'n_samples': 1000,
            'times_range': (300.6, 400.5)
        },
        'ex_E_spikes_@10kHz': {
            'shape': (80, 1000),
            'n_samples': 1000,
            'times_range': (300.6, 400.5)
        },
        'ex_E_spikes_@10kHz+avg': {
            'shape': (80, 2),
            'n_samples': 2,
            'times_range': (350.5, 400.5)
        },
        'ex_E_spikes_@offset': {
            'shape': (80, 2),
            'n_samples': 2,
            'times_range': (350.5, 400.5)
        },
    }
    _validate_results(snn, target_results)


# def test5():
#     stim_interval = {'dist': np.random.uniform, 'params': {'low':1.0, 'high':10.0}}
#     snn = run_test(stim_interval, True)
#     target_results = {
#         'ex_E_Vm_@10kHz+isi': {
#             'shape': (80, 1130),
#             'n_samples': 1130,
#             'times_range': (434.6, 547.5)
#         },
#         'ex_E_spikes_@10kHz': {
#             'shape': (80, 1000),
#             'n_samples': 1000,
#             'times_range': (434.6, 541.5)
#         },
#         'ex_E_spikes_@10kHz+avg': {
#             'shape': (80, 2),
#             'n_samples': 2,
#             'times_range': (484.5, 541.5)
#         },
#         'ex_E_spikes_@offset': {
#             'shape': (80, 2),
#             'n_samples': 2,
#             'times_range': (484.5, 541.5)
#         },
#     }
#     _validate_results(snn, target_results)

# test1()
# test2()
# test3()
# test4()
# test5()