import numpy as np

from fna.tasks.symbolic.embeddings import VectorEmbeddings
from fna.tasks.symbolic.sequences import SymbolicSequencer
from fna.encoders import NESTEncoder, InputMapper
from fna.tools.utils import data_handling as io

from examples import example_defaults

data_path = '../data'  # root path of the project's data folder
data_label = 'batch_tests'  # global label of the experiment (root folder's name within the data path)
instance_label = 'spiking_extraction'


def run_test(stim_interval=None):
    io.set_storage_locations(data_path, data_label, instance_label)

    # * Create target network:
    resolution = 0.1
    example_defaults.reset_kernel(resolution=resolution, np_seed=123456)
    snn, snn_recurrent_connections = example_defaults.default_network()

    n_strings = 8
    seq_length = 3

    sequencer = SymbolicSequencer(label='random', set_size=n_strings)
    sequence = sequencer.generate_random_sequence(T=seq_length)

    binary_codeword = VectorEmbeddings(vocabulary=sequencer.tokens).binary_codeword(dim=20, density=0.5)
    emb = binary_codeword

    signal_pars = {
        'duration': 50., # single values or rvs
        'amplitude': 250., # single value, list of dim, or rvs
        'kernel': ('box', {}), # (kernel_label, {parameters}).. see documentation
        'dt': 0.1 # dt
    }
    sig = emb.unfold(to_signal=True, **signal_pars)

    stim_onset = 0.1
    enc = sig
    stim_seq, stim_info = enc.draw_stimulus_sequence(sequence, onset_time=stim_onset,
                                                     continuous=True, intervals=stim_interval)

    print(stim_info)
    # inhomogeneous Poisson generator
    poisson_encoder = NESTEncoder('inhomogeneous_poisson_generator', stim_seq=stim_seq,
                                  label='poisson-input', dim=emb.embedding_dimensions)
    poisson_encoder.add_parrots(dt=resolution)

    # input synapses
    input_syn = {'model': 'static_synapse', 'delay': 0.1, 'weight': 3.}
    input_conn = {'rule': 'all_to_all'}

    input_synapses = {
        'connect_populations': [('E', poisson_encoder.name), ('I', poisson_encoder.name),],
        'weight_matrix': [None, None],
        'conn_specs': [input_conn, input_conn],
        'syn_specs': [input_syn, input_syn]
    }

    in_to_snn_connections = InputMapper(source=poisson_encoder, target=snn, parameters=input_synapses)
    variable_signal = stim_interval and not isinstance(stim_interval, float)

    extractor_parameters = {
            'ex_E_Vm_@10kHz+isi': {
                'population': 'E',
                'variable': 'V_m',
                'sampling_rate': 10000.,
                'sample_isi': True
            },
            'ex_E_spikes_@offset': {
            'population': 'E',
            'variable': 'spikes',
            'filter_time': 20.,
            'sampling_times': ['stim_offset']
            },
            'ex_parrots_spikes_@10kHz+isi': {
                'population': 'input-parrots',
                'variable': 'spikes',
                'sampling_rate': 10000.,
                'sample_isi': True
            },
    }

    if variable_signal:
        extractor_parameters.update({
            'ex_E_Vm_@10kHz': {
                'population': 'E',
                'variable': 'V_m',
                'sampling_rate': 10000.,
                'sample_isi': False,
                'save': True
            },
            'ex_E_spikes_@10kHz+avg': {
                'population': 'E',
                'variable': 'spikes',
                'filter_time': 20.,
                'average_states': True,
                'sampling_rate': 10000.,
                'sample_isi': False,
                'save': True
            },
            'ex_E_spikes_@10kHz+isi+avg': {
                'population': 'E',
                'variable': 'spikes',
                'filter_time': 20.,
                'average_states': True,
                'sampling_rate': 10000.,
                'sample_isi': True,
                'save': True
            },
        })
    else:
        extractor_parameters.update({
                'ex_E_spikes_@1000Hz': {
                    'population': 'E',
                    'variable': 'spikes',
                    'filter_time': 20.,
                    'sampling_rate': 1000.,
                    'sample_isi': False
                },
                'ex_E_spikes_@100Hz': {
                    'population': 'E',
                    'variable': 'spikes',
                    'filter_time': 20.,
                    'sampling_rate': 100.,
                    'sample_isi': False
                },
                'ex_E_Vm_@100Hz+isi': {
                    'population': 'E',
                    'variable': 'V_m',
                    'sampling_rate': 100.,
                    'sample_isi': True,
                    'save': False
                },
                'ex_E_spikes_@100+avg': {
                    'population': 'E',
                    'variable': 'spikes',
                    'average_states': True,
                    'filter_time': 20.,
                    'sampling_rate': 100.,
                    'sample_isi': False
                },
        })

    if variable_signal:
        snn.connect_state_extractors(extractor_parameters, encoder=poisson_encoder, input_mapper=in_to_snn_connections,
                                     variable_signal=True, stim_onset=stim_onset, to_memory=True)
    else:
        snn.connect_state_extractors(extractor_parameters, encoder=poisson_encoder, input_mapper=in_to_snn_connections,
                                     stim_duration=signal_pars['duration'], stim_isi=stim_interval,
                                     stim_onset=stim_onset, to_memory=True)

    total_delay = in_to_snn_connections.total_delay + snn.state_extractor.max_extractor_delay
    snn.simulate(stim_seq.t_stop + total_delay + resolution)
    snn.state_extractor.extract_global_states(stim_info)
    snn.state_extractor.compile_state_matrices(dataset_label='test_spiking_extractor')
    snn.extract_population_activity()

    snn.state_extractor.report()
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
    snn = run_test(None)
    target_results = {
        'ex_E_Vm_@10kHz+isi': {
            'shape': (80, 1500),
            'n_samples': 1500,
            'times_range': (0.2, 150.1)
        },
        'ex_E_Vm_@100Hz+isi': {
            'shape': (80, 15),
            'n_samples': 15,
            'times_range': (10.1, 150.1)
        },
        'ex_E_spikes_@1000Hz': {
            'shape': (80, 150),
            'n_samples': 150,
            'times_range': (1.1, 150.1)
        },
        'ex_E_spikes_@100Hz': {
            'shape': (80, 15),
            'n_samples': 15,
            'times_range': (10.1, 150.1)
        },
        'ex_E_spikes_@100+avg': {
            'shape': (80, 3),
            'n_samples': 3,
            'times_range': (50.1, 150.1)
        },
        'ex_E_spikes_@offset': {
            'shape': (80, 3),
            'n_samples': 3,
            'times_range': (50.1, 150.1)
        },
    }
    _validate_results(snn, target_results)

def test2():
    snn = run_test(10.)
    target_results = {
        'ex_E_Vm_@10kHz+isi': {
            'shape': (80, 1800),
            'n_samples': 1800,
            'times_range': (0.2, 180.1)
        },
        'ex_E_Vm_@100Hz+isi': {
            'shape': (80, 18),
            'n_samples': 18,
            'times_range': (10.1, 180.1)
        },
        'ex_E_spikes_@1000Hz': {
            'shape': (80, 150),
            'n_samples': 150,
            'times_range': (1.1, 170.1)
        },
        'ex_E_spikes_@100Hz': {
            'shape': (80, 15),
            'n_samples': 15,
            'times_range': (10.1, 170.1)
        },
        'ex_E_spikes_@100+avg': {
            'shape': (80, 3),
            'n_samples': 3,
            'times_range': (50.1, 170.1)
        },
        'ex_E_spikes_@offset': {
            'shape': (80, 3),
            'n_samples': 3,
            'times_range': (50.1, 170.1)
        },
    }
    _validate_results(snn, target_results)

def test3():
    rng = np.random.default_rng(123)
    stim_interval = {'dist': rng.uniform, 'params': {'low':1.0, 'high':10.0}}
    snn = run_test(stim_interval)
    target_results = {
        'ex_E_Vm_@10kHz+isi': {
            'shape': (80, 1600),
            'n_samples': 1600,
            'times_range': (0.2, 160.1)
        },
        'ex_E_spikes_@offset': {
            'shape': (80, 3),
            'n_samples': 3,
            'times_range': (50.1, 158.1)
        },
        'ex_E_Vm_@10kHz': {
            'shape': (80, 1500),
            'n_samples': 1500,
            'times_range': (0.2, 158.1)
        },
        'ex_E_spikes_@10kHz+avg': {
            'shape': (80, 3),
            'n_samples': 3,
            'times_range': (50.1, 158.1)
        },
        'ex_E_spikes_@10kHz+isi+avg': {
            'shape': (80, 3),
            'n_samples': 3,
            'times_range': (57.1, 160.1)
        },
    }
    _validate_results(snn, target_results)

# test3()