import numpy as np
import nest

from fna.encoders import NESTEncoder, InputMapper

from fna.tasks.symbolic.embeddings import VectorEmbeddings
from fna.tasks.symbolic.sequences import SymbolicSequencer

from examples import example_defaults
from fna.tools.utils import data_handling as io

data_path = '../data'  # root path of the project's data folder
data_label = 'readout_tests'  # global label of the experiment (root folder's name within the data path)
instance_label = 'spiking_readout'
io.set_storage_locations(data_path, data_label, instance_label)

def test():
    # Create target network
    resolution = 0.1
    rng = np.random.default_rng(12369874)
    example_defaults.reset_kernel(resolution=resolution, np_seed=123456)
    snn, snn_recurrent_connections = example_defaults.default_network(N=100)

    # Create the input (sequence, encoding)
    alphabet_size = 3
    sequencer = SymbolicSequencer(label='random', set_size=alphabet_size, rng=rng)
    emb = VectorEmbeddings(vocabulary=sequencer.tokens, rng=rng).binary_codeword(dim=50, density=0.2)

    # unfold embedding to continuous signal
    signal_pars = {
        'duration': 20.,  # single values or rvs
        'amplitude': 250.,  # single value, list of dim, or rvs
        'kernel': ('box', {}),  # (kernel_label, {parameters}).. see documentation
        'dt': 0.1  # dt
    }
    embedded_signal = emb.unfold(to_signal=True, **signal_pars)

    # inhomogeneous Poisson generator
    # create it without a stim_seq initially, update state later on during each batch separately
    encoder = NESTEncoder('inhomogeneous_poisson_generator', label='poisson-input',
                          dim=emb.embedding_dimensions)
    # input synapses
    input_synapses = {
        'connect_populations': [('E', encoder.name), ('I', encoder.name), ],
        'weight_matrix': [None, None],
        'conn_specs': [{'rule': 'all_to_all'}, {'rule': 'all_to_all'}],
        'syn_specs': [{'model': 'static_synapse', 'delay': 0.1, 'weight': 3.},
                      {'model': 'static_synapse', 'delay': 0.1, 'weight': 3.}]
    }
    in_to_snn_connections = InputMapper(source=encoder, target=snn, parameters=input_synapses)

    # state extractor
    extractor_parameters = {
            'E_Vm_@offset': {
                'population': 'E',
                'variable': 'V_m',
                'sampling_times': ['stim_offset']}}

    decoding_parameters = {
        'readout_E_ridge': {
            'algorithm': 'ridge',
            'extractor': 'E_Vm_@offset',  # state (extractor) matrix label
            'save': True  # whether to store intermediate weights trained after each batch
        },
        'readout_E_pinv': {
            'algorithm': 'pinv',
            'extractor': 'E_Vm_@offset',  # state (extractor) matrix label
            'save': True  # whether to store intermediate weights trained after each batch
        },
        'readout_E_logistic': {
            'algorithm': 'logistic',
            'extractor': 'E_Vm_@offset',  # state (extractor) matrix label
            'save': True  # whether to store intermediate weights trained after each batch
        },
        'readout_E_perceptron': {
            'algorithm': 'perceptron',
            'extractor': 'E_Vm_@offset',  # state (extractor) matrix label
            'save': True  # whether to store intermediate weights trained after each batch
        },
        'readout_E_svm-linear': {
            'algorithm': 'svm-linear',
            'extractor': 'E_Vm_@offset',  # state (extractor) matrix label
            'save': True  # whether to store intermediate weights trained after each batch
        },
        'readout_E_svm-rbf': {
            'algorithm': 'svm-rbf',
            'extractor': 'E_Vm_@offset',  # state (extractor) matrix label
            'save': True  # whether to store intermediate weights trained after each batch
        },
        'readout_E_force': {
            'algorithm': 'force',
            'extractor': 'E_Vm_@offset',  # state (extractor) matrix label
            'save': True  # whether to store intermediate weights trained after each batch
        }
    }

    ####################################################################################################################
    n_epochs = 1
    n_batches = 1
    batch_size = 20
    continuous = False
    stim_onset = 0.1

    # TODO @zbarni this used to be after the transient set, check if there's a conflict?
    # # create and connect extractors and decoders
    # snn.connect_state_extractors(extractor_parameters, encoder=encoder, input_mapper=in_to_snn_connections,
    #                              stim_duration=signal_pars['duration'], stim_isi=None, stim_onset=stim_onset,
    #                              to_memory=True)
    # snn.create_decoder(decoding_parameters)
    # total_delay = in_to_snn_connections.total_delay + snn.state_extractor.max_extractor_delay

    ####################################################################################################################
    # Transient set
    batch_sequence = sequencer.generate_random_sequence(T=2, verbose=False)
    batch_stim_seq = embedded_signal.draw_stimulus_sequence(batch_sequence, onset_time=stim_onset,
                                                            continuous=continuous, intervals=None, verbose=False)
    snn.process_batch(batch_label='transient', encoder=encoder, stim_seq=batch_stim_seq)
    stim_onset = snn.next_onset_time()

    ####################################################################################################################
    # Training
    # generate training batches
    batch_labels = ['{}_train_batch={}'.format(data_label, batch+1) for batch in range(n_batches)]
    batch_seq = [sequencer.generate_random_sequence(T=batch_size, verbose=False) for _ in range(n_batches)]
    batch_targets = [sequencer.generate_default_outputs(batch_sequence, max_memory=0, max_prediction=0,
                        max_chunk=0, chunk_memory=False, chunk_prediction=False) for batch_sequence in batch_seq]

    # create and connect extractors and decoders
    snn.connect_state_extractors(extractor_parameters, encoder=encoder, input_mapper=in_to_snn_connections,
                                 stim_duration=signal_pars['duration'], stim_isi=None, stim_onset=stim_onset,
                                 to_memory=True)
    snn.create_decoder(decoding_parameters, rng=rng)
    total_delay = in_to_snn_connections.total_delay + snn.state_extractor.max_extractor_delay

    # train
    data_batch = {'inputs': batch_seq, 'decoder_outputs': batch_targets}
    snn.train(data_batch, n_epochs, embedded_signal, encoder, stim_onset, total_delay=total_delay,
              symbolic=True, continuous=continuous, verbose=True, save=False)
    ####################################################################################################################
    # Prediction / Testing

    stim_onset = snn.next_onset_time()
    test_size = 10
    test_label = 'fna_test_set'
    test_sequence = sequencer.generate_random_sequence(T=test_size)
    target_outputs = sequencer.generate_default_outputs(test_sequence, max_memory=0, max_chunk=0, max_prediction=0,
                                                        chunk_memory=False, chunk_prediction=False)

    data_test = {'inputs': [test_sequence], 'decoder_outputs': [target_outputs]}
    test_results = snn.test(data_test, embedded_signal, encoder, snn.next_onset_time(), total_delay=total_delay,
                            symbolic=True, continuous=continuous, output_parsing="k-WTA")

test()