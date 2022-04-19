import sys
import os

sys.path.append(os.environ.get('NEST_PYTHON_PREFIX'))

from fna.encoders import NESTEncoder, InputMapper
from fna.tasks.symbolic.embeddings import VectorEmbeddings
from fna.tasks.symbolic.sequences import SymbolicSequencer

from examples import example_defaults

def test():
    # * Create target network:
    resolution = 0.1
    example_defaults.reset_kernel(resolution=resolution, np_seed=123456)
    snn, snn_recurrent_connections = example_defaults.default_network()

    n_strings = 8
    seq_length = 10

    sequencer = SymbolicSequencer(label='random', set_size=10)
    sequence = sequencer.generate_random_sequence(T=seq_length)
    binary_codeword = VectorEmbeddings(vocabulary=sequencer.tokens).binary_codeword(dim=20, density=0.5)
    emb = binary_codeword

    # unfold embedding to continuous signal
    signal_pars = {
        'duration': 50., # single values or rvs
        'amplitude': 250., # single value, list of dim, or rvs
        'kernel': ('box', {}), # (kernel_label, {parameters}).. see documentation
        'dt': 0.1 # dt
    }
    embedded_signal = emb.unfold(to_signal=True, **signal_pars)
    stim_interval = 10.
    stim_onset = 0.1
    stim_seq, stim_info = embedded_signal.draw_stimulus_sequence(sequence, onset_time=stim_onset,
                                                                 continuous=True, intervals=stim_interval)
    # inhomogeneous Poisson generator
    poisson_encoder = NESTEncoder('inhomogeneous_poisson_generator',  label='poisson-input', dim=emb.embedding_dimensions)

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

    extractor_parameters = {
        'ex_E_Vm': {
            'population': 'E',
            'variable': 'V_m',
            'sampling_times': ['stim_offset']  # each offset creates a separate SM
        },
    }

    decoding_parameters = {
        'my_readout_E_svm': {
            'algorithm': 'svm',
            # the following items are optional. Leaving one out (e.g., extractor_label) will create as many readouts as
            # are necessary for all values of that item (e.g., all extractors)
            'extractor': 'ex_E_Vm',    # state matrix label
        }
    }

    snn.connect_state_extractors(extractor_parameters, encoder=poisson_encoder, input_mapper=in_to_snn_connections,
                                 stim_duration=signal_pars['duration'], stim_isi=stim_interval, to_memory=True)
    snn.create_decoder(decoding_parameters)

    #################################################################################################
    # this part if actually done during batch processing, for testing purposes simulate here manually

    # target outputs created separately for each batch
    target_outputs = sequencer.generate_default_outputs(sequence, max_memory=0, max_chunk=0, max_prediction=0,
                                                        chunk_memory=False, chunk_prediction=False)
    # this is called within train when the first batch is processed
    snn.decoders.connect(snn.state_extractor, target_outputs)
