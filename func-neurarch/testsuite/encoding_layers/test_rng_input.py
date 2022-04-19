import numpy as np
import nest.raster_plot
import hashlib

from fna.tasks.symbolic import SymbolicSequencer, NonAdjacentDependencies, ArtificialGrammar
from fna.tasks.symbolic.embeddings import VectorEmbeddings, DynamicEmbeddings
from fna.tools.parameters import extract_nestvalid_dict
from fna.tools.utils.system import set_kernel_defaults
from fna.decoders.extractors import set_recording_device

from examples.example_defaults import default_neuron_parameters


def test_sequences():
    rng_seed = 123
    rng = np.random.default_rng(rng_seed)

    ### SymbolicSequencer
    random_sequencer = SymbolicSequencer(label='random', set_size=10, rng=rng)
    sample_sequences = random_sequencer.draw_subsequences(n_subseq=2, length_range=(5, 10))
    assert sample_sequences == [['0', '6', '5', '0', '9', '2'], ['1', '3', '8', '4', '9', '4']]

    ### NonAdjacentDependencies
    nad_sequencer = NonAdjacentDependencies(vocabulary_size=10, filler_variability=1, dependency_length=4, rng=rng)
    nad_sequencer.generate_stringset(set_length=100, generator=False, violations=None)
    assert nad_sequencer.string_set[0] == ['A2', 'X0', 'X0', 'X0', 'X0', 'B2']


def test_embeddings():
    rng_seed = 1234
    rng = np.random.default_rng(rng_seed)

    # ################################
    # input from embedding
    # ######
    ######## a) continuous signal
    # a) unfold to spikes
    spk_encoding_pars = {
        'duration': 150.,
        'rate_scale': 100.,
        'dt': 0.1,
        'jitter': None
    }

    # #######
    random_sequencer = SymbolicSequencer(label='random', set_size=10, rng=rng)
    sequence = random_sequencer.generate_random_sequence(T=12)
    enc = VectorEmbeddings(vocabulary=random_sequencer.tokens, rng=rng).one_hot()
    sig = enc.unfold(to_spikes=True, **spk_encoding_pars)
    stim_seq, time_info = sig.draw_stimulus_sequence(sequence, onset_time=0.1, continuous=True, intervals=None)
    assert sequence[:5] == ['9', '9', '9', '3', '1']
    assert np.all(np.isclose(stim_seq.spiketrains[2].spike_times[:3], [1070.,  1070.,  1079.1]))

    ###############
    # b) Frozen noise
    random_sequencer = SymbolicSequencer(label='random', set_size=10, rng=rng)
    sequence = random_sequencer.generate_random_sequence(T=6)
    emb = DynamicEmbeddings(vocabulary=random_sequencer.tokens, rng=rng)
    spk_pattern = emb.frozen_noise(n_processes=100, pattern_duration=200., rate=10., resolution=0.1, jitter=None)
    stim_seq = spk_pattern.draw_stimulus_sequence(sequence, onset_time=0.1, continuous=False, intervals=None)
    assert np.all(np.isclose(spk_pattern.stimulus_set['0'].spiketrains[2.].spike_times, np.array([110.4, 146.3, 183.4])))

    ### VectorEmbeddings
    one_hot = VectorEmbeddings(vocabulary=random_sequencer.tokens, rng=rng).one_hot()
    binary_codeword = VectorEmbeddings(vocabulary=random_sequencer.tokens, rng=rng).binary_codeword(dim=100,
                                                                                                    density=0.3)
    # assert hashlib.md5(binary_codeword.stimulus_set['0']).hexdigest() == 'b2c2b4c63837902c241df79c0ac19d56'
    assert hashlib.md5(binary_codeword.stimulus_set['0']).hexdigest() == '6c80326cd9325da0523d48e78b0192eb'

    embedding = VectorEmbeddings(vocabulary=random_sequencer.tokens, rng=rng)
    scalar = embedding.real_valued(dim=1, distribution=rng.uniform, parameters={'low': -1., 'high': 1.})
    # assert np.isclose(scalar.stimulus_set['0'][0], 0.7086996383573585)
    assert np.isclose(scalar.stimulus_set['0'][0], 0.7781713260204495)


def test_grammar():
    rng_seed = 123
    rng = np.random.default_rng(rng_seed)

    # A.a) Elman sequences
    Elman = {
        'label': 'Elman',
        'alphabet': ['b', 'a', 'd', 'i', 'g', 'u'],
        'states': ['b', 'a', 'd', 'i1', 'i2', 'g', 'u1', 'u2', 'u3'],
        'start_states': ['b', 'd', 'g'],
        'terminal_states': ['a', 'i2', 'u3'],
        'transitions': [
            ('b', 'a', 1.),
            ('d', 'i1', 1.),
            ('i1', 'i2', 1.),
            ('g', 'u1', 1.),
            ('u1', 'u2', 1.),
            ('u2', 'u3', 1.)],
        'rng': rng  # this is important for reproducibility!
    }
    elman = ArtificialGrammar(**Elman)
    elman.generate_string_set(n_strings=10, str_range=(1, 10), correct_strings=True)

    elman_seq = elman.generate_sequence()
    rnd_elman_seq = elman.generate_random_sequence(T=5)
    # print(elman_seq[:5])
    # print(rnd_elman_seq)
    assert elman_seq[:5] == ['b', 'a', 'g', 'u', 'u']
    assert rnd_elman_seq == ['i', 'd', 'u', 'd', 'b']

# test_sequences()
# test_embeddings()
# test_grammar()
