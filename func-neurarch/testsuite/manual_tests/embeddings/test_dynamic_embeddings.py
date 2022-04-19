import numpy as np
from tasks.symbolic.embeddings import VectorEmbeddings, DynamicEmbeddings

from tasks.symbolic import SymbolicSequencer, NonAdjacentDependencies

# ######################################################################################################################
random_sequencer = SymbolicSequencer(label='random', set_size=10)

nad_sequencer = NonAdjacentDependencies(vocabulary_size=10, filler_variability=1, dependency_length=4)
nad_sequencer.generate_stringset(set_length=100, generator=False, violations=None)


# ######
sequencer = random_sequencer
sample_sequences = sequencer.draw_subsequences(n_subseq=2, length_range=(5, 10))

# vector embedding
one_hot = VectorEmbeddings(vocabulary=sequencer.tokens).one_hot()
binary_codeword = VectorEmbeddings(vocabulary=sequencer.tokens).binary_codeword(dim=100, density=0.3)
scalar = VectorEmbeddings(vocabulary=sequencer.tokens).real_valued(dim=1, distribution=np.random.uniform, parameters={
    'low': -1., 'high': 1.})
random_vector = VectorEmbeddings(vocabulary=sequencer.tokens).real_valued(dim=100, distribution=np.random.normal,
                                                                    parameters={'loc': 10., 'scale': 2.5})
# #######
enc = random_vector

enc.plot_sample(sample_sequences[0])
# ######################################################################################################################
# Input Signal
# ######################################################################################################################
# a) continuous signal
signal_pars = {
    'duration': 250., # single values or rvs {'dist': np.random.normal, 'params': {'loc': 100., 'scale': 50.}}
    'amplitude': 1., # single value, list of dim, or rvs {'dist': np.random.normal, 'params': {'loc': 100., 'scale': 50.}}
    'kernel': ('box', {}), # (kernel_label, {parameters}).. see documentation
    'dt': 0.1 # dt
}
sig = enc.unfold(to_signal=True, **signal_pars)


sig.plot_sample(sample_sequences[0], continuous=True, intervals=None)
sig.plot_sample(sample_sequences[0], continuous=False, intervals=None)

# ISI
sig.plot_sample(sample_sequences[0], continuous=True, intervals=100.)
sig.plot_sample(sample_sequences[0], continuous=False, intervals=100.)

sig.plot_sample(sample_sequences[0], continuous=True, intervals={'dist': np.random.normal, 'params': {'loc': 100., 'scale': 50.}})
sig.plot_sample(sample_sequences[0], continuous=False, intervals={'dist': np.random.normal, 'params': {'loc': 100., 'scale': 50.}})


# intervals = 10.
# stim_seq = sig.draw_stimulus_sequence(seq, onset_time=0., continuous=True, intervals=10.)
# stim_seq2 = sig.draw_stimulus_sequence(seq, onset_time=0., continuous=False, intervals=10.)

# for sseq in sample_sequences:
#     sig.plot_sample(sseq, continuous=True, intervals=None, display=True, save=False)
#     # sig.plot_sample(sseq, continuous=False, intervals=None, display=True, save=False)
#     sig.plot_sample(sseq, continuous=False, intervals=None, # 100., {'dist': np.random.normal, 'params': {'loc':
#                     # 100., 'scale': 50.}}
#                     display=True, save=False)
#     sig.plot_sample(sseq, continuous=False, intervals=100., display=True, save=False)


#
# from tools.visualization.helper import plot_trajectory, get_cmap
# import matplotlib.pyplot as pl
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = pl.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# stim_seq = sig.draw_stimulus_sequence(sample_sequences[0], onset_time=0., continuous=False, intervals=None)
#
# colors = get_cmap(len(sample_sequences[0]), cmap='jet')
# for idx, (token, embedding) in enumerate(zip(sample_sequences[0], stim_seq)):
#     plot_trajectory(embedding.as_array(), label=token, color=colors(idx), ax=ax, display=False)
# pl.legend()
# pl.show()


# ###############
# # b) Frozen noise
# spk_pattern = DynamicEmbeddings(vocabulary=sequencer.tokens).frozen_noise(n_processes=100, pattern_duration=200., rate=10.,
#                                                                     resolution=0.1, jitter=None, rng=None)
# # sk1 = spk_pattern.draw_stimulus_sequence(seq, onset_time=0.1, continuous=True, intervals=100.)
# # sk2 = spk_pattern.draw_stimulus_sequence(seq, onset_time=0.1, continuous=False, intervals=None)
# for sseq in sample_sequences:
#     spk_pattern.plot_sample(sseq, continuous=True, intervals=None, display=True, save=False)
#     spk_pattern.plot_sample(sseq, continuous=False, intervals=None, display=True, save=False)
#     spk_pattern.plot_sample(sseq, continuous=False, intervals=100., display=True, save=False)
#     spk_pattern.plot_sample(sseq, continuous=False, intervals={'dist': np.random.normal, 'params': {'loc': 100.,
#                                                                                                'scale': 50.}},
#                             display=True, save=False)

# ###############
# c) unfold to spikes
spk_encoding_pars = {
    'duration': 250., #{'dist': np.random.normal, 'params': {'loc': 100., 'scale': 50.}},
    'rate_scale': 10., #{'dist': np.random.normal, 'params': {'loc': 100., 'scale': 50.}},
    'dt': 0.1,
    'shift_range': 100.,
    'jitter': None
}
# spk_vec = DynamicEmbeddings(vocabulary=sequencer.tokens).unfold_discrete_set(stimulus_set=enc.stimulus_set,
#                                                                    to_spikes=True, to_signal=False, **spk_encoding_pars)
# stim_seq = spk_vec.draw_stimulus_sequence(seq, onset_time=0.1, continuous=True, intervals=100.)

spk_vec = enc.unfold(to_spikes=True, **spk_encoding_pars)

spk_vec.plot_sample(sample_sequences[0], continuous=True, intervals=None)
spk_vec.plot_sample(sample_sequences[0], continuous=False, intervals=None)

# ISI
spk_vec.plot_sample(sample_sequences[0], continuous=True, intervals=100.)
spk_vec.plot_sample(sample_sequences[0], continuous=False, intervals=100.)

spk_vec.plot_sample(sample_sequences[0], continuous=True, intervals={'dist': np.random.normal, 'params': {'loc': 100., 'scale': 50.}})
spk_vec.plot_sample(sample_sequences[0], continuous=False, intervals={'dist': np.random.normal, 'params': {'loc': 100., 'scale': 50.}})



#
# for sseq in sample_sequences:
#     spk_vec.plot_sample(sseq, continuous=True, intervals=None, display=True, save=False)
    # spk_vec.plot_sample(sseq, continuous=False, intervals=None, display=True, save=False)
    # spk_vec.plot_sample(sseq, continuous=True, intervals=100., display=True, save=False)
    # spk_vec.plot_sample(sseq, continuous=False, intervals=100., display=True, save=False)

