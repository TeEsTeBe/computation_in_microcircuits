# visualize stimulus set
import matplotlib.pyplot as pl
import numpy as np

from tasks.symbolic.sequences import SymbolicSequencer
from tasks.preprocessing import ImageFrontend

# random input sequence
vocabulary_size = 10
T = 1000
plot_nStrings = 3

sequencer = SymbolicSequencer(label='random sequence', set_size=vocabulary_size)
seq = sequencer.generate_random_sequence(T=T)

sample_sequences = sequencer.draw_subsequences(n_subseq=1, seq=seq, length_range=(5, 15)) # to plot

# ######################################################################################################################
# image data
image_mnist = ImageFrontend(path='../data/mnist/', label='mnist', vocabulary=sequencer.tokens)
image_cifar = ImageFrontend(path='../data/cifar-10/', label='cifar-10', vocabulary=sequencer.tokens)


# Unfold vector representations into continuous inputs
signal_pars = {
    'duration': 250., # single values or rvs
    'amplitude': 1., # single value, list of dim, or rvs
    'kernel': ('alpha', {'tau': 200.}), # (kernel_label, {parameters}).. see documentation
    'dt': 0.1 # dt
}
sig = image_mnist.unfold(to_signal=True, **signal_pars)

stim_seq = [x for x in image_mnist.draw_stimulus_sequence(sample_sequences[0], as_array=False, unfold=True)]

for seq in sample_sequences:
    print("\nSample String: {0!s}".format(seq))
    for idx, enc in enumerate([image_mnist]):
        enc.plot_sample(seq, continuous=False, display=True, save=False)
        enc.plot_sample(seq, continuous=True, display=True, save=False)






spk_encoding_pars = {
    'duration': 250.,
    'rate_scale': 50.,
    'dt': 0.1,
    'jitter': None
}
image_mnist.unfold(to_spikes=True, **spk_encoding_pars)
image_cifar.unfold(to_spikes=True, **spk_encoding_pars)

for seq in sample_sequences:
    print("\nSample String: {0!s}".format(seq))
    for idx, enc in enumerate([image_mnist, image_cifar]):
        enc.plot_sample(seq, continuous=False, intervals=100., display=True, save=False)
        enc.plot_sample(seq, continuous=True, display=True, save=False)
