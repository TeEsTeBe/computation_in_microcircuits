from tools.visualization.plotting import recurrence_plot, plot_from_dict, plot_histogram
from tools.visualization.helper import label_bars
import matplotlib.pyplot as pl
import numpy as np
import itertools


import fna.tools.utils.logger
logger = fna.tools.utils.logger.get_logger(__name__)
logger.propagate = False

# Parameters
n_strings = 10000
rnd_seq_length = 10000

# ################################################################################
# A) Random sequences
# ################################################################################
from tasks.symbolic import SymbolicSequencer




# ################################################################################
# B) Sequences with predefined transition table
# ################################################################################
from tasks.symbolic import ArtificialGrammar


# ################################################################################
# Analog tasks
# ################################################################################
from tasks.symbolic.sequences import SymbolicSequencer
from tasks.symbolic.embeddings import VectorEmbeddings

T = 1000

sequencer = SymbolicSequencer(label='random sequence', set_size=T)
sequence = sequencer.generate_random_sequence(T=T)

embedding = VectorEmbeddings(vocabulary=sequencer.tokens).real_valued(dim=1, distribution=np.random.uniform, parameters={
    'low': 0., 'high': 0.5})

stim_seq = embedding.draw_stimulus_sequence(sequence, as_array=True)

# a) NARMA
from tasks.analog import narma

#narma_params = {'alpha': 0.3, 'beta': 0.05, 'gamma': 1.5, 'eps': 0.1}
target = narma(stim_seq[0], n=15)


import matplotlib.pyplot as plt
plt.plot(stim_seq[0][:100])
plt.plot(target[0][:100])



# Temporal XOR - discrete
from tasks.symbolic.sequences import SymbolicSequencer
from tasks.symbolic.embeddings import VectorEmbeddings

sequencer = SymbolicSequencer(label='random sequence', alphabet=['A', 'B'], eos='#')
sequencer.generate_stringset(set_length=10, length_range=(2, 2))
sequence = sequencer.generate_sequence()

indices = [i for i, x in enumerate(sequence) if x == sequencer.eos]
output = [None for _ in sequence]
accept = [False for _ in sequence]
for idx in indices:
    accept[idx] = True
    if sequence[idx-2] != sequence[idx-1]:
        output[idx] = 1.
    elif sequence[idx-2] == sequence[idx-1]:
        output[idx] = -1.
target_outputs = [{'label': 'XOR', 'output': output, 'accept': accept}]

embedding = VectorEmbeddings(vocabulary=sequencer.tokens).one_hot()
embedding.stimulus_set.update({'#': np.zeros_like(embedding.stimulus_set['A']),
                               'B': -1*embedding.stimulus_set['B']})
embedding.plot_sample(sequence)

# Temporal XOR - continuouse


sig = embedding.unfold(to_signal=True, **{'duration': 250., 'amplitude': 1.})
signal = sig.draw_stimulus_sequence(sequence, onset_time=0.1, continuous=True, intervals=100.)[0].as_array().sum(axis=0)



sig.plot_sample(sequence, continuous=True)



# ##################################
# continuous integration
from tasks.symbolic.sequences import SymbolicSequencer
from tasks.symbolic.embeddings import VectorEmbeddings
import matplotlib.pyplot as plt

T = 100

# discrete random
sequencer = SymbolicSequencer(label='random sequence', set_size=T)
sequence = sequencer.generate_random_sequence(T=T)
embedding = VectorEmbeddings(vocabulary=sequencer.tokens).real_valued(dim=1, distribution=np.random.uniform, parameters={
    'low': 0., 'high': 0.5})

stim_seq = embedding.draw_stimulus_sequence(sequence, as_array=True)
target_outputs = [{'label': 'integrator', 'output': np.cumsum(stim_seq), 'accept': [True for _ in sequence]}]

plt.plot(stim_seq[0], 'o-k')
plt.plot(target_outputs[0]['output'], 'r-')
plt.show()


# continuous, step
sequencer = SymbolicSequencer(label='random sequence', set_size=T)

# continuous +/- steps

