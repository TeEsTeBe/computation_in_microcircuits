import numpy as np
from tasks.symbolic.embeddings import VectorEmbeddings

from tasks.symbolic import SymbolicSequencer, NonAdjacentDependencies, NaturalLanguage

# ######################################################################################################################
random_sequencer = SymbolicSequencer(label='random', set_size=10)

nad_sequencer = NonAdjacentDependencies(vocabulary_size=10, filler_variability=1, dependency_length=4)
nad_sequencer.generate_stringset(set_length=100, generator=False, violations=None)

lang_sequencer = NaturalLanguage
# ######
sequencer = random_sequencer
sample_sequences = sequencer.draw_subsequences(n_subseq=2, length_range=(5, 10))

# ######################################################################################################################
# Embeddings
# ######################################################################################################################
one_hot = VectorEmbeddings(vocabulary=sequencer.tokens).one_hot()

binary_codeword = VectorEmbeddings(vocabulary=sequencer.tokens).binary_codeword(dim=100, density=0.1)

scalar = VectorEmbeddings(vocabulary=sequencer.tokens).real_valued(dim=1, distribution=np.random.uniform, parameters={
    'low': -1., 'high': 1.})

random_vector1 = VectorEmbeddings(vocabulary=sequencer.tokens).real_valued(dim=None, distribution=np.random.uniform, parameters={
     'low': 0., 'high': 1.})

random_vector2 = VectorEmbeddings(vocabulary=sequencer.tokens).real_valued(dim=1000, distribution=np.random.normal,
                                                                    parameters={'loc': 0., 'scale': 0.5})
# word2vec = VectorEmbeddings() / Word2Vec()
# semantic_embbeding = VectorEmbeddings() / SemanticEmbedding()
# co_occurrence_embedding = VectorEmbedding() / CoOccurrence(seq)
# (...)

emb = [one_hot, binary_codeword, scalar, random_vector1, random_vector2]
for seq in sample_sequences:
    print("\nSample String: {0!s}".format(seq))
    for idx, enc in enumerate(emb):
        enc.plot_sample(seq, save=False)

# ######################################################################################################################
# TODO - plot discrete input space, recurrence, pairwise distances

from tools.visualization.plotting import plot_discrete_space

for seq in sample_sequences:
    print("\nSample String: {0!s}".format(seq))
    for idx, enc in enumerate(emb):
        plot_discrete_space(enc.draw_stimulus_sequence(seq, as_array=True), data_label=enc.label, label_seq=seq,
                            metric='PCA', colormap='jet', display=True, save=False)

# for idx, (seq_obj, seq) in enumerate(zip(emb, sample_sequences)):
#     fig = pl.figure()
#     fig.suptitle(seq_obj.name)
#     for nn in range(6):
#         ax = fig.add_subplot(3, 3, nn+1)
#         recurrence_plot(sequence_to_int(seq), dt=nn, ax=ax, display=False, save=False)