from fna.tools import utils
from fna.tasks.symbolic import ArtificialGrammar, NonAdjacentDependencies, NaturalLanguage

logger = utils.logger.get_logger(__name__)
logger.propagate = False

# Parameters
n_strings = 10000
rnd_seq_length = 10000


def test_Elman():
    # ################################################################################
    # A) Sequences with predefined transition table
    # ################################################################################

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
    }
    elman = ArtificialGrammar(**Elman)
    elman.generate_string_set(n_strings=n_strings, str_range=(1, 10), correct_strings=True)

    # generate input sequence
    elman_seq = elman.generate_sequence()
    # rnd_elman_seq = elman.generate_random_sequence(T=rnd_seq_length)

    # generate targets
    target_outputs = elman.generate_default_outputs(elman_seq)

    # A.b) Reber grammar
    FSG_G1 = {
        'label': 'Reber Grammar',
        'states': ['M1', 'V2', 'T1', 'V1', 'T2', 'R1', 'X1', 'X2', 'R2', 'M2', '#'],
        'alphabet': ['M', 'V', 'T', 'R', 'X', '#'],
        'start_states': ['M1', 'V2'],
        'terminal_states': ['#'],
        'transitions': [
                ('M1', 'T1', 0.5),
                ('M1', 'V1', 0.5),
                ('T1', 'T1', 0.5),
                ('T1', 'V1', 0.5),
                ('V1', 'T2', 1. / 3.),
                ('V1', 'R1', 1. / 3.),
                ('V1', '#', 1. / 3.),
                ('T2', '#', 1.),
                ('R1', 'X1', 0.5),
                ('R1', 'X2', 0.5),
                ('V2', 'X1', 0.5),
                ('V2', 'X2', 0.5),
                ('X1', 'T1', 0.5),
                ('X1', 'V1', 0.5),
                ('X2', 'R2', 1. / 3.),
                ('X2', 'M2', 1. / 3.),
                ('X2', '#', 1. / 3.),
                ('R2', 'R2', 1. / 3.),
                ('R2', 'M2', 1. / 3.),
                ('R2', '#', 1. / 3.),
                ('M2', '#', 1.)],
        'eos': '#'
        }
    # build artificial grammar
    reber = ArtificialGrammar(**FSG_G1)

    # generate string set
    reber.generate_string_set(n_strings=n_strings, str_range=(1, 10000), correct_strings=True, nongramm_fraction=0.)

    # generate input sequence
    reber_seq = reber.generate_sequence()
    # rnd_reber_seq = reber.generate_random_sequence(T=rnd_seq_length)

    # generate default targets
    # TODO task-specific objects should override generate_default_outputs to include their own..
    reber.generate_default_outputs(reber_seq)

    # TODO call different stored grammar dictionaries
    # TODO embedded grammars


def test_non_adjacent():
    # ################################################################################
    # B) Sequences with non-adjacent dependencies
    # ################################################################################
    # non-adjacent dependencies

    vocabulary_size = 10
    filler_variability = 1
    dependency_length = 5

    nad = NonAdjacentDependencies(vocabulary_size, filler_variability, dependency_length)
    nad.generate_stringset(set_length=n_strings, generator=False, violations=None)

    # generate/analyse input sequence
    nad_seq = nad.generate_sequence()
    nad.count(nad_seq)

    # generate targets
    nad.generate_default_outputs(nad_seq)

# TODO rename when ready
def tst_natural_language():
    # ################################################################################
    # C) Natural Language Sentences
    # ################################################################################
    # TODO @Renato this needs to be checked
    # nltk.download()

    # C.a) "Manual" sentences
    process_sentences = ["""At eight o'clock on Thursday morning Arthur didn't feel very good."""]
    lang = NaturalLanguage(label="example sentence", text_data=process_sentences, character_level=False)

    # C.b) From corpus
    corpus_label = "treebank"
    lang_corp = NaturalLanguage(label=corpus_label, text_data=None, character_level=False)
    # lang_corp.display_parse_tree()

    # C.c) character-level processing
    lang_char = NaturalLanguage(label="sentence character", text_data=process_sentences, character_level=True)

# # ##################################################################################
# # Analyse and compare
# # ##################################################################################
# seq_objects = [elman, reber, nad, lang, lang_corp, lang_char]
# sequences = [elman.generate_sequence(), reber.generate_sequence(), nad.generate_sequence(), lang.sequence, lang_corp.sequence, lang_char.sequence]
# random_sequences = [x.generate_random_sequence(T=len(y)) for x, y in zip(seq_objects, sequences)]
#
# # frequency distributions and set sizes
# vocabulary_sizes = [len(x.tokens) for x in seq_objects]
# string_lengths = [x.string_length() for x in seq_objects]
#
# frequencies = [x.count(s, as_freq=True) for x, s in zip(seq_objects, sequences)]
# freq_rnd = [x.count(s, as_freq=True) for x, s in zip(seq_objects, random_sequences)]
# freq_common = [x.most_common(s, 100, as_freq=True) for x, s in zip(seq_objects, sequences)]
#
# print("Vocabulary sizes: ")
# [print(" {0!s} [{1!s}]".format(x, y.name)) for x, y in zip(vocabulary_sizes, seq_objects)]
#
# fig2 = pl.figure(figsize=(18, 12))
# axes = [fig2.add_subplot(len(freq_common), 1, x+1) for x in range(len(freq_common))]
# [plot_from_dict(x, ax) for x, ax in zip(freq_common, axes)]
# [ax.set_title("Top-10 [{0!s}]".format(x.name)) for x, ax in zip(seq_objects, axes)]
# pl.show()
#
#
# fig, axes = pl.subplots(1, len(string_lengths), figsize=(30, 3))
# # axes = [fig.add_subplot(1, len(string_lengths), x+1) for x in range(len(string_lengths))]
# [plot_histogram(x, n_bins=100, mark_mean=True, ax=ax, display=False) for x, ax in zip(string_lengths, axes)]
# [ax.set_title("String length [{0!s}]: \nMean={1!s}".format(x.name, np.mean(string_lengths[idx]))) for (idx, x), ax in zip(enumerate(seq_objects), axes)]
# pl.show()
#
# # string complexity
# string_complexity = [x.string_set_complexity() for x in seq_objects]
#
# from tools.visualization.plotting import plot_matrix, plot_histogram
# from tools.utils.operations import empty
#
# for idx, x in enumerate(string_complexity):
#     if not empty(x['edit_distance']):
#         fig, axes = pl.subplots(2, 2, figsize=(10, 10))
#         fig.suptitle("String complexity [{0!s}]".format(seq_objects[idx].name))
#         axes = list(itertools.chain(*axes))
#         ax_idx = 0
#         for k, v in x.items():
#             plot_matrix(v, ax=axes[ax_idx], display=False)
#             plot_histogram(v[v!=0.], 100, mark_mean=True, display=False, ax=axes[ax_idx+2])
#             axes[ax_idx].set_title(k)
#             ax_idx += 1
#     pl.show()
#
# # np.mean(string_complexity['edit_distance'][string_complexity['edit_distance']!=0.])
#
# # measure sequence complexity
# compressibility = [x.compressibility(s) for x, s in zip(seq_objects, sequences)]
# comp_rnd = [x.compressibility(s) for x, s in zip(seq_objects, random_sequences)]
#
# entropy = [x.entropy(s) for x, s in zip(seq_objects, sequences)]
# ent_rnd = [x.entropy(s) for x, s in zip(seq_objects, random_sequences)]
#
# te = [x.topographical_entropy(s) for x, s in zip(seq_objects, sequences) if not isinstance(x, NaturalLanguage)]
# te_rnd = [x.topographical_entropy(s) for x, s in zip(seq_objects, random_sequences) if not isinstance(x,
#                                                                                                      NaturalLanguage)]