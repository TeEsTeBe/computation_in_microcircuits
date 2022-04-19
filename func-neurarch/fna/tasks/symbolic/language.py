import numpy as np
from more_itertools import locate
from collections import Counter

# internal imports
from fna.tasks.symbolic.sequences import SymbolicSequencer
from fna.tools import check_dependency
from fna.tools import utils

has_nltk = check_dependency('nltk')
if has_nltk:
    import nltk

logger = utils.logger.get_logger(__name__)


# TODO work in progress..
class NaturalLanguage(SymbolicSequencer):
    """
    Natural language parser to use NL datasets and apply the standard preprocesing and generation for use within the
    scope of fna
    Note: relies on nltk, if the library is not installed, this class cannot be used

    See, e.g.:
    ____
    [1] - Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
    """

    def __init__(self, label, text_data=None, character_level=False, rng=None):
        """
        Default constructor
        :param text_data:
        """
        assert has_nltk, "NaturalLanguage requires nltk."
        if rng is None:
            self.rng = np.random.default_rng()
            logger.warning("NaturalLanguage sequences will not be reproducible!")
        else:
            self.rng = rng

        def load_corpus(corpus_name):
            """
            Import one of the available NL corpora
            :return:
            """
            assert hasattr(nltk.corpus, corpus_name), "{0!s} corpus not available in nltk. Load corpus manually if " \
                                                      "necessary".format(corpus_name)
            logger.info("Loading {0!s} corpus from nltk".format(corpus_name))
            corpus = getattr(nltk.corpus, corpus_name)
            string_set = list(corpus.sents())
            sequence = list(corpus.words())
            eos = Counter([x[-1] for x in string_set]).most_common()[0][0]
            return corpus, string_set, sequence, eos

        def load_from_txt(input_text, eos=None):
            """
            Load corpus from a list of sentences
            :param input_text:
            :return:
            """
            string_set = [nltk.word_tokenize(x) for x in input_text]
            # remove spaces and split sentences
            if eos is None:
                eos = Counter([x[-1] for x in string_set]).most_common()[0][0]
            sequence = self._concatenate_stringset(string_set, separator=eos)
            return string_set, sequence, eos

        if text_data is None and hasattr(nltk.corpus, label):
            self.corpus, self.string_set, self.sequence, eos = load_corpus(label)
        else:
            assert isinstance(text_data, list) and isinstance(text_data[0], str), "Provide input text as a list of " \
                                                                                  "sentences"
            self.string_set, self.sequence, eos = load_from_txt(text_data)
        tokens = list(np.unique(self.sequence))

        self._sentence_level_parse()
        if character_level:
            self._character_level_parse()

        SymbolicSequencer.__init__(self, label=label, set_size=len(tokens), alphabet=tokens, eos=eos)
        self.print()

    def print(self):
        """
        Displays all the relevant information.
        """
        logger.info("***************************************************************************")
        if self.name is not None:
            logger.info(("Natural Language: {0!s}".format(self.name)))
        logger.info("Number of strings: {0!s}".format(len(self.string_set)))
        logger.info("Vocabulary size: {0!s}".format(len(self.tokens)))
        logger.info("Sequence length: {0!s}".format(len(self.sequence)))
        logger.info("Most common: {0!s}".format(Counter(self.sequence).most_common(10)))

    @staticmethod
    def pos_tags(text):
        """
        Attribute part-of-speech tags to the input text
        :param text: list of tokens (words)
        :return:
        """
        return nltk.pos_tag(text)

    def _character_level_parse(self):
        """
        Consider single characters as the tokens and words as the strings
        :return:
        """
        self.sequence = [char for x in self.sequence for char in x]
        self.tokens = list(np.unique(self.sequence))

    def _sentence_level_parse(self, eos='.'):
        """
        After loading a text corpus / input, parse it to populate the string set
        :return:
        """
        index_pos = list(locate(self.sequence, lambda a: a == eos))
        self.string_set = [self.sequence[i+1:index_pos[idx+1]] for idx, i in enumerate(index_pos[:-1])]

    def replace_rare(self, min_occurrences=10, replace_token='UNK'):
        """
        Replace rare tokens with 'UNK'
        :return:
        """
        pass

    def word2id(self):
        """
        Assign a unique numeric id to each word
        :return:
        """
        pass

    def display_parse_tree(self):
        """
        Plot and display example sentences parse trees
        :return:
        """
        assert hasattr(self, "corpus"), "method only applies to annotated corpora"
        example_sent = self.rng.choice(self.corpus.parsed_sents())
        example_sent.draw()

