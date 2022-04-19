import numpy as np

# internal imports
from fna.tasks.symbolic.sequences import SymbolicSequencer
from fna.tools.utils.operations import empty, flatten_list
from fna.tools import utils

logger = utils.logger.get_logger(__name__)


class NonAdjacentDependencies(SymbolicSequencer):
    """
    Generate input and output sequences for simple tasks involving non-adjacent dependencies.
    Each input string consists of a frame of the type "A (n*X) B", where A and B are the dependents and X is the
    filler. n represents the span of the dependency (how many intervening items)

    See also, e.g.:
    ----------
    [1] - Fitz, H. (2011). A Liquid-State Model of Variability Effects in Learning Nonadjacent Dependencies.
    CogSci 2011 Proceedings, 897–902.
    [2] - Lazar, A. (2009). SORN: a Self-organizing Recurrent Neural Network. Frontiers in Computational
    Neuroscience, 3(October), 23.
    [3] - Onnis, ...
    """
    def __init__(self, vocabulary_size, filler_variability, dependency_length, rng=None):
        """
        NonAdjacentDependencies instance constructor
        :param vocabulary_size: number of unique "words" or "frames". Each new frame is a novel dependency,
        e.g. for vocabulary_size=3, "A1 ... B1", "A2 ... B2", "A3 ... B3".
        :param filler_variability: size of filler set
        :param dependency_length: number of filler items per word (can be an integer or a RNG)
        :return:
        """
        if rng is None:
            self.rng = np.random.default_rng()
            logger.warning("NonAdjacentDependencies sequences will not be reproducible!")
        else:
            self.rng = rng

        symbols = list(flatten_list([['A{0!s}'.format(i), 'B{0!s}'.format(i), 'X{0!s}'.format(j)] for i in range(
            vocabulary_size) for j in range(filler_variability)]))
        SymbolicSequencer.__init__(self, label='nAD', set_size=len(list(np.unique(symbols))),
                                   alphabet=list(np.unique(symbols)), eos='.', rng=self.rng)
        self.start_symbols = ['A{0!s}'.format(i) for i in range(vocabulary_size)]
        self.terminal_symbols = ['B{0!s}'.format(i) for i in range(vocabulary_size)]
        self.dependency_length = dependency_length
        self.fillers = ['X{0!s}'.format(i) for i in range(filler_variability)]
        self.accepted_patterns = ['A'+str(i)+'B'+str(i) for i in range(len(symbols))]
        self.string_set = []
        # self.input_sequence = []

    def generate_string(self, generator=False):
        """
        Generate an individual string, `word' or frame
        :param generator: retrieve the string as a generator (True) or a list (False)
        :return: string
        """
        start = self.rng.choice(self.start_symbols)
        idx = start[-1]
        dependent = [n for n in self.terminal_symbols if n[-1] == idx][0]
        if not empty(self.fillers):
            pattern = [start, self.dependency_length * [self.rng.choice(self.fillers)], dependent]
        else:
            pattern = [start, dependent]
        if generator:
            return flatten_list(pattern)
        else:
            return list(flatten_list(pattern))

    # TODO match signature of base method!
    def generate_stringset(self, set_length, generator=False, violations=None):
        """
        Generate the complete stringset for the experiment
        :param set_length: total number of strings to generate (or allowed, if limited)
        :param generator: retrieve the strings as generators (True) or lists (False)
        :param violations: (int=n_strings) introduce syntactic violations in n_strings in the set
        :return:
        """
        logger.info('Generating {0!s} strings, according to {1!s} rules...'.format(set_length, self.name))
        for t in range(set_length):
            string = self.generate_string(generator)
            self.string_set.append(string)
            if violations is not None:
                string[-1] = self.rng.choice(self.terminal_symbols)

    # @staticmethod
    def _non_matching_frames(self, string_set, terminals, violations=0.5):
        """
        Generate strings that violate the dependency
        """
        ng_strings = self.rng.permutation(len(string_set))[:int(violations*len(string_set))]

        for string in string_set[ng_strings]:
            match_token = string[-1]  # TODO can be removed?
            string[-1] = self.rng.choice(terminals)

    def generate_sequence(self):
        """
        Generate a complete sequence by concatenating the strings, separated by the eos marker
        :return: input sequence
        """
        assert not empty(self.string_set), "String set is empty, generate it first"
        return self._concatenate_stringset(self.string_set, separator=self.eos)

