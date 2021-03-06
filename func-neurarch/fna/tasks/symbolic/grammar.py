import bisect

import numpy as np
from pandas import DataFrame

# internal imports
from fna.tools import utils
from fna.tasks.symbolic.sequences import SymbolicSequencer
from fna.tools.utils.operations import empty

logger = utils.logger.get_logger(__name__)


class ArtificialGrammar(SymbolicSequencer):
    """
    Symbolic sequence with transition rules specified by a directed graph, along with corresponding transition
    probabilities. The sequences are generated by traversing the graph. This formalism can be used to define any
    sequence, provided the states and transitions can be specified (each symbol as a state)

    See also, e.g.:
    ----------
    [1] - Pothos, E. M. (2010). An entropy model for artificial grammar learning. Frontiers in Psychology, 1(June), 16.
    [2] - Duarte, R., Seriès, P., & Morrison, A. (2014). Self-Organized Artificial Grammar Learning in Spiking Neural
    Networks. In Proceedings of the 36th Annual Conference of the Cognitive Science Society (pp. 427–432).
    [3] - Bollt, E. M., & Jones, M. a. (2000). The Complexity of Artificial Grammars. Nonlinear Dynamics Psychology and
    Life Sciences, 4(2), 153–168.
    [4] - Pothos, E. M. (2007). Theories of artificial grammar learning. Psychological Bulletin, 133(2),
    227–244.
    """
    def __init__(self, label, states, alphabet, transitions, start_states, terminal_states, eos=None, rng=None):
        """
        :param states: a list containing the states of the grammar, i.e., the nodes of the directed graph.
        For simplicity and robustness of implementation, the nodes should correspond to the individual
        (unique) symbols that constitute the language
        :param alphabet: a list containing the unique symbols that ought to be represented.
        In many cases, these symbols are different from the states, given that the same symbol may correspond to
        several different nodes, in which case, the different states for the same symbol are numbered (see examples)
        :param transitions: list of tuples with the structure (source_state, target_state, transition probability).
        e.g. [('a','b',0.1), ('a','c',0.3)]
        :param start_states: a list containing the possible start symbols
        :param terminal_states: a list containing the terminal symbols
        :param eos: symbol to use as eos marker
        :param rng: random number generator state object (optional). Either None or a numpy.random.default_rng object,
        or an object with the same interface
        """
        self.name = label
        self.states = states
        if rng is None:
            self.rng = np.random.default_rng()
            logger.warning("ArtificialGrammar sequences will not be reproducible!")
        else:
            self.rng = rng
        symbols = list(np.sort(alphabet))
        SymbolicSequencer.__init__(self, label=label, set_size=len(symbols), alphabet=symbols, eos=eos, rng=rng)

        self.transitions = transitions
        self.start_symbols = start_states
        self.terminal_symbols = terminal_states
        self.string_set = []
        self.grammaticality = []
        self.input_sequence = []
        self.print()

    def print(self):
        """
        Displays all the relevant information.
        """
        logger.info('***************************************************************************')
        if self.name is not None:
            logger.info(('Generative mechanism: %s' % self.name))
        logger.info('Unique states: {0}'.format(self.states))
        logger.info('Alphabet: {0}'.format(self.tokens))
        _ = self.transition_table(display=True)

    def transition_table(self, display=True):
        """
        Creates a look-up table with all allowed transitions and their probabilities
        """
        logger.info('Transition table: ')
        table = np.zeros((len(self.states), len(self.states)))
        for i, ii in enumerate(self.states):
            for j, jj in enumerate(self.states):
                tmp = [v[2] for v in self.transitions if (v[0] == jj and v[1] == ii)]
                if tmp:
                    table[i, j] = tmp[0]
        if display:
            df = DataFrame(table, columns=self.states, index=self.states)
            print(df)

        return table

    def validate(self):
        """
        Verify that all the start and end states are members of the state set and if the alphabet is
        different from the states
        """
        assert set(self.start_symbols).issubset(set(self.states)), 'start_symbols not in states'
        assert set(self.terminal_symbols).issubset(set(self.states)), 'end_symbols not in states'

        if not set(self.tokens).issubset(set(self.states)):
            test_var = set(self.tokens).difference(set(self.states))
            logger.debug(test_var)
            return test_var
        else:
            return True

    def generate_string(self):
        """
        Generate a single grammatical string by traversing the grammar
        :return: string as a list of symbols
        """
        string = [self.start_symbols[self.rng.integers(len(self.start_symbols))]]

        while string[-1] not in self.terminal_symbols:
            allowed_transitions = [x for i, x in enumerate(self.transitions) if x[0] == string[-1]]

            assert (len(allowed_transitions) >= 0), 'No allowed transitions from node {0}'.format(string[-1])
            if len(allowed_transitions) == 1:
                string.append(allowed_transitions[0][1])
            else:
                cumPr = np.cumsum([n[2] for n in allowed_transitions])
                idx = bisect.bisect_right(cumPr, self.rng.random())
                string.append(allowed_transitions[idx][1])
        return string

    def generate_string_set(self, n_strings, str_range=(1, 1000), correct_strings=True, nongramm_fraction=0.):
        """
        Generates a grammatical string set containing nStrings strings...
        :param n_strings: [int] Total number of strings to generate
        :param str_range: [tuple] string length, should be specified as interval [min_len, max_len]
        :param correct_strings: [bool] in most cases the eos marker is added as part of the string, if True,
        the eos markers are removed from the strings
        :param nongramm_fraction: [float] fraction of nongrammatical items to be introduced in the dataset
        :returns string_set: iterator of generated strings
        """
        if self.name is not None:
            logger.info('Generating {0!s} strings, according to {1!s} rules'.format(n_strings, self.name))

        while len(self.string_set) < n_strings:
            string = self.generate_string()
            if str_range:
                while not (str_range[0] < len(string) < str_range[1]):
                    string = self.generate_string()
                self.string_set.append(string)
            else:
                self.string_set.append(string)
            self.grammaticality.append(True)
        if nongramm_fraction:
            chng_str = [(idxx, self.string_set[idxx])
                        for idxx in list(self.rng.choice(len(self.string_set),
                                                          size=int(nongramm_fraction*len(self.string_set))))]
            strings = [x[1] for x in chng_str]
            idxs = [x[0] for x in chng_str]
            for idx_str, string in enumerate(self.string_set):
                if string in strings and idx_str in idxs:
                    # change one random element through a forbidden transition
                    rnd_token_idx = self.rng.integers(len(string))
                    allowed_transitions = [x for x in self.transitions if x[0]==string[rnd_token_idx]]
                    new_token = self.rng.choice(self.states)
                    new_string = []
                    for idx_tk, tk in enumerate(string):
                        if idx_tk == rnd_token_idx:
                            new_string.append(new_token)
                        else:
                            new_string.append(tk)
                    self.grammaticality[idx_str] = False
                    self.string_set[idx_str] = new_string
                    while new_token in [x[1] for x in allowed_transitions]:
                        rnd_token_idx = self.rng.integers(len(string))
                        allowed_transitions = [x for x in self.transitions if x[0]==string[rnd_token_idx]]
                        new_token = self.rng.choice(self.states)
        if correct_strings:
            self.correct_strings(terminal_symbol=self.eos)

        logger.info('Example String: {0}'.format(''.join(self.string_set[self.rng.integers(len(self.start_symbols))])))

    def correct_strings(self, terminal_symbol=None):
        """
        Remove terminal symbol (eos), which is typically provided as one the states in the grammar and correct the
        states when they are indexed (e.g. when the same symbol corresponds to different states of the grammar,
        as it is specified here)
        This function removes the eos symbol from the string set and corrects the states and terminals to match the
        alphabet.
        Note that this is not strictly necessary, since the complete sequences will contain the eos marker as a
        string separator
        :return:
        """
        if terminal_symbol is None:
            terminal_symbol = self.eos
        for idx_str, strr in enumerate(self.string_set):
            if terminal_symbol is not None and terminal_symbol in strr:
                strr.remove(terminal_symbol)
            self.string_set[idx_str] = [sym[0] if len(sym) > 1 else sym for sym in strr]

        if terminal_symbol in self.tokens:
            self.tokens.remove(terminal_symbol)

    def generate_sequence(self, n_strings=100):
        """
        Generate a sequence by concatenating the string set.
        Strings will be separated by the eos marker (if any).
        If no string set has been generated, this function will generate a default string set
        :param n_strings: total number of strings (if not already generated)
        :return: input sequence
        """
        if empty(self.string_set):
            logger.warning("String set is empty. Generating strings with default parameters.")
            self.generate_string_set(n_strings)
        self.input_sequence = self._concatenate_stringset(self.string_set, separator=self.eos)
        return self.input_sequence
