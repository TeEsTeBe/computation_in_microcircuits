import copy
import os
import pickle as pkl
import numpy as np
from pandas import DataFrame
from collections import Counter
from math import log
from gzip import compress
from tqdm import tqdm
import nltk

from fna.tools import utils
from fna.tools.utils.operations import empty
from fna.tools.utils import data_handling as io

logger = utils.logger.get_logger(__name__)


def chunk(seq, n):
    """
    Chunk the sequence into all of its constituent n-grams and determine their frequency in the set
    :param seq: [list or generator] full symbolic sequence (should be as long as possible, particularly for large n)
    :param n: [int] order parameter (chunk "length")
    :return:
    """
    all_ngrams = [''.join(list(seq)[ii:ii + n]) for ii in range(len(list(seq)))]
    un_ngrams = np.unique(all_ngrams).tolist()

    # check if all possibilities are represented (n-gram frequency):
    count = []
    for ii, nn in enumerate(un_ngrams):
        if len(nn) < n:
            un_ngrams.pop(ii)
        else:
            count.append(len(np.where(np.array(all_ngrams) == nn)[0]))
    return all_ngrams, un_ngrams


def chunk_transitions(seq, n, display=True):
    """
    Determine the transition matrix for n-gram sequences
    :param seq: [list or generator] full symbolic sequence (should be as long as possible, particularly for large n)
    :param n: [int] order parameter (chunk "length")
    :param display: [bool] show transition table
    :return M: [array]
    """
    all_ngrams, un_ngrams = chunk(seq, n)
    M = np.zeros((len(un_ngrams), len(un_ngrams)))
    nGrams = np.array(all_ngrams)
    for ii, i in enumerate(un_ngrams):
        for jj, j in enumerate(un_ngrams):
            M[ii, jj] = float(any(nGrams[np.where(nGrams == i)[0][:-1] + 1] == j))
    if display:
        df = DataFrame(M, columns=un_ngrams, index=un_ngrams)
        print(df)
    return M


def hamming_distance(seq1, seq2):
    """
    Calculate hamming distance between 2 sequences
    :param seq1:
    :param seq2:
    """
    return sum(map(str.__ne__, str(seq1), str(seq2)))


def edit_distance(seq1, seq2):
    """
    Calculate edit distance between 2 sequences. Requires the editdistance package
    :param seq1:
    :param seq2:
    """
    return nltk.edit_distance(seq1, seq2)


# ######################################################################################################################
class SymbolicSequencer(object):
    """
    Build patterned symbolic sequences.
    Contains the generic constructors to implement structured symbolic sequences
    """
    # TODO - load_sequence, plots
    def __init__(self, label, set_size=None, alphabet=None, eos=None, rng=None):
        """
        :param label: [string] label of the current task
        :param set_size: [int] number of unique symbols
        :param alphabet: [list] unique tokens
        :param eos: [string] end-of-sentence marker
        :param rng: [numpy.random] seeded random number generator
        """
        logger.info("Generating symbolic sequencer")
        self.tokens = []
        self.name = label
        if alphabet is not None:
            self.tokens = alphabet
        else:
            self.tokens = [str(i) for i in range(set_size)]
        self.eos = eos
        self.string_set = []
        self.input_sequence = []

        if rng is None:
            self.rng = np.random.default_rng()
            logger.warning("SymbolicSequencer sequences will not be reproducible!")
        else:
            self.rng = rng

    def generate_stringset(self, set_length, length_range, generator=False, verbose=True):
        """
        Generate the complete stringset for the experiment
        :param set_length: total number of strings to generate (or allowed, if limited)
        :param generator: retrieve the strings as generators (True) or lists (False)
        :param violations: (int=n_strings) introduce syntactic violations in n_strings in the set
        :return:
        """
        if verbose:
            logger.info('Generating {0!s} strings...'.format(set_length, self.name))
        self.string_set = self.draw_subsequences(n_subseq=set_length, seq=None, length_range=length_range)

    def generate_sequence(self):
        """
        Generate a complete sequence by concatenating the strings, separated by the eos marker
        :return: input sequence
        """
        assert not empty(self.string_set), "String set is empty, generate it first"
        self.input_sequence = self._concatenate_stringset(self.string_set, separator=self.eos)
        return self.input_sequence

    @staticmethod
    def _concatenate_stringset(string_set, separator=''):
        """
        Concatenates a string set (list of strings) into a list of symbols, placing the separator symbol between strings
        :param string_set: list of strings
        :param separator: string symbol separating different strings
        :return:
        """
        str_set = copy.deepcopy(string_set)
        if separator is not None:
            [n.insert(0, separator) for idx, n in enumerate(list(str_set)) if idx != 0]
        symbol_seq = np.concatenate(list(str_set)).tolist()
        return symbol_seq

    def generate_random_sequence(self, T=0, verbose=True):
        """
        Randomly draw items from the alphabet
        :return: input sequence
        """
        if verbose:
            logger.info('Generating a random sequence of length {0!s}, '
                        'from a set of {1!s} symbols'.format(T, len(self.tokens)))
        if T == len(self.tokens):  # draw without repetition (each symbol only once)
            replace = False
        else:
            replace = True
        return list(self.rng.choice(self.tokens, T, replace=replace))

    def draw_subsequences(self, n_subseq, seq=None, length_range=(5, 10), verbose=False):
        """
        Draw sample sub-sequences from a main sequence or from the stringset
        :param n_subseq: number of subsequences to draw
        :param seq: sequence to draw from (if no stringset is available)
        :param length_range: tuple (min, max) or None (will be the
        :return:
        """
        if not empty(self.string_set):
            idx = self.rng.integers(0, len(self.string_set), n_subseq)
            out_str = [self.string_set[ii] for ii in idx]
        else:
            if seq is None:
                # generate a small random sequence
                seq = self.generate_random_sequence(T=int(n_subseq*max(length_range)), verbose=verbose)
            # take random chunks
            idx = self.rng.integers(0, len(seq)-n_subseq, n_subseq)
            if length_range[0] == length_range[1]:
                lengths = [length_range[0] for _ in range(n_subseq)]
            else:
                lengths = self.rng.integers(length_range[0], length_range[1], n_subseq)
            out_str = [seq[ii:ii + xx] for ii, xx in zip(idx, lengths)]
        return out_str

    @staticmethod
    def count(sequence, as_freq=False):
        """
        Computes the frequency of each item in the sequence
        :param sequence: list of tokens
        :param as_freq: bool - return total counts (False) of frequencies
        :return dict: {item: frequency (count/length)}
        """
        if as_freq:
            return {k: v/len(sequence) for k, v in Counter(sequence).items()}
        else:
            return dict(Counter(sequence))

    @staticmethod
    def most_common(sequence, n, as_freq=False):
        """
        Return the frequency of the n most common tokens in the sequence
        :param sequence: list of tokens
        :param n: n most common
        :param as_freq: bool - return total counts (False) of frequencies
        :return:
        """
        ctr = Counter(sequence).most_common(n)
        if as_freq:
            return {k: v/len(sequence) for (k, v) in ctr}
        else:
            return dict(ctr)

    def entropy(self, sequence):
        """
        Calculate the entropy of a sequence.
        :param sequence: full symbolic sequence
        """
        cnt = [self.count(sequence)[i] for i in np.unique(sequence)]
        d = sum(cnt)
        ent = []
        for i in [float(i) / d for i in cnt]:
            # round corner case that would cause math domain error
            if i == 0:
                i = 1
            ent.append(i * log(i, 2))
        return -1 * sum(ent)

    @staticmethod
    def topographical_entropy(seq, max_lift=10):
        """
        Compute TE of a grammar, using the "lift" method.
        See, e.g.:
        -------------
        [1] - Bollt and Jones (2000)
        [2] - Shiff and Katan (2014)
        :return:
        """
        logger.info("Computing topographical entropy..")
        if len(seq) <= 1000:
            logger.warning("Sequence is too short, entropy estimates may not be reliable.")
        TE = []
        nn, top_ent = 0, 0
        for nn in range(max_lift):
            M = chunk_transitions(seq, nn + 1, display=False)
            eigs = np.linalg.eigvals(M)

            max_eig = np.real(np.max(eigs))
            TE.append(np.log(max_eig))
            logger.info("Lift: {0!s}; Entropy: {1!s}".format(nn + 1, TE[-1]))
            if (len(TE) > 1) and (np.round(np.diff(TE), 1)[-1] == 0.):
                top_ent = TE[-2]
                break
            else:
                top_ent = TE[-1]
        return nn, top_ent, TE

    @staticmethod
    def compressibility(sequence):
        """
        Determine the compressibility ratio of the input sequence
        :param sequence: symbolic sequence (list of symbols)
        :return:
        """
        return len(compress(''.join(sequence).encode())) / len(sequence)

    @staticmethod
    def _memory_targets(seq, n_mem):
        """
        Create the outputs for sequence memorization tasks
        :param seq:
        :param n_mem:
        :return output, accept:
        """
        output = seq[:-(n_mem + 1)]
        while len(output) < len(seq):
            output.insert(0, None)
        accept = [x is not None for x in output]  # list(np.array(output).astype(bool))
        return output, accept

    @staticmethod
    def _prediction_targets(seq, n_pred):
        """
        Create the outputs for sequence prediction tasks
        :return:
        """
        output = seq[(n_pred + 1):]
        while len(output) < len(seq):
            output.append(None)
        accept = [x is not None for x in output]  # list(np.array(output).astype(bool))
        return output, accept

    def _chunk_targets(self, chunk_seq, n_chunk, max_chunk_memory=0, max_chunk_prediction=0):
        """
        Create the outputs for sequence chunking tasks
        :param chunk_seq: sequence of n_chunck chunks
        :param n_chunk: chunk length
        :param max_chunk_memory: chunk memorization (max delay, if None set to 0)
        :param max_chunk_prediction: chunk prediction (max_pred, if None set to 0)
        :return targets: dictionary
        """
        targets = []
        output = chunk_seq[:-n_chunk]
        for idx, n in enumerate(chunk_seq):
            if idx < n_chunk:
                output.insert(idx, None)
        accept = [x is not None for x in output]

        if n_chunk > 0:
            targets.append({
                'label': '{0!s}-chunk recognition'.format(n_chunk + 1),
                'output': output,
                'accept': accept})
            for n_mem in range(max_chunk_memory):
                output, accept = self._memory_targets(output, n_mem)
                targets.append({
                    'label': '{0!s}-chunk {1!s}-memory'.format(n_chunk + 1, n_mem + 1),
                    'output': output,
                    'accept': accept})
            for n_pred in range(max_chunk_prediction):
                output, accept = self._prediction_targets(output, n_pred)
                targets.append({
                    'label': '{0!s}-chunk {1!s}-prediction'.format(n_chunk + 1, n_pred + 1),
                    'output': output,
                    'accept': accept})
        return targets

    def generate_default_outputs(self, input_sequence, max_memory=0, max_chunk=0, max_prediction=0,
                                 chunk_memory=False, chunk_prediction=False, verbose=False):
        """
        For every symbolic sequence task, there is a set of standard tasks that apply:
        - stimulus classification / recognition / invariance
        - stimulus memory
        - stimulus predictions
        - chunk classification / memory / prediction
        This function generates all the target outputs for these default tasks (for task-specific outputs,
        see the corresponding class

        :return:
        """
        targets = [{
            'label': 'classification',
            'output': input_sequence,
            'accept': [True for _ in input_sequence]}]
        for n_mem in range(max_memory):
            output, accept = self._memory_targets(input_sequence, n_mem)
            targets.append({
                'label': '{0!s}-step memory'.format(n_mem+1),
                'output': output,
                'accept': accept})
        for n_chunk in range(max_chunk):
            chunks, elements = chunk(input_sequence, n=n_chunk+1)
            if chunk_memory:
                ch_mem = max_memory
            else:
                ch_mem = 0
            if chunk_prediction:
                ch_pred = max_prediction
            else:
                ch_pred = 0
            for x in self._chunk_targets(chunks, n_chunk, max_chunk_memory=ch_mem, max_chunk_prediction=ch_pred):
                targets.append(x)
        for n_pred in range(max_prediction):
            output, accept = self._prediction_targets(input_sequence, n_pred)
            targets.append({
                'label': '{0!s}-step prediction'.format(n_pred+1),
                'output': output,
                'accept': accept})

        if verbose:
            logger.info("Generated {0} decoder outputs: {1}".format(len(targets), str([x['label'] for x in targets])))

        # TODO shouldn't the targets be somehow stored in the class object?
        return targets

    def string_length(self, string_set=None):
        """
        Compute the length of all strings in the string_set
        :param string_set: list of strings or None
        :return:
        """
        if string_set is None and not empty(self.string_set):
            string_set = self.string_set
        elif string_set is None:
            string_set = []

        return [len(string) for string in string_set]

    def string_set_complexity(self, string_set=None):
        """
        Evaluate the complexity of the string set as the pairwise distance between strings
        :param string_set:
        :return:
        """
        if string_set is None and not empty(self.string_set):
            string_set = self.string_set
        elif string_set is None:
            string_set = []
        logger.info("Evaluating string set complexity...")
        edit_dists = np.zeros((len(string_set), len(string_set)))
        hamming_dists = np.zeros((len(string_set), len(string_set)))

        iu1 = np.tril_indices_from(edit_dists)
        for i, j in tqdm(zip(iu1[0], iu1[1]), desc="Calculating pairwise distances: ", total=len(iu1[0])):
            edit_dists[i, j] = edit_distance(string_set[i], string_set[j])
            hamming_dists[i, j] = hamming_distance(i, j)

        return {'edit_distance': edit_dists, 'hamming_distance': hamming_dists}

    def save(self, label=None):
        try:
            if label is None:
                filename = "{}_{}_{}.pkl".format(io.filename_prefixes['sequencer'], self.name, io.data_label)
            else:
                filename = "{}_{}_{}.pkl".format(io.filename_prefixes['sequencer'], self.name, label)
            with open(os.path.join(io.paths['inputs'], filename), 'wb') as f:
                pkl.dump(self, f)
        except Exception as e:
            logger.warning("Could not save Sequencer {}, storage paths not set?".format(self.name))

    # TODO plot distributions (string length, token frequency, ... string complexity), recurrence plots


