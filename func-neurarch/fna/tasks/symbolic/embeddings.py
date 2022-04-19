import os
from abc import ABC, abstractmethod
import pickle as pkl
import numpy as np
import matplotlib.pyplot as pl
from scipy.signal import fftconvolve

from fna.tools.signals import make_simple_kernel, pad_array
from fna.encoders.generators import generate_spike_template
from fna.tools.signals.analog import AnalogSignal, AnalogSignalList
from fna.tools.utils.operations import empty, is_binary, determine_decimal_digits
from fna.tools.visualization.plotting import plot_matrix
from fna.tools.visualization.helper import fig_output, check_axis
from fna.tools import utils
from fna.tools import signals
from fna.tools.utils import data_handling as io


logger = utils.logger.get_logger(__name__)


class Embeddings(ABC):
    """
    Abstract Base Class
    """
    def __init__(self, vocabulary, eos=None, rng=None):
        """
        Default constructor
        :param vocabulary: list of unique tokens to encode
        """
        # logger.info("Creating symbolic embeddings")
        self.vocabulary = vocabulary
        # self.dimensions = len(vocabulary)
        self.embedding_dimensions = len(vocabulary)
        self.stimulus_set = None
        self.label = None
        self.eos = eos
        self.rng = rng

    def __iter__(self):
        """
        Iterate stimulus set
        :return:
        """
        for k, v in self.stimulus_set.items():
            if isinstance(v, list):
                for item in v:
                    yield item
            else:
                yield v

    def __len__(self):
        """
        Determine the total size of the stimulus set
        :return:
        """
        return len([x for x in self])

    @abstractmethod
    def draw_stimulus_sequence(self, seq, verbose=True):
        """
        Parse a symbolic sequence and return the corresponding encoded stimuli
        :param seq: ordered list of symbols
        :param verbose: [bool] - log info
        :return: dim x T array
        """
        if verbose:
            logger.info("Generating stimulus sequence: {0!s} symbols".format(len(seq)))
        return self._sequence_iterator(seq)

    def _sequence_iterator(self, seq):
        """
        Returns a generator
        :return:
        """
        for token in seq:
            if isinstance(token, np.ndarray):
                print(token)
            tk = self.stimulus_set[token]
            if isinstance(tk, list):
                encoded = tk[self.rng.choice(len(tk))]
            else:
                encoded = tk
            yield encoded

    @abstractmethod
    def plot_sample(self, seq, ax=None):
        """
        Plots an example sequence embedding in the provided axes
        :return:
        """
        pass

    def print(self):
        """
        Log the relevant information regarding the stimulus embedding
        :return:
        """
        logger.info("- {0!s} symbols encoded as {1!s}".format(len(self.vocabulary), self.label))

    def generate_default_outputs(self, input_sequence, max_memory=10, max_chunk=5, max_prediction=5):
        """
        For every symbolic sequence task, there is a set of standard tasks that apply:
        - stimulus classification / recognition / invariance
        - stimulus memory
        - context-dependent predictions
        This function generates all the target outputs for these default tasks in a non-symbolic manner,
        i.e. the ouput sequences are sequences of token embeddings
        :param input_sequence:
        :param max_memory:
        :param max_chunk:
        :param max_prediction:
        :return:
        """
        pass

    def save(self, label=None):
        """
        Save entire Embedding object
        """
        try:
            if label is None:
                filename = "{}_{}_{}.pkl".format(io.filename_prefixes['embedding'], self.label, io.data_label)
            else:
                filename = "{}_{}_{}.pkl".format(io.filename_prefixes['embedding'], self.label, label)
            with open(os.path.join(io.paths['inputs'], filename), 'wb') as f:
                pkl.dump(self, f)
        except Exception as e:
            logger.warning("Could not save Embedder {}, storage paths not set?".format(self.label))


class VectorEmbeddings(Embeddings):
    """
    Convert symbolic tokens into vectors, according to one of the available embedding methods:
    - one-hot encoding
    - binary-codeword
    - n-dimensional random values
    - word2vec
    - lexical encoding
    - frequency co-occurence statistics
    """
    def __init__(self, vocabulary, rng=None):
        if rng is None:
            self.rng = np.random.default_rng()
            logger.warning("VectorEmbeddings results will not be reproducible!")
        else:
            self.rng = rng
        super().__init__(vocabulary, rng=self.rng)

    def one_hot(self):
        """
        One-hot encoding
        :return:
        """
        self.label = 'one-hot'
        one_hot = np.eye(len(self.vocabulary))
        self.stimulus_set = {k: v for i, (k, v) in enumerate(zip(self.vocabulary, one_hot))}
        self.embedding_dimensions = len(self.vocabulary)
        return self

    def binary_codeword(self, dim=None, density=0.1):
        """
        Each token is represented by a k-dimensional vector with a fixed number of active bits
        :param dim: vector size
        :param density: fraction of active bits
        :return:
        """
        self.label = 'binary-codeword'
        if dim is None:
            dim = len(self.vocabulary)
        self.embedding_dimensions = dim
        n_active_bits = int(dim * density)
        data_set = {}
        for k in self.vocabulary:
            vec = np.zeros(dim)
            # ensure that no index is chosen multiple times
            vec[self.rng.choice(np.arange(0, dim, 1).astype(int), size=n_active_bits, replace=False)] = 1.
            data_set.update({k: vec})
        if self.eos is not None:
            data_set.update({self.eos: np.zeros(dim)})
        self.stimulus_set = data_set
        return self

    def real_valued(self, dim=None, distribution=None, parameters=None):
        """
        Each token is represented by a k-dimensional vector of random real numbers drawn from a given probability
        distribution
        :param dim: dimensionality of the target vector
        :param distribution: probability distribution function (random value sample, function)
        :param parameters: parameters to pass to the distribution function
        :return:
        """
        if dim == 1:
            self.label = 'scalar'
        else:
            self.label = 'real-valued'
        if dim is None:
            dim = len(self.vocabulary)
        if distribution is None:
            distribution = self.rng.uniform
        if parameters is None:
            parameters = {'low': 0., 'high': 1.}

        parameters.update({'size': dim})
        self.stimulus_set = {k: distribution(**parameters) for k in self.vocabulary}
        if self.eos is not None:
            self.stimulus_set.update({self.eos: np.zeros(dim)})
        self.embedding_dimensions = dim
        self.label += '-{0!s}'.format(distribution.__name__)
        return self

    def unfold(self, to_spikes=False, to_signal=False, verbose=True, **params):
        """
        Unfold a vector embedding in time, turning it into a continuous signal or spike pattern.
        :return DynamicEmbedding object:
        """
        return DynamicEmbeddings(self.vocabulary, self.eos, rng=self.rng).unfold_discrete_set(
            stimulus_set=self.stimulus_set, to_spikes=to_spikes, to_signal=to_signal, verbose=verbose, **params)

    def draw_stimulus_sequence(self, seq, as_array=False, verbose=True):
        """
        Parse a symbolic sequence and return the corresponding encoded stimuli
        :param seq: ordered list of symbols
        :param as_array: return the sequence as a full array (True) or as an iterator (False)
        :param verbose: [bool] - log info
        :return: dim x T array
        """
        if verbose:
            logger.info("Generating stimulus sequence: {0!s} symbols".format(len(seq)))
        out = self._sequence_iterator(seq)
        if as_array:
            enc_sequence = [enc for enc in out]
            return np.array(enc_sequence).T
        else:
            return out

    def plot_sample(self, seq, ax=None, save=False, display=True):
        """
        Plots an example sequence embedding in the provided axes
        :param seq: sequence to plot
        :return:
        """
        stim_seq = self.draw_stimulus_sequence(seq, as_array=True)

        if ax is not None:
            fig, ax = check_axis(ax)
        else:
            fig, ax = pl.subplots(1, 1)

        if len(seq) < 100:
            fig.suptitle(r'Example string: {0!s} [{1!s}]'.format(' '.join(seq), self.label))
        if self.embedding_dimensions == 1:
            fig.set_size_inches(10, 2, forward=True)
            x = np.arange(0, len(seq))
            ax.stem(stim_seq[0])
            ax.plot(stim_seq[0], '--', lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels(seq)
            fig_output(fig, save=save, display=display)
        else:
            fig.set_size_inches(8, 8, forward=True)
            plot_matrix(stim_seq, seq, ax, save, display)

    # def save(self):
    #     super().save()


class DynamicEmbeddings(Embeddings):
    """
    Symbolic embedding with temporally extended patterns
    """
    def __init__(self, vocabulary, eos=None, rng=None):
        """
        Default constructor
        :param vocabulary: list of unique tokens to encode
        """
        if rng is None:
            self.rng = np.random.default_rng()
            logger.warning("DynamicEmbeddings results will not be reproducible!")
        else:
            self.rng = rng
        super().__init__(vocabulary, eos, rng)
        self._is_spks = False
        self._jitter = None
        self._is_signal = False
        self.dt = 0.1
        self.total_set_size = len(self.vocabulary)
        self._is_iterator = False

    def frozen_noise(self, n_processes=10, pattern_duration=50., rate=10., resolution=0.1, jitter=None):
        """
        Each token is represented by a spatiotemporal pattern spike pattern, an instance of frozen Poissonian noise
        :param n_processes: number of Poisson processes that constitute the pattern
        :param pattern_duration: duration of each embedding (either a fixed value for all stimuli, or a list
        specifying different durations for each token
        :param rate: mean rate of Poisson process (either a single, fixed value for all stimuli, or a list/array
        specifying different rate amplitudes for different tokens)
        :param resolution: pattern dt
        :param jitter: None or tuple with (jitter_value, compensate=True/False), where compensate refers to whether
        the pattern duration should be adjusted so that each spike pattern retains the mean rate
        :param rng: Random number generator. None or a numpy.random.RandomState object
        :return self:
        """
        self.label = 'frozen-noise'
        self._is_spks = True
        self.stimulus_set = {}
        self._jitter = jitter
        self.dt = resolution
        self.total_set_size *= n_processes
        self.embedding_dimensions = n_processes

        if self.total_set_size < 10000: # TODO - if dataset is too large, generate stimuli online
            logger.info("Populating Stimulus Set... ")

        for n, token in enumerate(self.vocabulary):
            duration = pattern_duration[n] if isinstance(pattern_duration, list) and len(pattern_duration) == \
                                              len(self.vocabulary) else pattern_duration
            rt = rate[n] if isinstance(rate, list) and len(rate) == len(self.vocabulary) else rate

            if jitter is not None and jitter[1]:
                duration += (jitter[0] * 2)

            self.stimulus_set.update({token: self._generate_spk_template(n_processes, rates=rt, duration=duration,
                                                                         resolution=resolution, rng=self.rng)})
        return self

    @staticmethod
    def _generate_spk_template(n_processes, rates, duration, resolution, rng):
        """
        Generate a spatiotemporal spike template as a collection of independent Poisson processes
        :param n_processes: number of processes that constitute the pattern
        :param rates: single value or list/array of rate values
        :param duration: duration of pattern [ms]
        :param resolution: dt
        :param rng: None or a numpy.random.RandomState object
        :return spattern: SpikeList object with unique spatiotemporal spike pattern
        """
        if isinstance(rates, float) or isinstance(rates, int):
            spattern = generate_spike_template(n_neurons=n_processes, rate=rates, duration=duration,
                                               resolution=resolution, rng=rng)
        else:
            spattern = signals.spikes.SpikeList([], [], t_start=0., t_stop=duration)
            for n_neuron in range(n_processes):
                rt = rates[n_neuron] + 0.00000001
                spk_pattern = generate_spike_template(n_neurons=1, rate=rt, duration=duration,
                                                      resolution=resolution, rng=rng)
                if empty(spk_pattern.spiketrains):
                    spk_train = signals.spikes.SpikeTrain([], t_start=resolution, t_stop=duration)
                else:
                    spk_train = spk_pattern.spiketrains[0]
                spattern.append(n_neuron, spk_train)
        return spattern

    def unfold_discrete_set(self, stimulus_set, seq=None, to_spikes=False, to_signal=False, verbose=True, **params):
        """
        Convert a discrete stimulus set (vector or scalar representations) to a continuous signal
        :param stimulus_set: {token: embedding} dictionary
        :param seq: sequence of tokens to encode
        :param to_spikes: [bool] - convert input vectors to spike patterns (each vector dimension corresponds to 1
        Poisson process firing at a rate proportional to the value in that dimension)
        :param to_signal: [bool] - convert input vectors to a continuous signal
        :param verbose: [bool] - log info
        :param params: encoding-specific parameter dictionaries (see documentation)
        :return self:
        """
        self.label = 'continuous'
        self.stimulus_set = {}
        if to_spikes:
            self._is_spks = True
            self._jitter = None
            self.label += ' spike unfolding'
        if to_signal:
            self._is_signal = True
            self.label += ' signal unfolding'

        # determine total size (single stimulus vector dimensionality * number of unique stimuli)
        if isinstance(list(stimulus_set.values())[0], list):
            self.total_set_size = np.sum(list({k: len(v) for k, v in stimulus_set.items()}.values())) * len(list(
                stimulus_set.values())[0][0])
            self.embedding_dimensions = len(list(stimulus_set.values())[0][0])
        else:
            self.total_set_size *= len(list(stimulus_set.values())[0])
            self.embedding_dimensions = len(list(stimulus_set.values())[0])

        if self.total_set_size < 1000000: # TODO - verify threshold
            if verbose:
                logger.info("Populating Stimulus Set... ")
            for token, representations in stimulus_set.items():
                if isinstance(representations, list):
                    if to_signal:
                        self.stimulus_set.update(
                            {token: [self._vec2signal(token, vec, **params) for idx, vec in representations]})
                    if to_spikes:
                        self.stimulus_set.update(
                            {token: [self._vec2spks(token, vec, **params) for vec in representations]})
                else:
                    if to_signal:
                        self.stimulus_set.update({token: self._vec2signal(token, representations, **params)})
                    if to_spikes:
                        self.stimulus_set.update({token: self._vec2spks(token, representations, **params)})
        else:
            assert seq is not None, "stimulus sequence required to generate Dynamic stimuli online"
            if verbose:
                logger.info("Converting sequence to DynamicEmbedding iterator: ")
            self._is_iterator = True
            self.stimulus_set = self._unfold_discrete_sequence(stimulus_set, seq, to_spikes=to_spikes,
                                                             to_signal=to_signal, **params)
        # self.print()
        return self

    def _unfold_discrete_sequence(self, stimulus_set, seq, to_spikes=False, to_signal=True, **params):
        """
        Convert a discrete stimulus sequence (vector or scalar representations) to a continuous signal. This is
        applicable when the stimulus_set or dimensionality are too large to pre-generate all dynamic embeddings. In
        this case, DynamicEmbeddings are generated online for each token individually
        :param stimulus_set: {token: embedding} dictionary
        :param seq: sequence of tokens to encode
        :param to_spikes: [bool] - convert input vectors to spike patterns (each vector dimension corresponds to 1
        Poisson process firing at a rate proportional to the value in that dimension)
        :param to_signal: [bool] - convert input vectors to a continuous signal
        :param params: encoding-specific parameter dictionaries (see documentation)
        :return self:
        """
        self._is_iterator = True
        for token in seq:
            tk = stimulus_set[token]
            if isinstance(tk, list):
                encode_tk = tk[self.rng.choice(len(tk))]
            else:
                encode_tk = tk

            if to_signal:
                encoded_stim = self._vec2signal(tk, encode_tk, **params)
            elif to_spikes:
                encoded_stim = self._vec2spks(tk, encode_tk, **params)
            else:
                raise TypeError("Either to_spikes or to_signal must be True")
            yield encoded_stim

    def _vec2signal(self, token, input_vector, amplitude, duration, dt=0.1, onset=0., kernel=('box', {})):
        """
        Converts a vector into a continuous signal by kernel convolution
        :param token: vocabulary token for which the signal should be unfolded
        :param input_vector:
        :param amplitude: [float, list, dict] signal amplitude: scalar, list of values (==! n_tokens), or distribution
        :param duration: [float, list, dict] signal duration: scalar, list of values (==! n_tokens), or distribution
        :param dt:
        :param onset:
        :param kernel:
        :return:
        """
        self.dt = dt

        # case where amplitudes and/or durations are drawn from distribution
        if isinstance(amplitude, dict):
            amp = amplitude['dist'](**amplitude['params'])
        elif isinstance(amplitude, list):
            assert len(amplitude) == len(self.vocabulary), "Nr of token amplitudes does not match vocabulary length!"
            token_idx = self.vocabulary.index(token)
            amp = amplitude[token_idx]
        else:
            amp = amplitude

        if isinstance(duration, dict):
            rounding_precision = determine_decimal_digits(self.dt)
            duration = np.round(duration['dist'](**duration['params']), rounding_precision)
        elif isinstance(duration, list):
            assert len(duration) == len(self.vocabulary), "Number of token durations does not match vocabulary length!"
            token_idx = self.vocabulary.index(token)
            duration = duration[token_idx]

        offset = onset + duration
        time_axis = np.arange(onset, offset, dt)
        dim = input_vector.shape[0]
        self.embedding_dimensions = dim

        if not isinstance(kernel[1], dict):
            raise TypeError("Incorrect kernel parameters")

        # generate empty AnalogSignalList
        signal_array = AnalogSignalList([], [], times=time_axis, dt=dt, t_start=min(time_axis),
                                        t_stop=max(time_axis) + dt, dims=dim)
        # signals = []
        for i in range(dim):
            signal_ = np.zeros(len(time_axis))
            if input_vector[i]:
                mid_point = len(time_axis) / 2.
                signal_[int(mid_point)] = 1.

            if not is_binary(input_vector):
                amplitude = input_vector[i] * amp
            else:
                amplitude = amp

            kern = make_simple_kernel(kernel[0], width=duration, height=amplitude, resolution=dt, normalize=False,
                                      **kernel[1])
            signal = AnalogSignal(fftconvolve(signal_, kern, mode='same'), dt=dt, t_start=onset, t_stop=offset)
            signal_array.append(i, signal)
        return signal_array, time_axis

    def _vec2spks(self, token, input_vector, duration=50., rate_scale=1., dt=0.1, jitter=None, shift_range=None):
        """
        Converts a vector into a spatiotemporal pattern of spike trains, where each dimension corresponds to a
        Poisson process firing at a rate of v_i * rate_scale
        :param token: vocabulary token for which the signal should be unfolded
        :param input_vector: n-dimensional vector to encode
        :param duration: [float, list, dict] temporal extent of spike pattern;
            Can be scalar, list of values (==! n_tokens), or distribution
        :param rate_scale: [float, list, dict] rate amplitude constant scale value
            Can be scalar, list of values (==! n_tokens), or distribution
        :param dt: time resolution [ms]
        :param onset: start time [ms]
        :param jitter: None or tuple with (jitter_value, compensate=True/False), where compensate refers to whether
        the pattern duration should be adjusted so that each spike pattern retains the mean rate
        :param shift_range: If there are negative values in the input vectors, shift them by adding a fixed value
        :param rng: Random number generator
        :return:
        """
        self.dt = dt

        # case where amplitudes and/or durations are drawn from distribution
        if isinstance(rate_scale, dict):
            rate_scale = rate_scale['dist'](**rate_scale['params'])
        elif isinstance(rate_scale, list):
            assert len(rate_scale) == len(self.vocabulary), "Nr of token amplitudes does not match vocabulary length!"
            token_idx = self.vocabulary.index(token)
            rate_scale = rate_scale[token_idx]

        if isinstance(duration, dict):
            rounding_precision = determine_decimal_digits(self.dt)
            duration = np.round(duration['dist'](**duration['params']), rounding_precision)
        elif isinstance(duration, list):
            assert len(duration) == len(self.vocabulary), "Number of token durations does not match vocabulary length!"
            token_idx = self.vocabulary.index(token)
            duration = duration[token_idx]

        if jitter is not None:
            self._jitter = jitter
            if jitter[1]:
                duration += (jitter[0] * 2)

        dim = input_vector.shape[0]
        if np.any(input_vector < 0.) and shift_range is None:
            raise TypeError("Spike encoding cannot accept negative input values..")
        elif np.any(input_vector < 0.) and shift_range is not None:
            input_vector += shift_range
            if np.any(input_vector < 0.):
                raise TypeError("Spike encoding cannot accept negative input values (increase shift)..")

        self.embedding_dimensions = dim
        return self._generate_spk_template(n_processes=dim, rates=input_vector*rate_scale, duration=duration,
                                           resolution=dt, rng=self.rng)

    def load_signal(self, array=None, stimulus_set=None, dim=None):
        """

        :param dim:
        :return:
        """
        self._is_signal = True
        self.label = 'continuous-signal'
        self.stimulus_set = {}
        pass

    def load_spikes(self, stimulus_set):
        """

        :param stimulus_set:
        :return:
        """
        self._is_spks = True
        pass

    def time_warp(self, warp_ratio=1.):
        """
        Maintain the stimulus properties, but rescale the time axis
        :return:
        """
        pass

    def _sequence_iterator(self, seq, onset_time=0., intervals=None):
        """
        Returns a generator
        :return:
        """
        tok = None
        for idx, token in enumerate(seq):
            if self._is_iterator and self._is_signal:
                encoded, _ = next(self.stimulus_set)
            elif self._is_iterator and self._is_spks:
                encoded = next(self.stimulus_set)
            elif not self._is_iterator and self._is_signal:
                tk, _ = self.stimulus_set[token]
                if isinstance(tk, list):
                    encoded = tk[self.rng.choice(len(tk))]
                else:
                    encoded = tk
            else:
                tk = self.stimulus_set[token]
                if isinstance(tk, list):
                    encoded = tk[self.rng.choice(len(tk))]
                else:
                    encoded = tk

            # setup interval value for this step
            if intervals is not None:
                if isinstance(intervals, int) or isinstance(intervals, float):
                    interval = intervals
                elif isinstance(intervals, list):
                    interval = intervals[self.vocabulary.index(token)]
                elif isinstance(intervals, dict):
                    rounding_precision = determine_decimal_digits(encoded.t_stop)
                    interval = np.round(np.floor(intervals['dist'](**intervals['params'])), rounding_precision)
                else:
                    raise TypeError("Inter-Stimulus-Intervals should be provided as a single value or a "
                                    "dictionary with keys ['dist', 'params']")
            else:
                interval = 0

            if idx == 0:
                if self._is_signal:
                    tok = encoded.time_offset(onset_time)
                    # zero-pad
                    zero_pad = interval / tok.dt
                    tok = tok.zero_pad(n_steps=zero_pad)
                elif self._is_spks:
                    tok = encoded.copy()
                    tok.time_offset(onset_time)
            else:
                offset_time = tok.t_stop

                if self._is_signal:
                    # offset
                    tok = encoded.time_offset(offset_time)

                    if intervals is not None:
                        # zero-pad
                        zero_pad = interval / tok.dt
                        tok = tok.zero_pad(n_steps=zero_pad)
                elif self._is_spks:
                    offset_time += interval
                    tok = encoded.copy()
                    tok.time_offset(offset_time)
            yield tok, (tok.t_start, tok.t_stop, interval)

    def concatenate_sequence(self, stim_seq, onset):#, intervals):
        """

        :param seq:
        :return:
        """
        # logger.info("Concatenating stimulus sequence")

        tokens = []
        stim_timing_info = []
        for stim_data in stim_seq:
            tokens.append(stim_data[0])
            stim_timing_info.append(stim_data[1])

        stim_seq_array = None
        dt = None
        dim = None
        next = None

        if len(tokens) == 1:
            next = tokens[0]
            dt = next.dt
            if self._is_signal:
                stim_seq_array = next.copy().as_array()
                dim = next.dimensions
        else:
            for idx in range(len(tokens) - 1):
                next = tokens[idx + 1]
                tk = tokens[idx]
                if self._is_spks:
                    if idx == 0:
                        stim_seq = tk.copy()
                    stim_seq.merge(next)
                elif self._is_signal:
                    if idx == 0:
                        stim_seq_array = tk.copy().as_array()
                        stim_seq_array = np.append(stim_seq_array, next.as_array(), axis=1)
                        dt = tk.dt
                        dim = tk.dimensions
                    else:
                        stim_seq_array = np.append(stim_seq_array, next.as_array(), axis=1)

        if self._is_signal:
            time_axis = np.arange(onset, next.t_stop, dt)
            stim_seq = AnalogSignalList([], [], times=time_axis, dt=dt, t_start=min(time_axis),
                                        t_stop=max(time_axis) + dt, dims=dim)
            for idx in range(stim_seq_array.shape[0]):
                signal = AnalogSignal(stim_seq_array[idx, :], dt=dt, t_start=onset, t_stop=next.t_stop)
                stim_seq.append(idx, signal)
        return stim_seq, stim_timing_info

    def draw_stimulus_sequence(self, seq, onset_time=0., continuous=False, intervals=None, verbose=True):
        """
        Parse a symbolic sequence and return the corresponding encoded stimuli
        :param seq: ordered list of symbols
        :param onset_time: float
        :param continuous: retrieve a continuous signal, with the concatenated dynamic embeddings or yield the
        individual embeddings
        :param intervals: inter-stimulus intervals (either a single value, or a random variate generator)
        :param verbose: bool
        :return:
        """
        if verbose:
            logger.info("Generating stimulus sequence: {0!s} symbols".format(len(seq)))
        if continuous:
            out = self._sequence_iterator(seq, onset_time=onset_time, intervals=intervals)
            tokens, stim_info = self.concatenate_sequence(out, onset=onset_time)# , intervals=intervals)
        else:
            tokens = self._sequence_iterator(seq, onset_time, intervals)
            stim_info = None
        return tokens, stim_info

    def plot_sample(self, seq, stim_seq=None, continuous=False, intervals=None, display=False, save=False):
        """
        Plots an example sequence embedding in the provided axes
        :param seq: sequence to plot
        :return:
        """
        if stim_seq is None:
            stim_seq, _ = self.draw_stimulus_sequence(seq, onset_time=0., continuous=continuous, intervals=intervals)

        if self._is_spks and not continuous:
            fig = pl.figure(figsize=(15, 4))
            fig.suptitle(r'Example string: {0!s} [{1!s}]'.format(' '.join(seq), self.label))
            for iid, (sym, enc) in enumerate(zip(seq, stim_seq)):
                ax = fig.add_subplot(1, len(seq), iid + 1)
                enc[0].raster_plot(ax=ax, display=False, save=save)
                ax.set_xlabel('Time [ms]')
                ax.set_title(sym)
        elif self._is_spks and continuous:
            fig = pl.figure(figsize=(15, 4))
            fig.suptitle(r'Example string: {0!s} [{1!s}]'.format(' '.join(seq), self.label))
            ax = fig.add_subplot(1, 1, 1)
            stim_seq.raster_plot(ax=ax, display=display, save=save)

        elif self._is_signal and continuous:
            fig = pl.figure(figsize=(8, 8))
            fig.suptitle(r'Example string: {0!s} [{1!s}]'.format(' '.join(seq), self.label))
            stim_seq.plot(as_array=True, fig=fig, display=display, save=save)
            if self.embedding_dimensions < 500:
                fig2 = pl.figure(figsize=(10, 8))
                fig2.suptitle(r'Example string: {0!s} [{1!s}]'.format(' '.join(seq), self.label))
                stim_seq.plot(as_array=False, fig=fig2, display=display, save=save)

        elif self._is_signal and not continuous:
            fig = pl.figure(figsize=(10, 4))
            fig.suptitle(r'Example string: {0!s} [{1!s}]'.format(' '.join(seq), self.label))
            for iid, (sym, enc) in enumerate(zip(seq, stim_seq)):
                ax = fig.add_subplot(1, len(seq), iid + 1)
                enc[0].plot(as_array=True, fig=fig, ax=ax, display=False, save=save)
                ax.set_title(sym)
                ax.set_xlabel('Time (steps)')

    # def save(self):
    #     super().save()


# class Word2VecEmbedding(Embeddings):
#     """
#     Use the Word2Vec [1] algorithm to compute vector representations of natural language words.
#     Note: Requires TensorFlow
#
#     See:
#     [1] - Mikolov, Tomas et al. "Efficient Estimation of Word Representations
#     in Vector Space.", 2013. https://arxiv.org/pdf/1301.3781.pdf
#     (adapted from TensorFlow examples by Aymeric Damien https://github.com/aymericdamien/TensorFlow-Examples/)
#     """
#     pass
#
#
# class CoOccurrenceEmbedding(VectorEmbeddings):
#     """
#     Use stimulus co-occurence statistics to determine the appropriate embeddings
#     """
#     pass
