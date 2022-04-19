import gzip
import os.path
import shutil
import urllib.request
import pickle
from os import listdir
# import tables

import numpy as np
# from mnist.loader import MNIST
import itertools
import matplotlib.pyplot as pl
from scipy.io import wavfile
from tqdm import tqdm

from fna.tasks.symbolic.embeddings import Embeddings, DynamicEmbeddings
from fna.encoders.generators import generate_spike_template
from fna.tools.signals.spikes import SpikeList
from fna.tools.utils.operations import determine_decimal_digits
from fna.tools import utils, check_dependency

logger = utils.logger.get_logger(__name__)

has_mnist = check_dependency("mnist.loader")
has_tables = check_dependency("tables")


# ######################################################################################################################
# Auxiliary functions for image processing
def vec2im(vec):
    """
    vector encoding to original image
    :return:
    """
    try: # should work for most 2d images
        return np.array(vec).reshape(int(np.sqrt(len(vec))), int(np.sqrt(len(vec))))
    except ValueError:
        return np.array(vec).reshape(3, 32, 32).transpose([1, 2, 0])


def load_mnist(mnist_dir, concatenate=True):
    """
    Loads the MNIST dataset and stores the data in the mnist_dir directory.
    If data is already in the correct directory, it is just loaded
    :param mnist_dir: data directory
    :param concatenate: [bool] - return a single dataset or divided in train/test
    """
    assert has_mnist, "mnist required"
    from mnist.loader import MNIST

    def unpack_mnist(mnist_dir, concatenate=True):
        mnist = MNIST(mnist_dir)
        train_imgs, train_labels = mnist.load_training()
        test_imgs, test_labels = mnist.load_testing()

        if concatenate:
            imgs = train_imgs + test_imgs
            labels = list(train_labels) + list(test_labels)
            logger.info(
                "Concatenating dataset (original partition in train+test will be lost): \n\t- T={0!s}".format(len(labels)))
            data_set = {k: [] for k in np.unique(labels)}
            for k, v in zip(labels, imgs):
                data_set[k].append(np.array(v))
            return data_set
        else:
            logger.info(
                "Loading MNIST dataset: \n\t- T_train={0!s}\n\t- T_test={1!s}".format(len(train_labels),
                                                                                      len(test_labels)))
            train_set = {k: [] for k in np.unique(train_labels)}
            test_set = {k: [] for k in np.unique(test_labels)}
            for k, v in zip(train_labels, train_imgs):
                train_set[k].append(np.array(v))
            for k, v in zip(test_labels, test_imgs):
                test_set[k].append(np.array(v))
            return train_set, test_set

    if not os.path.exists(mnist_dir):
        os.makedirs(mnist_dir)

    if not os.listdir(mnist_dir):
        logger.info("MNIST target directory is empty! Downloading data...")
        mnist_url = 'http://yann.lecun.com/exdb/mnist/'
        file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz',
                      't10k-labels-idx1-ubyte.gz']
        file_urls = [mnist_url + name for name in file_names]
        for idx, url in enumerate(file_urls):
            outpath = mnist_dir + file_names[idx]
            if not os.path.isfile(outpath[:-3]):
                if not os.path.isfile(outpath):
                    urllib.request.urlretrieve(url, outpath)
                with gzip.open(outpath, 'rb') as f_in:
                    with open(outpath[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
    else:
        logger.info("MNIST target directory is not empty! Loading existing data...")
    return unpack_mnist(mnist_dir, concatenate)


def load_cifar(cifar_dir, concatenate=True):
    """
    Loads the CIFAR-10 dataset and stores the data in the cifar_dir directory.
    If data is already in the correct directory, it is just loaded
    :param cifar_dir: data directory
    :param concatenate: [bool] - return a single dataset or divided in train/test
    :return:
    """
    file_names = ['data_batch_{0}'.format(i+1) for i in range(5)]
    file_names.append('test_batch')
    file_urls = [cifar_dir + name for name in file_names]
    data = []
    for file in file_urls:
        with open(file, 'rb') as fo:
            cif_data = pickle.load(fo, encoding='bytes')
        data.append(cif_data)

    if concatenate:
        imgs = list(itertools.chain(*[x[b'data'] for x in data]))
        labels = list(itertools.chain(*[x[b'labels'] for x in data]))
        logger.info(
            "Concatenating CIFAR-10 dataset: \n\t- T={0!s}".format(len(labels)))
        data_set = {k: [] for k in np.unique(labels)}
        for k, v in zip(labels, imgs):
            data_set[k].append(np.array(v))
        return data_set

    else:
        train_data, test_data = data[:-1], data[-1]
        train_imgs = list(itertools.chain(*[x[b'data'] for x in train_data]))
        train_labels = list(itertools.chain(*[x[b'labels'] for x in train_data]))
        test_imgs = list(itertools.chain(*[x[b'data'] for x in test_data]))
        test_labels = list(itertools.chain(*[x[b'labels'] for x in test_data]))

        logger.info(
            "Loading CIFAR-10 dataset: \n\t- T_train={0!s}\n\t- T_test={1!s}".format(len(train_labels),
                                                                                  len(test_labels)))

        train_set = {k: [] for k in np.unique(train_labels)}
        test_set = {k: [] for k in np.unique(test_labels)}
        for k, v in zip(train_labels, train_imgs):
            train_set[k].append(np.array(v))
        for k, v in zip(test_labels, test_imgs):
            test_set[k].append(np.array(v))
        return train_set, test_set


# ######################################################################################################################
# Auxiliary functions for audio processing TODO
def load_audio(audio_dir):
    """
    Load audio data
    :return:
    """
    # TODO - make more generic
    audio_files = [f for f in listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))]

    labels = []
    speakers = []
    data = []
    rates = []
    durations = []
    logger.info(
        "Retrieving fsdd dataset: \n\t- T={0!s}".format(len(audio_files)))

    for a_file in audio_files:
        a_file_path = os.path.join(audio_dir, a_file)
        rate, d = wavfile.read(a_file_path)
        a_file_split = a_file[:-4].split('_')

        labels.append(a_file_split[0])
        speakers.append(a_file_split[1])
        rates.append(rate)
        durations.append(rate * 1000. / len(d))  # duration in ms
        data.append(d)

    data_set = {k: [] for k in np.unique(labels)}
    for k, v in zip(labels, data):
        data_set[k].append(np.array(v))
    return data_set, speakers, rates, durations


def load_hds(path, resolution=0.1, concatenate=True, language="english"):
    """
    Load Heidelberg digits dataset
    :param path: path to the folder storing the data (already downloaded)
    :param resolution: dt
    :param concatenate: merge train and test
    :param language: "english" or "german" or None
    :return:
    """
    assert has_tables, "tables required"
    import tables

    def load_hds_set(path, data_set="train", language="english"):
        file = "shd_{0!s}.h5".format(data_set)
        rounding_precision = determine_decimal_digits(resolution)
        fileh = tables.open_file(path+file, mode='r')
        units = fileh.root.spikes.units
        times = fileh.root.spikes.times
        stim_labels = fileh.root.labels
        digit_lang = fileh.root.extra.keys

        if language == "english":
            nk = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        elif language == "german":
            nk = ["null", "eins", "zwei", "drei", "vier", "fuenf", "sechs", "sieben", "acht", "neun"]
        else:
            nk = []

        raw_data = []
        labels = []
        for idx in tqdm(range(len(times)), desc="{0}".format(file), total=len(times)):

            if str(digit_lang[stim_labels[idx]].decode('UTF-8')) in nk:
                spk_times = np.round(times[idx] * 1000., rounding_precision)
                ids = 700 - units[idx]
                label = stim_labels[idx]
                offset = spk_times.max()
                tmp = [(ids[k], t) for k, t in enumerate(spk_times)]
                sl = SpikeList(tmp, list(np.unique(ids)), t_start=resolution, t_stop=offset)
                raw_data.append(sl)
                labels.append(label)
        return raw_data, labels

    if not os.path.exists(path):
        os.makedirs(path)

    if not os.listdir(path):
        logger.warning("Dataset not available, please download and store in target folder")
    else:
        logger.info("Loading SHD dataset...")

    train_set, train_labels = load_hds_set(path, data_set="train", language=language)
    test_set, test_labels = load_hds_set(path, data_set="test", language=language)

    if concatenate:
        data = train_set + test_set
        labels = train_labels + test_labels
        logger.info(
            "Concatenating dataset (original partition in train+test will be lost): \n\t- T={0!s}".format(len(labels)))

        data_set = {k: [] for k in np.unique(labels)}
        for k, v in zip(labels, data):
            data_set[k].append(v)
        return data_set
    else:
        logger.info("\t- T_train={0!s}\n\t- T_test={1!s}".format(len(train_labels), len(test_labels)))
        train_set = {k: [] for k in np.unique(train_labels)}
        test_set = {k: [] for k in np.unique(test_labels)}
        for k, v in zip(train_labels, train_set):
            train_set[k].append(v)
        for k, v in zip(test_labels, test_set):
            test_set[k].append(v)
        return train_set, test_set


def spike_patterns_from_rate_matrix_gen(rates_matrix, pattern_durations, resolution, rng=None, jitter=0,
                                        labels=None):
    """
    Generates SpikeLists which are based on the rates from the rates_matrix.

    :param rates_matrix: 2D matrix with rate encoded data set
    :param pattern_durations: list of durations [ms] for each data array in the rates_matrix
    :param resolution:
    :param rng: random number generator state object (optional). Either None or a numpy.random.RandomState object,
    or an object with the same interface
    :param jitter: jit
    :return spk_pttrns: generator for spike patterns
    """
    for i, rate_array in enumerate(rates_matrix):
        if len(pattern_durations) == rates_matrix.shape[0]:
            duration = pattern_durations[i] + 2 * jitter
        else:
            duration = pattern_durations[0] + 2 * jitter
        duration += (jitter * 2)
        spk_list = generate_spike_template(rate_array, duration, resolution=resolution, rng=rng)

        yield spk_list


def spikes_from_spectrogram(spectrogram, min_rate, max_rate, resolution, duration):
    """
    Encodes a spectrogram into a SpikeList.

    :param spectrogram: a single spectrogram of an audio file
    :param min_rate: minimum spike rate for the encoding
    :param max_rate: maximum spike rate for the encoding
    :param resolution: resolution for the spike generation
    :param duration: duration of the resulting SpikeList
    :return: a SpikeList which encodes the spectrogram
    """
    segment_duration = duration / spectrogram.shape[1]
    dim = spectrogram.shape[0]
    complete_ids = np.arange(dim)

    rate_encoded_spec = np.interp(spectrogram, (spectrogram.min(), spectrogram.max()), (min_rate, max_rate))
    spike_lists = list(spike_patterns_from_rate_matrix_gen(rate_encoded_spec.transpose(), [segment_duration], resolution))

    result_pattern = spike_lists[0]
    result_pattern.complete(id_list=complete_ids)

    for idx, spk_lst in enumerate(spike_lists[1:]):
        spk_lst.complete(id_list=complete_ids)
        spk_lst.dimensions = dim
        result_pattern.merge(spk_lst.time_offset(segment_duration * (idx + 1), new_spikeList=True))

    result_pattern.dimensions = dim
    result_pattern.t_stop = duration
    return result_pattern


# ######################################################################################################################
class ImageFrontend(Embeddings):
    """
    Standard object to load, parse and preprocess image data
    """
    def __init__(self, path=None, label=None, vocabulary=None, normalize=True):
        """
        ImageFrontend constructor
        :param path: full path to the location where images are stored
        :param label: dataset label, if any
        :param vocabulary: unique tokens in the vocabulary
        :param normalize: normalize dataset
        """
        super().__init__(vocabulary)
        self.path = path
        self.label = label
        self.label_map = None
        self.stimulus_set = None
        self.vocabulary = vocabulary
        self.dimensions = None
        self.encoding_parameters = None
        self.dynamic_embedding = None

        def load(frontend):
            """
            Load image dataset
            :return:
            """
            frontend.stimulus_set = {}
            if frontend.label == 'mnist':
                data_set = load_mnist(frontend.path, concatenate=True)
                data_labels = [k for k in data_set.keys()]
            elif frontend.label == 'cifar-10':
                data_set = load_cifar(frontend.path, concatenate=True)
                data_labels = [k for k in data_set.keys()]
            else:
                raise NotImplementedError("{0} load is not implemented yet".format(label))
            assert len(frontend.vocabulary) <= len(data_labels), "{0!s} can only be used in tasks with {1!s} tokens or " \
                                                             "less".format(frontend.label, len(data_labels))
            frontend.label_map = {modified: (original, modified) for original, modified in zip(data_labels, frontend.vocabulary)}
            for k in frontend.vocabulary:
                frontend.stimulus_set.update({k: data_set[frontend.label_map[k][0]]})
            frontend.dimensions = [x.shape[0] for x in frontend.stimulus_set[frontend.vocabulary[0]]][0] # assuming all images have the
            # same dimensions
            return frontend

        load(self)
        if normalize:
            self._normalize()
        self.total_size = self.dimensions * len(self)

    def _normalize(self):
        """
        Normalize image vectors to [0,1]
        :return:
        """
        for k, v in self.stimulus_set.items():
            self.stimulus_set[k] = [x/255. for x in v]

    def unfold(self, to_spikes=False, to_signal=False, **params):
        """
        Unfold the unique images (represented as vectors) into time-dependent processes (continuous signals or spike
        patterns, see DynamicEmbeddings). To avoid memory overload, this is done online, taking one token at a time
        and converting it
        :param to_spikes: [bool] convert to spike sequence
        :param to_signal: [bool] convert to continuous sequence
        :param params: [dict] - signal properties
        :return DynamicEmbedding:
        """
        self.encoding_parameters = {'to_spikes': to_spikes, 'to_signal': to_signal, 'params': params}
        self.dynamic_embedding = DynamicEmbeddings(self.vocabulary)
        self.dynamic_embedding.embedding_dimensions = self.dimensions

    def draw_stimulus_sequence(self, seq, as_array=False, unfold=False, onset_time=0., continuous=False, intervals=None,
                               verbose=True):
        """
        Parse a symbolic sequence and return the corresponding encoded stimuli
        :param seq: ordered list of symbols
        :param as_array: return the sequence as a full array (True) or as an iterator (False)
        :param verbose: log info
        :return: dim x T array
        """
        if verbose:
            logger.info("Generating stimulus sequence: {0!s} symbols".format(len(seq)))

        if unfold and self.dynamic_embedding is not None:
            self.dynamic_embedding = self.dynamic_embedding.unfold_discrete_set(self.stimulus_set, seq,
                                                                   to_spikes=self.encoding_parameters['to_spikes'],
                                                                   to_signal=self.encoding_parameters['to_signal'],
                                                                   verbose=verbose,
                                                                   **self.encoding_parameters['params'])
            if continuous:
                out = self.dynamic_embedding._sequence_iterator(seq, onset_time=onset_time, intervals=intervals)
                tokens, stim_info = self.dynamic_embedding.concatenate_sequence(out, onset=onset_time)
            else:
                tokens = self.dynamic_embedding._sequence_iterator(seq, onset_time, intervals)
                stim_info = None

            return tokens, stim_info
        else:
            out = self._sequence_iterator(seq)
        if as_array:
            enc_sequence = [enc for enc in out]
            return np.array(enc_sequence).T
        else:
            return out

    def plot_sample(self, seq, continuous=False, intervals=None, display=True, save=False):
        """
        Plots an example sequence
        :param seq: input sequence
        :param unfold: unfold sequence or not
        :return:
        """
        stim_seq = self.draw_stimulus_sequence(seq, as_array=False)

        fig = pl.figure(figsize=(10, 2))
        fig.suptitle(r'Example string: {0!s} [{1!s}]'.format(' '.join(seq), self.label))
        for iid, (sym, enc) in enumerate(zip(seq, stim_seq)):
            ax = fig.add_subplot(1, len(seq), iid + 1)
            ax.imshow(vec2im(enc), aspect='auto', cmap='binary')
            ax.set_xlabel(sym+str(self.label_map[sym]))
            ax.set_xticks([])
            ax.set_yticks([])

        if self.dynamic_embedding is not None:
            stim_seq, stim_info = self.draw_stimulus_sequence(seq, as_array=continuous, unfold=True, continuous=continuous,
                                                   intervals=intervals)
            self.dynamic_embedding.plot_sample(seq, stim_seq, continuous=continuous, intervals=intervals,
                                               display=display, save=save)

    def save(self):
        super().save()


# ######################################################################################################################
# TODO - this whole class in only to exemplify, the whole thing needs to be re-implemented
class AudioFrontend(DynamicEmbeddings):
    """
    Standard object to load, parse and preprocess audio data
    """
    def __init__(self, path=None, label=None, vocabulary=None, resolution=0.1, language=None):
        """
        AudioFrontend constructor
        :param path: full path to the location where images are stored
        :param label: dataset label, if any
        :param vocabulary:
        :param resolution:
        :param language: (if applicable)

        """
        super().__init__(vocabulary)
        self.path = path
        self.label = label
        self.label_map = None
        self.data_set = None
        self.dimensions = None
        self.stimulus_set = {}

        self._is_spks = True
        self._jitter = None
        self.dt = resolution
        self.total_set_size = len(self.vocabulary)
        self._is_iterator = False

        def load(self, concatenate=True):
            """
            Load dataset from the given path
            :param dimensions: number of input channels
            :return:
            """
            self.data_set = {}
            if self.label == 'fsdd':
                logger.info("free spoken digit audio dataset: ")
                data_set, speakers, rates, durations = load_audio(self.path)
                data_labels = data_set.keys()
                assert len(self.vocabulary) <= len(np.unique(data_labels)[0]), "{0!s} can only be used in tasks with " \
                                                                               "{1!s} tokens or less".format(self.label,
                                                                                                         len(data_labels))
                self.label_map = {modified: (original, modified) for original, modified in zip(data_labels, self.vocabulary)}
                for idx, k in enumerate(self.vocabulary):
                    self.data_set.update({k: data_set[self.label_map[k][0]]})
            elif self.label == 'hdspikes':
                logger.info("Heidelberg digits spiking dataset: ")
                self.dimensions = 700
                data_set = load_hds(path, resolution=resolution, concatenate=concatenate, language=language)
                data_labels = [k for k in data_set.keys()]
                assert len(self.vocabulary) <= len(data_labels), "{0!s} can only be used in tasks with {1!s} tokens " \
                                                                 "or " \
                                                                 "less".format(self.label, len(data_labels))
                self.label_map = {modified: (original, modified) for original, modified in
                                  zip(data_labels, self.vocabulary)}
                for k in self.vocabulary:
                    self.stimulus_set.update({k: data_set[self.label_map[k][0]]})
                self.dimensions = 700
            else:
                raise NotImplementedError("{0} load is not implemented yet".format(label))
            return self
        load(self)
        self.total_size = self.dimensions * len(self)

    def save(self):
        super().save()

    # def calculate_spectrograms(self, n_dim):
    #     """
    #     Calculate spectrograms for all the audio data in the dataset
    #     :param n_dim: number of frequency blocks of the spectrogram (Dimension of the frequency axis)
    #     :return:
    #     """
    #     self.spectrograms = {}
    #     self.frequency_axes = {}
    #     self.time_axes = {}
    #     self.dimensions = n_dim
    #     pad_to = (n_dim - 1) * 2
    #     for k, v in self.data_set.items():
    #         self.spectrograms[k] = []
    #         self.frequency_axes[k] = []
    #         self.time_axes[k] = []
    #         for audio_stream in v:
    #             spec, freq, t = specgram(audio_stream, pad_to=pad_to)
    #             self.spectrograms[k].append(spec)
    #             self.frequency_axes[k].append(freq)
    #             self.time_axes[k].append(t)
    #
    # def to_spikes(self, spec_dim, min_rate, max_rate, resolution): #, durations=None):
    #     """
    #     Convert spectrograms to spike lists
    #     :return:
    #     """
    #     self.spike_patterns = {}
    #     if self.spectrograms is None:
    #         self.calculate_spectrograms(spec_dim)
    #     for k, v in self.spectrograms.items():
    #         self.spike_patterns[k] = []
    #         for i, specg in enumerate(v):
    #             duration = self.time_axes[k][i].max() - self.time_axes[k][i].min() # TODO!!
    #             self.spike_patterns[k].append(spikes_from_spectrogram(specg, min_rate, max_rate, resolution, duration))

    # def draw_stimulus_sequence(self, seq, iterator=False, as_spect=False, as_spikes=False):
    #     """
    #     Parse a symbolic sequence and return the corresponding encoded stimuli
    #     :param seq: ordered list of symbols
    #     :param as_array: return the sequence as a full array (True) or as an iterator (False)
    #     :return: dim x T array
    #     """
    #     out = self._sequence_iterator(seq, as_spect=as_spect, as_spikes=as_spikes)
    #
    #     if not iterator:
    #         return [enc for enc in out]
    #     else:
    #         return out
    #
    # def _sequence_iterator(self, seq, as_spect=False, as_spikes=False):
    #     """
    #     Returns a generator
    #     :return:
    #     """
    #     for token in seq:
    #         if as_spect:
    #             tk = self.spectrograms[token]
    #         elif as_spikes:
    #             tk = self.spike_patterns[token]
    #         else:
    #             tk = self.data_set[token]
    #         if isinstance(tk, list):# and len(tk) != self.dimensions: multiple instances of the same token
    #             encoded = tk[np.random.choice(len(tk))]
    #         else:
    #             encoded = tk
    #         yield encoded

    # def plot_sample(self, seq, ax=None, plot_spectra=False, plot_spikes=False):
    #     """
    #     Plots an example sequence
    #     :param seq:
    #     :param ax:
    #     :return:
    #     """
    #     fig = pl.figure(figsize=(20, 10))
    #     fig.suptitle(r'Example string: {0!s} [{1!s}]'.format(' '.join(seq), self.label))
    #     stim_seq = self.draw_stimulus_sequence(seq, iterator=False, as_spect=False, as_spikes=False)
    #     if plot_spectra:
    #         stim_spec = self.draw_stimulus_sequence(seq, iterator=False, as_spect=True, as_spikes=False)
    #     if plot_spikes:
    #         stim_spk = self.draw_stimulus_sequence(seq, iterator=False, as_spect=False, as_spikes=True)
    #
    #     for iid, (sym, enc) in enumerate(zip(seq, stim_seq)):
    #         axes = [pl.subplot2grid((7, len(seq)), loc=(0, iid), rowspan=1, colspan=1),
    #                 pl.subplot2grid((7, len(seq)), loc=(1, iid), rowspan=3, colspan=1),
    #                 pl.subplot2grid((7, len(seq)), loc=(4, iid), rowspan=3, colspan=1)]
    #         axes[0].plot(enc)
    #         if plot_spectra and self.spectrograms is not None:
    #             pad_to = (self.dimensions - 1) * 2
    #             axes[1].specgram(stim_spec[iid], Fs=2, pad_to=pad_to, **{'interpolation': 'gaussian'})
    #         if plot_spikes and self.spike_patterns is not None:
    #             stim_spk[iid].raster_plot(with_rate=False, ax=axes[2])
    #             axes[2].set_xlabel('Time [ms]')
    #         axes[2].set_xlabel(str(self.label_map[sym]))
    #         # axes[2].set
    #         axes[0].set_title(sym)


# class Cochlea(object):
#     """
#
#     """
#     pass
#
#
