'''
# audio encoding
from os import listdir
from os.path import isfile, join
import cochlea
import thorns as th

import matplotlib.pyplot as pl
import numpy as np
from scipy.io import wavfile
from matplotlib.mlab import specgram

# from tools.input_architect.sequences import SymbolicSequencer
# from tools.signals.spikes import SpikeList
#
# # random input sequence
# vocabulary_size = 3
# T = 1000
# plot_nStrings = 3
#
# sym = SymbolicSequencer(label='random', set_size=vocabulary_size)
# seq = sym.generate_random_sequence(T=T)


# audio data (spoken digits)
# audio_path = '/home/neuro/Desktop/playground/func-neurarch/media/datasets/audio/free-spoken-digit-dataset' \
#             '/recordings/'
# audio_fsdd = AudioFrontend(path=audio_path, label='fsdd', vocabulary=nad.tokens).load()
# audio_fsdd.calculate_spectrograms(n_dim=100)
# audio_fsdd.to_spikes(spec_dim=100, min_rate=10., max_rate=100, resolution=0.1)

# #################
# test timit data
# from nltk.corpus import timit
# print(timit.utteranceids()) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
# item = timit.utteranceids()[5]
# print(timit.phones(item)) # doctest: +NORMALIZE_WHITESPACE
# print(timit.words(item))
# timit.play(item) # doctest: +SKIP


# ######################################################################################################################
# load data
# ######################################################################################################################
audio_dir = '/home/neuro/Desktop/playground/func-neurarch/media/datasets/audio/free-spoken-digit-dataset' \
            '/recordings/'
audio_files = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f))]

labels = []
speakers = []
data = []
rates = []
durations = []

for a_file in audio_files:
    a_file_path = join(audio_dir, a_file)
    rate, d = wavfile.read(a_file_path)
    a_file_split = a_file[:-4].split('_')

    labels.append(a_file_split[0])
    speakers.append(a_file_split[1])
    rates.append(rate)
    durations.append(rate * 1000. / len(d))  # duration in ms
    data.append(d)

N = 1000


# ######################################################################################################################
# cochlea model (python 2!) - dependence on thorns for spike handling can be removed in python3
# ######################################################################################################################
def parse_audio(audio_data, fs, plot=True):
    anf = cochlea.run_zilany2009(audio_data, fs, anf_num=(N, 0, 0), cf=(80, 20000, 100), seed=0,
                                 powerlaw='approximate')

    # spikes = to_spike_list(anf)
    # spikes.raster_plot()

    # Plot auditory nerve response
    anf_acc = th.accumulate(anf, keep=['cf', 'duration'])
    anf_acc.sort_values('cf', ascending=False, inplace=True)

    cfs = anf.cf.unique()
    fig, ax = pl.subplots(3, 1)
    ax[0].plot(audio_data)
    th.plot_neurogram(anf_acc, fs, ax=ax[1])
    th.plot_raster(anf[anf.cf == cfs[30]], ax=ax[2])
    ax[1].set_title("CF = {}".format(cfs[30]))

    return anf, fig

# def to_spike_list(cochlea_obj):
#
#     neuron_ids = []
#     tmp = []
#     for neuron_id, spk_times in cochlea_obj['spikes'].items():
#         neuron_ids.append(neuron_id)
#         tmp.append((neuron_id, spk_times))
#
#     return SpikeList(tmp, np.unique(neuron_ids).tolist())

    # spk_times = data[:, 1]
    # neuron_ids = data[:, 0]
    # tmp = [(neuron_ids[n], spk_times[n]) for n in range(len(spk_times))]
    # self.spiking_activity = signals.spikes.SpikeList(tmp, np.unique(neuron_ids).tolist())


for i, (sound_data, label, speaker) in enumerate(zip(data[:10], labels[:10], speakers[:10])):
    dd = sound_data / (sound_data.max() - sound_data.min()).astype(float)
    anf, fig = parse_audio(dd, 100e3)
    fig.suptitle(r'{0} - {1}'.format(label, speaker))
    img_path = '../data/figures/audio1_spikes_speaker-{0}_lbl-{1}_id-{2}'.format(speaker, label, i)
    pl.savefig(img_path)
    # pl.show()

# ######################################################################################################################
# Direct spectrum encoding
# ######################################################################################################################
from encoders.generators import generate_spike_template
import wave
import sys


def spectrograms_from_audio(audio_data, n_frequency_dims):
    """
    Calculates spectrograms for all audio data sets in audio_data.

    :param audio_data: iterable with the sampled data of multiple audio files
    :param n_frequency_dims: number of frequency blocks of the spectrogram (Dimension of the frequency axis)
    :return:
        spectrograms - list of spectrogram arrays
        frequencies - list of ndarrays with frequencies corresponding to the rows in *spectrum*
        times - list of the times corresponding to midpoints of segments (i.e the columns in *spectrum*)
    """
    pad_to = (n_frequency_dims - 1) * 2
    spectrograms = []
    frequencies = []
    times = []

    for audio_stream in audio_data:
        spec, freq, t = specgram(audio_stream, pad_to=pad_to)
        spectrograms.append(spec)
        frequencies.append(freq)
        times.append(t)

    return spectrograms, frequencies, times


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


def plot_audio(signal, N, spikes):
    fig = pl.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax1.set_title('Signal Wave...')
    ax1.plot(signal)
    # pl.savefig(img_path + '_wav')
    # pl.close()

    pad_to = (N - 1) * 2
    ax2.specgram(data[i], Fs=2, pad_to=pad_to, **{'interpolation': 'gaussian'})
    pl.show()
    # pl.savefig(img_path + '_specgram')
    # pl.close()

    spikes.raster_plot(with_rate=False, ax=ax3)
    ax3.set_xlabel('Time [ms]')
    return fig

# ############################################################
N = 1000
durations = 200.
resolution = 0.1
min_rate = 10.
max_rate = 100.
online = True

# assert hasattr(stim_obj, 'dataset'), 'You have to add a dataset to the stimulus object'

specgrams, freqs, times = spectrograms_from_audio(data, N)

if isinstance(durations, list) and len(durations) != len(specgrams):
    durations = (np.ones(len(specgrams)) * durations[0]).tolist()
else:
    durations = (np.ones(len(specgrams)) * durations).tolist()


def spikes_generator(specgrams):
    for i, specg in enumerate(specgrams):
        yield spikes_from_spectrogram(specg, min_rate, max_rate, resolution, durations[i])

if online:
    spike_patterns = spikes_generator(specgrams)
else:
    spike_patterns = list(spikes_generator(specgrams))


for i in range(10):
    spikes = next(spike_patterns)
    img_path = '../data/audio2_spikes_speaker-{0}_lbl-{1}_id-{2}'.format(speakers[i], labels[i], i)

    # spikes.raster_plot(with_rate=True, save=img_path, display=False)
    # pl.close()

    audio_file = audio_dir + '{0}_{1}_{2}.wav'.format(labels[i], speakers[i], i)
    spf = wave.open(audio_file, 'r')

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')

    # If Stereo
    if spf.getnchannels() == 2:
        print('Just mono files')
        sys.exit(0)

    fig = plot_audio(signal, N, spikes)
    fig.suptitle(r'{0} - {1}'.format(labels[i], speakers[i]))
    pl.savefig(img_path)
    # pl.show()


# def wav2mfcc(file_path, max_pad_len=20):
#     wave, sr = librosa.load(file_path, mono=True, sr=None)
#     wave = wave[::3]
#     mfcc = librosa.feature.mfcc(wave, sr=8000)
#     pad_width = max_pad_len - mfcc.shape[1]
#     mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
#     return mfcc
#
#
# def extract_mfcc(file_path, utterance_length):
#     # Get raw .wav data and sampling rate from librosa's load function
#     raw_w, sampling_rate = librosa.load(file_path, mono=True)
#
#     # Obtain MFCC Features from raw data
#     mfcc_features = librosa.feature.mfcc(raw_w, sampling_rate)
#     if mfcc_features.shape[1] > utterance_length:
#         mfcc_features = mfcc_features[:, 0:utterance_length]
#     else:
#         mfcc_features = np.pad(mfcc_features, ((0, 0), (0, utterance_length - mfcc_features.shape[1])),
#                                mode='constant', constant_values=0)
#
#     return mfcc_features
#
#
# def display_power_spectrum(wav_file_path):#, utterance_length):
#     # mfcc = extract_mfcc(wav_file_path, utterance_length)
#     mfcc = wav2mfcc(wav_file_path)
#
#     # Plot
#     pl.figure(figsize=(10, 6))
#     pl.subplot(2, 1, 1)
#     librosa.display.specshow(mfcc, x_axis='time')
#     pl.show()
#
#     # Feature information
#     print("Feature Shape: ", mfcc.shape)
#     print("Features: ", mfcc[:, 0])
#
#
# for i in range(10):
#     audio_file = audio_dir + '{0}_{1}_{2}.wav'.format(labels[i], speakers[i], i)
#
#     mfcc = wav2mfcc(audio_file)
#
#     # mfcc2 = extract_mfcc(audio_file, utterance_length=35)
#     display_power_spectrum(audio_file)#, utterance_length=10)
#
#     spf = wave.open(audio_file, 'r')
#
#     # Extract Raw Audio from Wav File
#     signal = spf.readframes(-1)
#     signal = np.fromstring(signal, 'Int16')
#
#     # If Stereo
#     if spf.getnchannels() == 2:
#         print('Just mono files')
#         sys.exit(0)
#
#     fig = pl.figure()
#     ax1 = fig.add_subplot(211)
#     # ax1.set_title('Signal Wave...')
#     ax1.plot(signal)
#     # pl.savefig(img_path + '_wav')
#     # pl.close()
#
#     dt = 0.1
#     NFFT = 1024  # the length of the windowing segments
#     Fs = int(1.0 / dt)  # the sampling frequency
#
#     signal_N = 10
#     pad_to = (signal_N - 1) * 2
#     ax2 = fig.add_subplot(212)
#     ax2.specgram(data[i], NFFT=NFFT, Fs=Fs)#, pad_to=pad_to)#, **{'interpolation': 'gaussian'})
#     # plt.savefig(img_path + '_specgram')
#     # pl.close()
#
#     pl.show()
'''

from tasks.preprocessing import AudioFrontend

import os
import urllib.request
import gzip, shutil
from tensorflow.keras.utils import get_file

cache_dir =os.path.expanduser("../data")
cache_subdir="hdspikes"
print("Using cache dir: %s"%cache_dir)

# The remote directory with the data files
base_url = "https://compneuro.net/datasets"

# Retrieve MD5 hashes from remote
response = urllib.request.urlopen("%s/md5sums.txt" % base_url)
data = response.read()
lines = data.decode('utf-8').split("\n")
file_hashes = {line.split()[1]:line.split()[0] for line in lines if len(line.split()) == 2}


def get_and_gunzip(origin, filename, md5hash=None):
    gz_file_path = get_file(filename, origin, md5_hash=md5hash, cache_dir=cache_dir, cache_subdir=cache_subdir)
    hdf5_file_path=gz_file_path[:-3]
    if not os.path.isfile(hdf5_file_path) or os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path):
        print("Decompressing %s"%gz_file_path)
        with gzip.open(gz_file_path, 'r') as f_in, open(hdf5_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return hdf5_file_path

# Download the Spiking Heidelberg Digits (SHD) dataset
files = [ "shd_train.h5.gz",
          "shd_test.h5.gz",
        ]

for fn in files:
    origin = "%s/%s"%(base_url,fn)
    hdf5_file_path = get_and_gunzip(origin, fn, md5hash=file_hashes[fn])
    print(hdf5_file_path)


# #############
import tables
import numpy as np
from tools.signals.spikes import SpikeList
from tools.utils.operations import determine_decimal_digits, iterate_obj_list

resolution = 0.1

hdf5_file_path = '../../data/hdspikes/shd_train.h5'

rounding_precision = determine_decimal_digits(resolution)
fileh = tables.open_file(hdf5_file_path, mode='r')
units = fileh.root.spikes.units
times = fileh.root.spikes.times
stim_labels = fileh.root.labels

digit_lang = fileh.root.extra.keys

for idx in range(len(times[:5])):


    spk_times = np.round(times[idx] * 1000., rounding_precision)
    ids = 700-units[idx]

    label = stim_labels[idx]

    offset = spk_times.max()

    tmp = [(ids[k], t) for k, t in enumerate(spk_times)]

    sl = SpikeList(tmp, list(np.unique(ids)), t_start=resolution, t_stop=offset)
    sl.raster_plot()

    # tmp = [(700-x, y*1000.) for x, y in zip(units[idx], times)]



# load_hds_set(path='../../data/hdspikes/', data_set="train", concatenate=True)
#
#
#
spoken_digits = AudioFrontend(path='../../data/hdspikes/', label='hdspikes', vocabulary=['a', 'b', 'c'], resolution=0.1)