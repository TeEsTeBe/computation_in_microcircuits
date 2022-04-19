#!/usr/bin/env python
"""
Run inner ear model from [Zilany2009]_.

"""
# from __future__ import division, absolute_import, print_function


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as dsp
import matplotlib.pyplot as pl

import thorns as th
import cochlea


from os import listdir
from os.path import isfile, join
from scipy.io import wavfile

# ######################################################################################################################
# encode
import tools.visualization.plotting

# audio_dir = '/home/neuro/Desktop/playground/func-neurarch/media/datasets/audio/free-spoken-digit-dataset' \
#             '/recordings/'
# TODO adjust this!
audio_dir = '../data/audio'
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

fs = 100e3 #8000. # [Hz] - free-spoken-digit-dataset
dd = data[0] / (data[0].max() - data[0].min()).astype(float)

# anf = cochlea.run_zilany2014(dd, fs, anf_num=(100, 0, 0), cf=(125, 20000, 100), seed=0, powerlaw='approximate',
#                              species='human')


def parse_audio(audio_data, plot=True):
    anf = cochlea.run_zilany2009(dd, fs, anf_num=(100, 0, 0), cf=(80, 20000, 100), seed=0, powerlaw='approximate')
    # Plot auditory nerve response
    anf_acc = th.accumulate(anf, keep=['cf', 'duration'])
    anf_acc.sort_values('cf', ascending=False, inplace=True)

    cfs = anf.cf.unique()
    fig, ax = pl.subplots(2, 1)
    th.plot_neurogram(anf_acc, fs, ax=ax[0])
    tools.visualization.plotting.plot_raster(anf[anf.cf == cfs[30]], ax=ax[1])
    ax[1].set_title("CF = {}".format(cfs[30]))

    return anf, fig


for sound_data, label, speaker in zip(data, labels, speakers):

    dd = sound_data / (sound_data.max() - sound_data.min()).astype(float)
    anf, fig = parse_audio(sound_data)
    fig.suptitle(r'{0} - {1}'.format(label, speaker))
    pl.show()