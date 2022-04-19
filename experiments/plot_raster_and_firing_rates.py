import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt

from utils.general_utils import get_data_dir


def plot_raster(spikingdata, max_time=450., ax=None, exc_color='black', inh_color='red', bigger_markers=False):
    if ax is None:
        fig, ax = plt.subplots()

    if bigger_markers:
        markersize = 1.
    else:
        markersize = 0.5

    tick_positions = []
    tick_labels = []
    min_senders = np.inf
    max_senders = -np.inf
    for population, times_and_senders in spikingdata.items():
        times = times_and_senders['times']
        senders = times_and_senders['senders']
        min_senders = min(min_senders, min(senders))
        max_senders = max(max_senders, max(senders))
        color = exc_color if population.endswith('exc') else inh_color
        ax.scatter(times, senders, marker="|", s=markersize, linewidths=markersize, color=color)
        tick_labels.append(population)
        tick_positions.append(min(senders) + (max(senders)-min(senders))/2)

    ax.set_ylim((min_senders, max_senders))
    ax.set_xlim((0, max_time))
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel('duration [ms]')

    return ax


def plot_firing_hist(rates, population, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    if 'exc' in population:
        color = 'black'
    elif 'inh' in population:
        color = 'red'
    else:
        color = 'gray'

    ax.hist(rates, bins=np.arange(0., 100.1, 100./16.), color=color)
    ax.set_xlim((0, 100))
    ax.set_yticks([])

    return ax


def get_row_and_col(popname):
    if 'exc' in popname:
        col = 0
    elif 'inh' in popname:
        col = 1
    else:
        col = 2

    if '23' in popname:
        row = 0
    elif '4' in popname:
        row = 1
    else:
        row = 2

    return row, col


def main(create_large_figures=False):

    if create_large_figures:
        figsize = (4.5, 3)
        name_appendix = '_large'
    else:
        figsize = (3, 2)
        name_appendix = ''

    figures_path = os.path.join(get_data_dir(), 'figures', 'raster_and_firing')
    information = {
        'hhneuron': {
            'microcircuit': {
                'spiking_data_path': os.path.join(figures_path, 'hhneuron_mc_spikedetector_events_450.0ms.pkl'),
                'firingrates_data_path': os.path.join(figures_path, 'hhneuron_mc_firing_rates_450.0ms_100.0ms.pkl'),
            },
            'amorphous': {
                'spiking_data_path': os.path.join(figures_path, 'hhneuron_am_spikedetector_events_450.0ms.pkl'),
                'firingrates_data_path': os.path.join(figures_path, 'hhneuron_am_firing_rates_450.0ms_100.0ms.pkl'),
            },
        },
        'nonoise': {
            'microcircuit': {
                'spiking_data_path': os.path.join(figures_path, 'hhneuronnonoise_mc_spikedetector_events_450.0ms.pkl'),
                'firingrates_data_path': os.path.join(figures_path, 'hhneuronnonoise_mc_firing_rates_450.0ms_100.0ms.pkl'),
            },
            'amorphous': {
                'spiking_data_path': os.path.join(figures_path, 'hhneuronnonoise_am_spikedetector_events_450.0ms.pkl'),
                'firingrates_data_path': os.path.join(figures_path, 'hhneuronnonoise_am_firing_rates_450.0ms_100.0ms.pkl'),
            },
        },
        'iaf': {
            'microcircuit': {
                'spiking_data_path': os.path.join(figures_path, 'iafneuron_mc_spikedetector_events_450.0ms.pkl'),
                'firingrates_data_path': os.path.join(figures_path, 'iafneuron_mc_firing_rates_450.0ms_100.0ms.pkl'),
            },
            'amorphous': {
                'spiking_data_path': os.path.join(figures_path, 'hhneuron_am_spikedetector_events_450.0ms.pkl'),
                'firingrates_data_path': os.path.join(figures_path, 'hhneuron_am_firing_rates_450.0ms_100.0ms.pkl'),
            },
        },
    }
    data = {}

    for neurontype, typedata in information.items():
        data[neurontype] = {}
        for network, networkdata in typedata.items():
            data[neurontype][network] = {}
            with open(networkdata['spiking_data_path'], 'rb') as spk_data_file:
                data[neurontype][network]['spiking'] = pickle.load(spk_data_file)
            with open(networkdata['firingrates_data_path'], 'rb') as firing_data_file:
                data[neurontype][network]['firing'] = pickle.load(firing_data_file)

    for (neurontype, data_per_network), title in zip(data.items(), ['original', 'disabled noise', 'iaf neuron']):
        for network, networkdata in data_per_network.items():
            plt.clf()
            plt.close('all')
            fig_raster, ax_raster = plt.subplots(figsize=figsize)
            spiking_data = networkdata['spiking']
            plot_raster(spiking_data, ax=ax_raster, bigger_markers=create_large_figures)
            ax_raster.set_yticks([140, 336, 476])
            ax_raster.set_yticklabels(['L5', 'L4', 'L2/3'])
            ax_raster.set_ylabel(title)
            if network == 'microcircuit':
                ax_raster.set_title('data-based')
            else:
                ax_raster.set_title(network)
            plt.tight_layout()
            fig_raster.savefig(os.path.join(figures_path, f'{neurontype}_{network}_raster_plot{name_appendix}.pdf'))
            fig_raster.savefig(os.path.join(figures_path, f'{neurontype}_{network}_raster_plot{name_appendix}.svg'), transparent=True)

            plt.clf()
            plt.close('all')
            rate_data = networkdata['firing']
            fig_firing, axes_firing = plt.subplots(3, 3, figsize=figsize, sharex='all', sharey='row')
            for population, rates in rate_data.items():
                row, col = get_row_and_col(population)
                plot_firing_hist(rates, population, ax=axes_firing[row][col])
            axes_firing[0][0].set_title('exc.')
            axes_firing[0][1].set_title('inh.')
            axes_firing[0][2].set_title('both')
            axes_firing[0][0].set_ylabel('L2/3')
            axes_firing[1][0].set_ylabel('L4')
            axes_firing[2][0].set_ylabel('L5')
            fig_firing.text(0.5, 0.02, 'firing rate [Hz]', ha='center')
            plt.tight_layout(w_pad=0.1, h_pad=0.1)
            fig_firing.savefig(os.path.join(figures_path, f'{neurontype}_{network}_firing_hists{name_appendix}.pdf'))
            fig_firing.savefig(os.path.join(figures_path, f'{neurontype}_{network}_firing_hists{name_appendix}.svg'), transparent=True)


if __name__ == "__main__":
    if 'large' in sys.argv:
        create_large_figures = True
    else:
        create_large_figures = False

    main(create_large_figures=create_large_figures)
