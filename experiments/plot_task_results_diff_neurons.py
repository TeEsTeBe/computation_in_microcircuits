import os
import argparse
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['hatch.linewidth'] = 0.4
import matplotlib.pyplot as plt

from utils.visualisation import plot_task_result_bars
from utils.general_utils import get_pickled_data_from_subfolders, get_data_dir


def main(hh_group, nonoise_group, iaf_group):
    readoutnames = ['l23_exc_nnls', 'l5_exc_nnls']
    colors = [
        {'microcircuit': 'grey', 'amorphous': 'black'},
        {'microcircuit': 'plum', 'amorphous': 'plum'},
        {'microcircuit': 'wheat', 'amorphous': 'wheat'},
    ]

    names = ['', 'no noise', 'iaf']
    barwidth = 0.15
    baseoffset = 0.2
    offsetstep = 0.09
    fig, axes = plt.subplots(nrows=len(readoutnames), figsize=(6, 5))

    for i, group_prefix in enumerate([hh_group, nonoise_group, iaf_group]):
        # Group names with different runs for spike and rate tasks for the data-based microcircuit (mc) and the amorphous circuit (am).
        # Change this to your chosen group names if necessary
        mc_spikes_group = f'{group_prefix}_microcircuit_spikes'
        am_spikes_group = f'{group_prefix}_amorphous_spikes'
        mc_rates_group = f'{group_prefix}_microcircuit_rates'
        am_rates_group = f'{group_prefix}_amorphous_rates'

        datapath_mc_classification = os.path.join(get_data_dir(), 'task_results', mc_spikes_group)
        result_dicts_mc_classification = get_pickled_data_from_subfolders(datapath_mc_classification)

        datapath_am_classification = os.path.join(get_data_dir(), 'task_results', am_spikes_group)
        result_dicts_amorphous_classification = get_pickled_data_from_subfolders(datapath_am_classification)

        datapath_mc_ratetasks = os.path.join(get_data_dir(), 'task_results', mc_rates_group)
        result_dicts_mc_ratetasks = get_pickled_data_from_subfolders(datapath_mc_ratetasks)

        datapath_amorphous_ratetasks = os.path.join(get_data_dir(), 'task_results', am_rates_group)
        result_dicts_amorphous_ratetasks = get_pickled_data_from_subfolders(datapath_amorphous_ratetasks)

        if names[i] == '':
            axes, xlabels = plot_task_result_bars(result_dicts_mc_classification, axes=axes, color=colors[i]['microcircuit'], width=barwidth, offset=-baseoffset, readoutnames=readoutnames, label=f'data-based', alpha=1., edgecolor='k', ecolor='k', zorder=10)
            axes, _ = plot_task_result_bars(result_dicts_amorphous_classification, axes=axes, color=colors[i]['amorphous'], width=barwidth, offset=baseoffset, readoutnames=readoutnames, label=f'amorphous', alpha=1., zorder=10)
            axes, _ = plot_task_result_bars(result_dicts_mc_ratetasks, axes=axes, color=colors[i]['microcircuit'], width=barwidth, offset=-baseoffset, readoutnames=readoutnames, label=None, alpha=1., edgecolor='k', ecolor='k', zorder=10)
            axes, _ = plot_task_result_bars(result_dicts_amorphous_ratetasks, axes=axes, color=colors[i]['amorphous'], width=barwidth, offset=baseoffset, readoutnames=readoutnames, label=None, alpha=1., zorder=10)
        else:
            additional_offset = ((i-1)*2-1) * offsetstep
            width_factor = 0.8
            alpha = 1.
            axes, xlabels = plot_task_result_bars(result_dicts_mc_classification, axes=axes, color=colors[i]['microcircuit'], width=width_factor*barwidth, offset=-baseoffset+additional_offset, readoutnames=readoutnames, label=names[i], alpha=alpha)
            axes, _ = plot_task_result_bars(result_dicts_amorphous_classification, axes=axes, color=colors[i]['amorphous'], width=width_factor*barwidth, offset=baseoffset+additional_offset, readoutnames=readoutnames, label=None, alpha=alpha)
            axes, _ = plot_task_result_bars(result_dicts_mc_ratetasks, axes=axes, color=colors[i]['microcircuit'], width=width_factor*barwidth, offset=-baseoffset+additional_offset, readoutnames=readoutnames, label=None, alpha=alpha)
            axes, _ = plot_task_result_bars(result_dicts_amorphous_ratetasks, axes=axes, color=colors[i]['amorphous'], width=width_factor*barwidth, offset=baseoffset+additional_offset, readoutnames=readoutnames, label=None, alpha=alpha)

        for ax in axes:
            ax.set_ylim((0, 1))
            ax.set_xticks(np.arange(0., len(xlabels)))
            ax.set_xticklabels(xlabels)

        axes[-1].legend()

    plt.tight_layout()
    figures_folder = os.path.join(get_data_dir(), 'figures', f'diffneurons_task_results_mc_am')
    os.makedirs(figures_folder, exist_ok=True)
    plt.savefig(os.path.join(figures_folder, f'diffneurons_taskresults_mc_amorphous.pdf'))


def parse_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--hh_group', help='Group name for the runs with the normal Hodgkin-Huxley neurons with intrinsic noise', required=True)
    parser.add_argument('--nonoise_group', help='Group name for the runs with the Hodgkin-Huxley neurons without intrinsic noise', required=True)
    parser.add_argument('--iaf_group', help='Group name for the runs with integrate-and-fire neurons', required=True)

    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_cmd()))