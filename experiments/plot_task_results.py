import os
import sys
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

from utils.visualisation import plot_task_result_bars
from utils.general_utils import get_pickled_data_from_subfolders, get_data_dir


if __name__ == "__main__":
    if len(sys.argv) > 1:
        group_prefix = f'{sys.argv[1]}_'
    else:
        group_prefix = ''

    # Group names with different runs for spike and rate tasks for the data-based microcircuit (mc) and the amorphous circuit (am).
    # Change this to your chosen group names if necessary
    mc_spikes_group = f'{group_prefix}microcircuit_spikes'
    am_spikes_group = f'{group_prefix}amorphous_spikes'
    mc_rates_group = f'{group_prefix}microcircuit_rates'
    am_rates_group = f'{group_prefix}amorphous_rates'

    datapath_mc_classification = os.path.join(get_data_dir(), 'task_results', mc_spikes_group)
    result_dicts_mc_classification = get_pickled_data_from_subfolders(datapath_mc_classification)

    datapath_am_classification = os.path.join(get_data_dir(), 'task_results', am_spikes_group)
    result_dicts_amorphous_classification = get_pickled_data_from_subfolders(datapath_am_classification)

    datapath_mc_ratetasks = os.path.join(get_data_dir(), 'task_results', mc_rates_group)
    result_dicts_mc_ratetasks = get_pickled_data_from_subfolders(datapath_mc_ratetasks)

    datapath_amorphous_ratetasks = os.path.join(get_data_dir(), 'task_results', am_rates_group)
    result_dicts_amorphous_ratetasks = get_pickled_data_from_subfolders(datapath_amorphous_ratetasks)

    readoutnames = ['l23_exc_nnls', 'l5_exc_nnls']
    # readoutnames = ['l23_exc_linreg', 'l5_exc_linreg']
    fig, axes = plt.subplots(nrows=len(readoutnames), figsize=(6, 5))
    axes, xlabels = plot_task_result_bars(result_dicts_mc_classification, axes=axes, color='grey', width=0.2, offset=-0.11, readoutnames=readoutnames, label='data-based circuit')
    axes, xlabels = plot_task_result_bars(result_dicts_amorphous_classification, axes=axes, color='black', width=0.2, offset=0.11, readoutnames=readoutnames, label='amorphous circuit')
    axes, xlabels = plot_task_result_bars(result_dicts_mc_ratetasks, axes=axes, color='grey', width=0.2, offset=-0.11, readoutnames=readoutnames, label=None)
    axes, xlabels = plot_task_result_bars(result_dicts_amorphous_ratetasks, axes=axes, color='black', width=0.2, offset=0.11, readoutnames=readoutnames, label=None)
    for ax in axes:
        ax.set_ylim((0, 1))
        ax.set_xticks(np.arange(0., len(xlabels)))
        ax.set_xticklabels(xlabels)

    axes[-1].legend()

    plt.tight_layout()
    figures_folder = os.path.join(get_data_dir(), 'figures', f'{group_prefix}task_results_mc_am')
    os.makedirs(figures_folder, exist_ok=True)
    plt.savefig(os.path.join(figures_folder, f'{group_prefix}taskresults_mc_amorphous.pdf'))
