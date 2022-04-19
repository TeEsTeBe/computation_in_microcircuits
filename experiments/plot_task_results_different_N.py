import os
import sys
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

from utils.general_utils import get_pickled_data_from_subfolders, get_aggregated_result, standard_error_of_mean, get_data_dir


if __name__ == "__main__":
    assert len(sys.argv) >= 2, 'you need to pass the group name as a parameter'
    groupname = sys.argv[1]
    mc_group = f'{groupname}_microcircuit_spikes'
    am_group = f'{groupname}_amorphous_spikes'

    # mc_group = 'microcircuit_different_N'
    # am_group = 'amorphous_different_N'

    datapath_mc = os.path.join(get_data_dir(), 'task_results', mc_group)
    datapath_amorphous = os.path.join(get_data_dir(), 'task_results', am_group)

    network_sizes = [160, 360, 560, 810, 1000, 2000, 5000, 10000]
    readoutnames = {
        'l23_exc_nnls': 'layer 2/3 readout',
        'l5_exc_nnls': 'layer 5 readout',
    }

    results_mc = {readout: {} for readout in readoutnames.keys()}
    results_amorphous = {readout: {} for readout in readoutnames.keys()}
    error_mc = {readout: {} for readout in readoutnames.keys()}
    error_amorphous = {readout: {} for readout in readoutnames.keys()}

    for N in network_sizes:
        for readout in readoutnames.keys():

            result_dicts = get_pickled_data_from_subfolders(datapath_mc, contains_string=f'_N={N}_')
            results_mc[readout][N] = get_aggregated_result(result_dicts, 'xor_spike_pattern', readout, 'kappa_test')
            error_mc[readout][N] = get_aggregated_result(result_dicts, 'xor_spike_pattern', readout, 'kappa_test', accumulation_function=standard_error_of_mean)

            result_dicts = get_pickled_data_from_subfolders(datapath_amorphous, contains_string=f'_N={N}_')
            results_amorphous[readout][N] = get_aggregated_result(result_dicts, 'xor_spike_pattern', readout, 'kappa_test')
            error_amorphous[readout][N] = get_aggregated_result(result_dicts, 'xor_spike_pattern', readout, 'kappa_test', accumulation_function=standard_error_of_mean)

    fig, axes = plt.subplots(nrows=len(readoutnames), figsize=(3, 4), constrained_layout=True)

    width = 0.25
    offset = 1.1 * width/2
    for readout_nr, (readout, readout_title) in enumerate(readoutnames.items()):
        heights_mc = list(results_mc[readout].values())
        height_errors_mc = list(error_mc[readout].values())
        xvals = np.arange(0, len(heights_mc))
        axes[readout_nr].bar(xvals - offset, heights_mc, yerr=height_errors_mc, capsize=2, ecolor='grey', width=width, color='grey', label='data-based circuit')

        heights_am = list(results_amorphous[readout].values())
        height_errors_amorphous = list(error_amorphous[readout].values())
        axes[readout_nr].bar(xvals + offset, heights_am, yerr=height_errors_amorphous, capsize=2, width=width, color='black', label='amorphous circuit')

        axes[readout_nr].set_xticks(xvals)
        axes[readout_nr].set_xticklabels(network_sizes, rotation=45.)

        axes[readout_nr].set_ylim(0, 0.9)
        axes[readout_nr].set_title(readout_title)
        axes[readout_nr].set_ylabel('performance')

    axes[-1].set_xlabel('number of neurons in the circuit')

    plt.legend(prop={'size': 8})
    figures_folder = os.path.join(get_data_dir(), 'figures', 'XOR_performance_different_N')
    os.makedirs(figures_folder, exist_ok=True)
    plt.savefig(os.path.join(figures_folder, 'xor_performance_different_N.pdf'))
