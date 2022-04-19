import os
import sys
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

from utils.general_utils import get_pickled_data_from_subfolders, get_aggregated_result, standard_error_of_mean, get_data_dir


if __name__ == "__main__":
    assert len(sys.argv) >= 2, 'You need to pass the group name to the script'
    groupname = sys.argv[1]
    mc_group = f'{groupname}_microcircuit_spikes'
    am_group = f'{groupname}_amorphous_spikes'

    # mc_group = 'microcircuit_different_steps'
    # am_group = 'amorphous_different_steps'

    datapath_mc = os.path.join(get_data_dir(), 'task_results', mc_group)
    datapath_amorphous = os.path.join(get_data_dir(), 'task_results', am_group)

    num_train_steps_list = np.arange(40, 481, 40, dtype=int)
    readoutnames = {
        'l23_exc_nnls': 'layer 2/3 readout',
        'l5_exc_nnls': 'layer 5 readout',
    }

    results_mc_train = {readout: {} for readout in readoutnames.keys()}
    results_mc_test = {readout: {} for readout in readoutnames.keys()}
    results_amorphous_train = {readout: {} for readout in readoutnames.keys()}
    results_amorphous_test = {readout: {} for readout in readoutnames.keys()}

    errors_mc_train = {readout: {} for readout in readoutnames.keys()}
    errors_mc_test = {readout: {} for readout in readoutnames.keys()}
    errors_amorphous_train = {readout: {} for readout in readoutnames.keys()}
    errors_amorphous_test = {readout: {} for readout in readoutnames.keys()}

    task = 'delayed_spike_pattern_classification_S1'

    for num_train_steps in num_train_steps_list:
        for readout in readoutnames.keys():

            result_dicts = get_pickled_data_from_subfolders(datapath_mc, contains_string=f'_train={num_train_steps}_')
            results_mc_train[readout][num_train_steps] = get_aggregated_result(result_dicts, task, readout, 'error_train')
            results_mc_test[readout][num_train_steps] = get_aggregated_result(result_dicts, task, readout, 'error_test')
            errors_mc_train[readout][num_train_steps] = get_aggregated_result(result_dicts, task, readout, 'error_train', accumulation_function=standard_error_of_mean)
            errors_mc_test[readout][num_train_steps] = get_aggregated_result(result_dicts, task, readout, 'error_test', accumulation_function=standard_error_of_mean)

            result_dicts = get_pickled_data_from_subfolders(datapath_amorphous, contains_string=f'_train={num_train_steps}_')
            results_amorphous_train[readout][num_train_steps] = get_aggregated_result(result_dicts, task, readout, 'error_train')
            results_amorphous_test[readout][num_train_steps] = get_aggregated_result(result_dicts, task, readout, 'error_test')
            errors_amorphous_train[readout][num_train_steps] = get_aggregated_result(result_dicts, task, readout, 'error_train', accumulation_function=standard_error_of_mean)
            errors_amorphous_test[readout][num_train_steps] = get_aggregated_result(result_dicts, task, readout, 'error_test', accumulation_function=standard_error_of_mean)

    fig, axes = plt.subplots(nrows=len(readoutnames), figsize=(3, 4), constrained_layout=True)
    linewidth = 0.7
    capsize = 2

    for readout_nr, (readout, readout_title) in enumerate(readoutnames.items()):
        readout_results_mc_train = list(results_mc_train[readout].values())
        readout_errors_mc_train = list(errors_mc_train[readout].values())
        axes[readout_nr].errorbar(num_train_steps_list, readout_results_mc_train, yerr=readout_errors_mc_train, color='black', label='training error for data-based circuits', linestyle='--', capsize=capsize, linewidth=linewidth)

        readout_results_mc_test = list(results_mc_test[readout].values())
        readout_errors_mc_test = list(errors_mc_test[readout].values())
        axes[readout_nr].errorbar(num_train_steps_list, readout_results_mc_test, yerr=readout_errors_mc_test, color='black', label='test error for data-based circuits', capsize=capsize, linewidth=linewidth)

        readout_results_am_train = list(results_amorphous_train[readout].values())
        readout_errors_am_train = list(errors_amorphous_train[readout].values())
        axes[readout_nr].errorbar(num_train_steps_list, readout_results_am_train, yerr=readout_errors_am_train, color='#cf232b', label='training error for amorphous circuits', linestyle='--', capsize=capsize, linewidth=linewidth)

        readout_results_amorphous_test = list(results_amorphous_test[readout].values())
        readout_errors_amorphous_test = list(errors_amorphous_test[readout].values())
        axes[readout_nr].errorbar(num_train_steps_list, readout_results_amorphous_test, yerr=readout_errors_amorphous_test, color='#cf232b', label='test error for amorphous circuits', capsize=capsize, linewidth=linewidth)

        axes[readout_nr].set_ylim(0, 0.5)
        axes[readout_nr].set_xlim(35, 485)
        axes[readout_nr].set_title(readout_title)
        axes[readout_nr].set_ylabel('classification error')

    axes[-1].set_xlabel('number of training examples')

    plt.legend(bbox_to_anchor=(0.0, -0.3, 1., -0.12), ncol=1, mode='expand', borderaxespad=0., prop={'size': 8})
    figures_folder = os.path.join(get_data_dir(), 'figures', 'different_training_steps')
    os.makedirs(figures_folder, exist_ok=True)
    plt.savefig(os.path.join(figures_folder, 'different_training_steps.pdf'))