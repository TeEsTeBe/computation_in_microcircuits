import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils.general_utils import get_pickled_data_from_subfolders, get_aggregated_result, standard_error_of_mean, get_data_dir


def plot_memory_task_results(groupname, lines_axes=None, lines_fig=None, save=True, plot_bars=True):

    networks = {
        'microcircuit': {'color': 'green', 'label': 'data-based'},
        'amorphous': {'color': '#cf232b', 'label': 'amorphous'},
        'degreecontrolled': {'color': 'grey', 'label': 'DC'},
        'degreecontrolled_no_io_specificity': {'color': 'black', 'label': 'DC (no io)'},
        'microcircuit_random_dynamics': {'color': 'olive', 'label': 'random dynamics'},
        'microcircuit_static': {'color': 'purple', 'label': 'static synapses'},
        'smallworld': {'color': 'blue', 'label': 'small-world'},
    }

    steps = 15

    readoutnames = {
        'l23_exc_nnls': 'L2/3 readout',
        'l5_exc_nnls': 'L5 readout',
    }

    tasknames = {
        'spike_pattern_classification_S1': 'stream 1',
        'spike_pattern_classification_S2': 'stream 2',
        # 'xor_spike_pattern': 'xor',
    }

    figures_folder = os.path.join(get_data_dir(), 'figures', 'memory_tasks')
    os.makedirs(figures_folder, exist_ok=True)

    if lines_axes is None:
        lines_fig, lines_axes = plt.subplots(nrows=len(tasknames), ncols=len(readoutnames.keys()), sharex='all', sharey='all', figsize=(4, 4))

    results = {}
    results_full = {}
    for networkname, networkdata in networks.items():
        results[networkname] = {}
        results_full[networkname] = {
            'all_tasks_and_readouts': []
        }
        # datapath = os.path.join(get_data_dir(), 'task_results', f'{networkname}_memorytasks_spikes_dur5_steps15')
        datapath = os.path.join(get_data_dir(), 'task_results', f'{groupname}_{networkname}_spikes')
        result_dicts = get_pickled_data_from_subfolders(datapath)

        if plot_bars:
            bars_fig, bars_axes = plt.subplots(nrows=len(tasknames), ncols=len(readoutnames.keys()), sharex='all', sharey='all')

        for task_nr, taskname in enumerate(tasknames.keys()):
            results[networkname][taskname] = {}
            results_full[networkname][taskname] = {}
            delay_taskname = taskname if taskname == 'xor_spike_pattern' else f'delayed_{taskname}'

            for readout_nr, readout in enumerate(readoutnames.keys()):
                readout_results = {}
                readout_results_error = {}
                delays = list(range(steps))
                for delay in delays:
                    if delay == 0:
                        readout_results[delay] = get_aggregated_result(result_dicts, taskname, readout, 'kappa_test')
                        readout_results_error[delay] = get_aggregated_result(result_dicts, taskname, readout, 'kappa_test', accumulation_function=standard_error_of_mean)
                    elif delay == 1:
                        readout_results[delay] = get_aggregated_result(result_dicts, delay_taskname, readout, 'kappa_test')
                        readout_results_error[delay] = get_aggregated_result(result_dicts, delay_taskname, readout, 'kappa_test', accumulation_function=standard_error_of_mean)
                    else:
                        readout_results[delay] = get_aggregated_result(result_dicts, f'{delay_taskname}_delay{delay}', readout, 'kappa_test')
                        readout_results_error[delay] = get_aggregated_result(result_dicts, f'{delay_taskname}_delay{delay}', readout, 'kappa_test', accumulation_function=standard_error_of_mean)

                results[networkname][taskname][readout] = {
                    'kappa_test': readout_results,
                    'sem': readout_results_error,
                }

                all_memory_results_list = []
                for result_dict in result_dicts:
                    memory_results = []
                    for delay in delays:
                        if delay == 0:
                            memory_results.append(result_dict[taskname][readout]['kappa_test'])
                        elif delay == 1:
                            memory_results.append(result_dict[delay_taskname][readout]['kappa_test'])
                        else:
                            memory_results.append(result_dict[f'{delay_taskname}_delay{delay}'][readout]['kappa_test'])

                    all_memory_results_list.append(memory_results)

                sums = [np.sum(x) for x in all_memory_results_list]
                results_full[networkname][taskname][readout] = {
                    'raw': all_memory_results_list,
                    'sums': sums,
                    'sums_mean': np.mean(sums),
                    'sums_sem': standard_error_of_mean(sums)
                }
                results_full[networkname]['all_tasks_and_readouts'].extend(sums)

                lines_axes[task_nr][readout_nr].plot(delays, readout_results.values(), color=networkdata['color'], alpha=1.0, label=networkdata['label'])

                if plot_bars:
                    bars_axes[task_nr][readout_nr].bar(delays, readout_results.values(), yerr=readout_results_error.values(), color='black', capsize=3)
                    if readout_nr == 0:
                        bars_axes[task_nr][readout_nr].set_ylabel('performance')
                    elif readout_nr == len(readoutnames)-1:
                        bars_ax_right = bars_axes[task_nr][readout_nr].twinx()
                        bars_ax_right.set_ylabel(tasknames[taskname])
                    if task_nr == len(tasknames)-1:
                        bars_axes[task_nr][readout_nr].set_xlabel('delay [steps]')
                    elif task_nr == 0:
                        bars_axes[task_nr][readout_nr].set_title(readoutnames[readout])
                    bars_axes[task_nr][readout_nr].set_ylim((0, 1))

        if plot_bars:
            bars_fig.suptitle(networkname)
            bars_fig.savefig(os.path.join(figures_folder, f'{networkname}_memorytask_bars.pdf'))

    # adjust line plot for all networks combined
    lines_axes[0][0].set_ylabel('performance')
    lines_axes[1][0].set_ylabel('performance')
    lines_axes[1][0].set_xlabel('delay [steps]')
    lines_axes[1][1].set_xlabel('delay [steps]')
    lines_axes[0][0].set_title('L2/3 readout')
    lines_axes[0][1].set_title('L5 readout')
    lines_ax_top_right = lines_axes[0][1].twinx()
    lines_ax_bottom_right = lines_axes[1][1].twinx()
    lines_ax_top_right.set_ylabel('stream 1')
    lines_ax_bottom_right.set_ylabel('stream 2')
    lines_ax_top_right.set_yticks([])
    lines_ax_bottom_right.set_yticks([])
    lines_axes[0][0].set_ylim((0,1))
    lines_axes[0][0].set_xlim((0,steps))
    # lines_axes[0][1].legend(prop={'size': 8})

    lines_fig.tight_layout(w_pad=0.3)

    if save:
        lines_fig.savefig(os.path.join(figures_folder, 'memorytasks_lines.pdf'))
        lines_fig.savefig(os.path.join(figures_folder, 'memorytasks_lines.svg'), transparent=True)

    with open(os.path.join(figures_folder, 'results.pkl'), 'wb') as results_file:
        pickle.dump(results, results_file)

    with open(os.path.join(figures_folder, 'results_full.pkl'), 'wb') as results_full_file:
        pickle.dump(results_full, results_full_file)

    return lines_axes, lines_fig


if __name__ == "__main__":
    assert len(sys.argv) >= 2, 'You have to add the group name as a parameter to the script!'
    groupname = sys.argv[1]
    plot_memory_task_results(groupname)
