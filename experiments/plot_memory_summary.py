import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.general_utils import get_data_dir

network_colors = {
    'microcircuit': 'green',
    'amorphous': '#cf232b',
    'degreecontrolled': 'grey',
    'degreecontrolled_no_io_specificity': 'black',
    'microcircuit_random_dynamics': 'olive',
    'microcircuit_static': 'purple',
    'smallworld': 'blue',
}

# network_labels = {
#     'microcircuit': 'data-based',
#     'amorphous': 'amorphous',
#     'degreecontrolled': 'degree-controlled',
#     'degreecontrolled_no_io_specificity': 'degree-controlled (no io)',
#     'microcircuit_random_dynamics': 'random dynamics',
#     'microcircuit_static': 'static synapses',
#     'smallworld': 'small-world',
# }
network_labels = {
    'microcircuit': 'DB',
    'amorphous': 'AM',
    'degreecontrolled': 'DC',
    'degreecontrolled_no_io_specificity': 'DCio',
    'microcircuit_random_dynamics': 'RD',
    'microcircuit_static': 'SS',
    'smallworld': 'SW',
}


def get_memory(taskresults, cutoff=0.05):
    max_idx = np.argmax(taskresults)
    from_max = taskresults[max_idx:]
    cutoff_idx = np.argmax(from_max<cutoff)

    return cutoff_idx


def main():
    assert len(sys.argv) >= 2, 'You need to give the pickled results as parameter!'

    results_path = sys.argv[1]
    with open(results_path, 'rb') as results_file:
        results = pickle.load(results_file)
    figures_path = os.path.join(get_data_dir(), 'figures', 'memory')
    os.makedirs(figures_path, exist_ok=True)

    plt.clf()
    plt.close('all')
    plot_bars_per_stream_and_readout(results)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'memory_sum.pdf'))
    plt.savefig(os.path.join(figures_path, 'memory_sum.svg'), transparent=True)

    plt.clf()
    plt.close('all')
    plot_averaged_bars(results)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'averaged_memory_bars.pdf'))
    plt.savefig(os.path.join(figures_path, 'averaged_memory_bars.svg'), transparent=True)

    plt.clf()
    plt.close('all')
    fig_legend = plot_legend()
    fig_legend.savefig(os.path.join(figures_path, 'legend.pdf'))
    fig_legend.savefig(os.path.join(figures_path, 'legend.svg'), transparent=True)

    if len(sys.argv) >= 3:
        avg_task_results_path = sys.argv[2]
        df = pd.read_csv(avg_task_results_path, index_col=0)
        old_memory_results = {}
        for network in results.keys():
            old_memory_results[network] = {
                'all_tasks_and_readouts': df.loc['memory'][network]
            }

        plt.clf()
        plt.close('all')
        plot_averaged_bars(old_memory_results)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, 'old_memory_bars.pdf'))
        plt.savefig(os.path.join(figures_path, 'old_memory_bars.svg'), transparent=True)


def plot_legend():
    fig = plt.figure()
    fig_legend = plt.figure(figsize=(9, 0.6))

    ax = fig.add_subplot(111)
    linewidth = 8
    db, = ax.plot([0], [0], label=f'data-based (DB)', color=network_colors['microcircuit'], linewidth=linewidth)
    am, = ax.plot([0], [0], label=f'amorphous (AM)', color=network_colors['amorphous'], linewidth=linewidth)
    dc, = ax.plot([0], [0], label=f'degree-controlled (DC)', color=network_colors['degreecontrolled'], linewidth=linewidth)
    dcio, = ax.plot([0], [0], label=f'degree-controlled without i/o (DCio)', color=network_colors['degreecontrolled_no_io_specificity'], linewidth=linewidth)
    rd, = ax.plot([0], [0], label=f'random dynamics (RD)', color=network_colors['microcircuit_random_dynamics'], linewidth=linewidth)
    ss, = ax.plot([0], [0], label=f'static synapses (SS)', color=network_colors['microcircuit_static'], linewidth=linewidth)
    sm, = ax.plot([0], [0], label=f'small-world (SW)', color=network_colors['smallworld'], linewidth=linewidth)

    fig_legend.legend(handles=[db, am , dc, dcio, rd, ss, sm], ncol=4)

    return fig_legend


def plot_averaged_bars(results, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(2, 2))

    averaged_values = {}
    networks = np.array(list(network_labels.keys()))

    for network in networks:
        averaged_values[network] = np.mean(results[network]['all_tasks_and_readouts'])

    values = np.array(list(averaged_values.values()))
    sorted_indices = np.argsort(values)[::-1]

    sorted_colors = [network_colors[net] for net in networks[sorted_indices]]
    ax.bar(range(len(networks)), values[sorted_indices], color=sorted_colors)
    ax.set_xticks(range(len(networks)))
    ax.set_xticklabels([network_labels[net] for net in networks[sorted_indices]], rotation=90)
    for xtick, color in zip(ax.get_xticklabels(), sorted_colors):
        xtick.set_color(color)
    ax.set_ylabel('avg. memory')
    # fig.autofmt_xdate()

    return ax


def plot_bars_per_stream_and_readout(results, axes=None):
    readout_labels = {
        'l23_exc_nnls': 'L2/3 readout',
        'l5_exc_nnls': 'L5 readout',
    }
    tasks = ['spike_pattern_classification_S1', 'spike_pattern_classification_S2']
    readouts = readout_labels.keys()
    networks = list(results.keys())
    if axes is None:
        fig, axes = plt.subplots(nrows=len(tasks), ncols=len(readouts), sharey='all', sharex='all', figsize=(4, 4))
    for task_nr, task in enumerate(tasks):
        for readout_nr, readout in enumerate(readouts):
            network_memory_dict = {}
            network_sems_dict = {}
            for network in networks:
                # delay_kappa_dict = results[network][task][readout]['kappa_test']
                network_memory_dict[network] = results[network][task][readout]['sums_mean']
                network_sems_dict[network] = results[network][task][readout]['sums_sem']
                # kappas = np.array(list(delay_kappa_dict.values()))
                # delays = np.array(delay_kappa_dict.keys())

                # if fct == 'steps':
                #     network_memory_dict[network] = get_memory(kappas)
                # elif fct == 'sum':
                #     network_memory_dict[network] = np.sum(kappas)
                # elif fct == 'normalized':
                #     network_memory_dict[network] = np.sum(kappas/(np.max(kappas)))

            colors = [network_colors[net] for net in networks]
            for x, y, sem, color in zip(range(len(networks)), network_memory_dict.values(), network_sems_dict.values(),
                                        colors):
                axes[task_nr][readout_nr].errorbar(x, y, yerr=sem, ecolor=color, capsize=3)
            axes[task_nr][readout_nr].bar(range(len(networks)), network_memory_dict.values(), color=colors)
            axes[task_nr][readout_nr].set_xticks(range(len(networks)))
            # axes[task_nr][readout_nr].set_xticklabels([network_labels[net] for net in networks], rotation=45)
            axes[task_nr][readout_nr].set_xticklabels([network_labels[net] for net in networks], rotation=90)
            for xtick, color in zip(axes[task_nr][readout_nr].get_xticklabels(), colors):
                xtick.set_color(color)
            if task_nr == 0:
                axes[task_nr][readout_nr].set_title(readout_labels[readout])
            if readout_nr == 0:
                axes[task_nr][readout_nr].set_ylabel('memory [sum]')
            elif readout_nr == len(readouts) - 1:
                ax_right = axes[task_nr][readout_nr].twinx()
                ax_right.set_ylabel(f'stream {task[-1]}')
                ax_right.set_yticks([])
    # fig.autofmt_xdate()

    return axes


if __name__ == '__main__':
    main()