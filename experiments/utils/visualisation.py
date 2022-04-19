import os
import pickle

import nest
import nest.raster_plot
import numpy as np
from matplotlib import pyplot as plt

from experiments.utils import general_utils


def raster_plots(spike_detectors, bin_width=10., save_dir=None, show=True, name=''):

    for pop, sdet in spike_detectors.items():
        spk_times = nest.GetStatus(sdet)[0]['events']['times']
        if len(spk_times) > 1:
            nest.raster_plot.from_device(sdet, title=pop, hist=True, hist_binwidth=bin_width)
            if save_dir is not None:
                plt.savefig(os.path.join(save_dir, f'{name}_{pop}_raster_plot.pdf'))
            if show:
                plt.show()
        elif show:
            print(f'Number of spikes in {pop}: {len(spk_times)}')


def combined_raster_plot(spike_detectors, xlim=None):
    plt.figure(figsize=(15, 8))
    tick_positions = []
    tick_labels = []
    spikedetector_events = {}
    pop_name_translator = {
        'l23_exc': 'L2/3-E',
        'l23_inh': 'L2/3-I',
        'l4_exc': 'L4-E',
        'l4_inh': 'L4-I',
        'l5_exc': 'L5-E',
        'l5_inh': 'L5-I'
    }
    min_sender = np.inf
    max_sender = -np.inf
    for pop_name, sdet in spike_detectors.items():
        ev = nest.GetStatus(sdet, 'events')[0]
        spikedetector_events[pop_name] = ev
        senders = ev['senders']
        times = ev['times']
        xlim_mask = times < xlim[1]
        times = times[xlim_mask]
        senders = senders[xlim_mask]
        if len(senders) > 0:
            min_sender = min(min_sender, min(senders))
            max_sender = max(max_sender, max(senders))
            tick_positions.append(senders.mean())
            if pop_name in pop_name_translator.keys():
                tick_label = pop_name_translator[pop_name]
            else:
                tick_label = pop_name
            tick_labels.append(tick_label)
        else:
            print(f'No spikes detected for population {pop_name}')
        color = 'black' if pop_name.endswith('exc') else '#cf232b'
        plt.scatter(times, senders, marker='.', label=pop_name, color=color)
    fontsize = 20.
    plt.xlabel('time [ms]', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel('population', fontsize=fontsize)
    plt.yticks(tick_positions, tick_labels, fontsize=fontsize)
    plt.tick_params(axis='y', length=0)
    if xlim is not None:
        plt.xlim(xlim)
    if np.inf not in [min_sender, max_sender]:
        plt.ylim((min_sender-2, max_sender+2))
    else:
        print(f'No spikes detected at all!')

    return spikedetector_events


def firing_rate_hist(ax, spk_det, sim_time, num_neurons, title, cut_first_ms=200., color='black'):
    events = nest.GetStatus(spk_det, 'events')
    rates = np.zeros(num_neurons)

    senders = []
    for e in events:
        ts = e['times']
        mask = (cut_first_ms<ts) & (ts<sim_time)
        s = e['senders'][mask]
        senders = np.concatenate((senders, s))

    if len(senders) > 1:  # 0:
        unique_ids, counts = np.unique(senders, return_counts=True)

        for i, (neuron_id, spk_count) in enumerate(zip(unique_ids, counts)):
            rates[i] = spk_count / ((sim_time-cut_first_ms) / 1000.)

        if ax is not None:
            ax.hist(rates, bins=np.arange(0., 100.1, 100/16.), color=color)
            ax.title.set_text(f'{title} ({round(rates.mean(), 1)} spk/sec)')
    elif ax is not None:
        ax.title.set_text(f'{title} (0.)')

    if ax is not None:
        ax.set_xlabel('firing rate')
        ax.set_ylabel('neuron counts')
        ax.set_xlim((0, 100))

    return rates


def plot_all_firing_rate_hists(spike_detectors, pop_counts, sim_time, no_plots=False, cut_first_ms=200.):

    layers = ['23', '4', '5']

    if not no_plots:
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), tight_layout=True, sharey='row')
    else:
        axes = []
        for i in range(len(layers)):
            axes.append([None, None, None])

    firing_rates = {}

    for i, l in enumerate(layers):
        title = f'L{l}'
        exc = f'l{l}_exc'
        inh = f'l{l}_inh'
        both = f'l{l}'

        firing_rates[exc] = firing_rate_hist(axes[i][0], spike_detectors[exc], sim_time, pop_counts[exc], title=title+'-E', cut_first_ms=cut_first_ms, color='black')
        firing_rates[inh] = firing_rate_hist(axes[i][1], spike_detectors[inh], sim_time, pop_counts[inh], title=title+'-I', cut_first_ms=cut_first_ms, color='#cf232b')
        firing_rates[both] = firing_rate_hist(axes[i][2], spike_detectors[exc] + spike_detectors[inh], sim_time, pop_counts[both], title=title+' combined', cut_first_ms=cut_first_ms, color='grey')

    return firing_rates


def store_firing_rate_histograms(network, raster_plot_duration, results_folder, spike_detectors, cut_first_ms=0.):
    firingrates_per_pop = plot_all_firing_rate_hists(spike_detectors, network.pop_counts, raster_plot_duration,
                                                     cut_first_ms=cut_first_ms)
    plt.savefig(os.path.join(results_folder, f'firingrates_{raster_plot_duration}ms_cut{cut_first_ms}ms.pdf'))
    with open(os.path.join(results_folder, f'firing_rates_{raster_plot_duration}ms_{cut_first_ms}ms.pkl'), 'wb') as rates_file:
        pickle.dump(firingrates_per_pop, rates_file)


def store_rasterplot(results_folder, spike_detectors, xlim=None):
    spikedetector_events = combined_raster_plot(spike_detectors, xlim=xlim)
    xlim_string = '' if xlim is None else f'_{xlim[1]}ms'
    with open(os.path.join(results_folder, f'spikedetector_events{xlim_string}.pkl'), 'wb') as spkevents_file:
        pickle.dump(spikedetector_events, spkevents_file)
    plt.savefig(os.path.join(results_folder, f'rasterplot{xlim_string}.pdf'))


def plot_neuron_sample_traces(results_folder, sample_multimeter):
    # plot vm traces of random neuron sample
    sample_events = nest.GetStatus(sample_multimeter)[0]['events']
    all_ids = np.unique(sample_events['senders'])
    fig, ax = plt.subplots(nrows=len(all_ids))
    for i, neuron_id in enumerate(all_ids):
        vms = sample_events['V_m'][sample_events['senders'] == neuron_id]
        times = sample_events['times'][sample_events['senders'] == neuron_id]
        ax[i].plot(times, vms)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, 'neuron_traces.pdf'))


def plot_task_result_bars(result_dicts, axes=None, readoutnames=None, width=0.4, offset=0., color=None, label=None, alpha=1., ecolor=None, **barargs):
    if ecolor is None:
        ecolor = color
    tasknames = list(result_dicts[0].keys())
    if readoutnames is None:
        readoutnames = list(result_dicts[0][tasknames[0]].keys())

    if axes is None:
        fig, axes = plt.subplots(nrows=len(readoutnames))

    if len(readoutnames) == 1:
        axes = [axes]

    taskname_substitutes = {
        'spike_pattern_classification_S1': r'$\mathrm{tcl_1(t)}$',
        'delayed_spike_pattern_classification_S1': r'$\mathrm{tcl_1(t-\Delta t)}$',
        'spike_pattern_classification_S2': r'$\mathrm{tcl_2(t)}$',
        'delayed_spike_pattern_classification_S2': r'$\mathrm{tcl_2(t-\Delta t)}$',
        'xor_spike_pattern': r'$\mathrm{XOR}$',
        'r1/r2': r'$\mathrm{r1/r2}$',
        '(r1-r2)^2': r'$\mathrm{(r1-r2)^2}$',
    }

    readout_substitutes = {
        'l23_exc_nnls': 'layer 2/3 readout',
        'l23_exc_linreg': 'layer 2/3 readout (linear regression)',
        'l5_exc_nnls': 'layer 5 readout',
        'l5_exc_linreg': 'layer 5 readout (linear regression)',
    }

    for readout_id, readout in enumerate(readoutnames):
        bar_data = {}
        errorbar_data = {}
        for old_taskname, new_taskname in taskname_substitutes.items():
            task_results = []
            for results in result_dicts:
                if old_taskname in results.keys():
                    if 'kappa_test' in results[old_taskname][readout].keys():  # spike pattern classification
                        task_results.append(results[old_taskname][readout]['kappa_test'])
                    else:  # firing rate tasks
                        task_results.append(results[old_taskname][readout]['cc_test'])

            bar_data[new_taskname] = np.nanmean(task_results)
            # errorbar_data[new_taskname] = np.nanstd(task_results)
            errorbar_data[new_taskname] = general_utils.standard_error_of_mean(task_results)

        xvals = np.arange(0, len(bar_data.values())) + offset
        # axes[readout_id].bar(x=xvals, height=bar_data.values(), yerr=errorbar_data.values(), capsize=3, ecolor=color, width=width, color=color, label=label)
        # h = np.array(list(bar_data.values()))
        # h += np.random.uniform(0, 0.1, size=len(h))
        # axes[readout_id].bar(x=xvals, height=h, yerr=errorbar_data.values(), capsize=3, ecolor=color, width=width, color=color, label=label, hatch='\\\\\\')
        # axes[readout_id].bar(x=xvals, height=h, yerr=[[0 for _ in errorbar_data.values()], errorbar_data.values()], ecolor=color, capsize=2, width=width, color=color, label=label, **barargs)
        axes[readout_id].errorbar(x=xvals, y=list(bar_data.values()), yerr=[[0 for _ in errorbar_data.values()], list(errorbar_data.values())], ecolor=ecolor, capsize=2, fmt='none', alpha=alpha)
        axes[readout_id].bar(x=xvals, height=bar_data.values(), width=width, color=color, label=label, alpha=alpha, **barargs)
        # axes[readout_id].bar(x=xvals, height=bar_data.values(), yerr=[[0 for _ in errorbar_data.values()], list(errorbar_data.values())], ecolor=color, capsize=2, width=width, color=color, label=label, **barargs)
        ax_title = readout
        if readout in readout_substitutes.keys():
            ax_title = readout_substitutes[readout]
        axes[readout_id].set_title(ax_title)
        axes[readout_id].set_ylabel('performance')

    return axes, list(bar_data.keys())


def plot_histograms_per_pop(data_dict, axes=None, label=None, color=None, density=False, bins=None, binrange=None):
    n_populations = len(list(data_dict.keys()))

    pop_name_translator = {
        'l23_exc': 'L2/3-E',
        'l23_inh': 'L2/3-I',
        'l4_exc': 'L4-E',
        'l4_inh': 'L4-I',
        'l5_exc': 'L5-E',
        'l5_inh': 'L5-I'
    }

    if axes is None:
        fig, axes = plt.subplots(ncols=2, nrows=int(n_populations/2), sharex='all', sharey='all')

    for i, (pop_name, pop_data) in enumerate(sorted(data_dict.items())):
        # indegrees, outdegrees = connection_utils.get_in_and_outdegrees(pop_neurons)
        # degrees_combined = np.array([i+o for (i, o) in zip(indegrees, outdegrees)])

        row = i // 2
        col = i % 2

        if bins is not None:
            hist, bin_edges = np.histogram(pop_data, bins=bins, density=density, range=binrange)
        else:
            hist, bin_edges = np.histogram(pop_data, density=density, range=binrange)
        degrees = bin_edges[:-1] + np.diff(bin_edges)/2
        axes[row, col].plot(degrees, hist, label=label, color=color)

        # axes[row, col].hist(pop_data, label=label, color=color, histtype='step', density=density, bins=bins)
        if pop_name in pop_name_translator.keys():
            title = pop_name_translator[pop_name]
        else:
            title = pop_name
        axes[row, col].set_title(title)

    return axes


def plot_degree_distributions_per_pop(degrees_per_pop, axes=None, label=None, color=None, bins=None, binrange=None):
    axes = plot_histograms_per_pop(degrees_per_pop, axes=axes, density=True, label=label, color=color, bins=bins, binrange=binrange)

    for row_nr, ax_cols in enumerate(axes):
        for col_nr, ax in enumerate(ax_cols):
            ax.set_xlim((0, 300))

        ax_cols[0].set_ylabel('probability')

    axes[-1][0].set_xlabel('indegree + outdegree')
    axes[-1][1].set_xlabel('indegree + outdegree')
    axes[-1][-1].legend()
    plt.tight_layout()

    return axes
