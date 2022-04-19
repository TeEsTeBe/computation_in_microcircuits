"""
Network analysis
----------------------
Standardized functions to evaluate the properties of the different network architectures in a
task-independent manner
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as pl

# NEST
import nest

# internal imports
from fna.tools.visualization import plotting, helper
from fna import tools
from fna.tools.utils import operations
logger = tools.utils.logger.get_logger(__name__)


# ######################################################################################################################
def single_neuron_dcresponse(neuron_gid, input_amplitudes, input_times, spike_list, analogs, plot=True,
                             display=True, save=False):
    """
    Extract relevant data and analyse single neuron fI curves and other measures.

    :param neuron_gid: int, list or tuple with neuron id
    :param input_amplitudes: amplitude values
    :param input_times: times when amplitude changes
    :param plot: bool
    :param display: bool
    :param save: bool
    :return results: dict
    """
    vm_list = analogs[0]

    # extract single response:
    min_spk = spike_list.first_spike_time()
    idx = max(np.where(input_times < min_spk)[0])
    interval = [input_times[idx], input_times[idx + 1]]
    single_spk = spike_list.time_slice(interval[0], interval[1])
    while tools.utils.operations.empty(single_spk.isi()):
        idx += 1
        interval = [input_times[idx + 1], input_times[idx + 2]] # if the first analysis intervals contains no ISIs,
        # look at the next
        single_spk = spike_list.time_slice(interval[0], interval[1])
    single_vm = vm_list.time_slice(interval[0], interval[1])

    if len(analogs) > 1:
        other_analogs = [x for x in analogs[1:]]
        single_analogs = [x.time_slice(interval[0], interval[1]) for x in other_analogs]
    else:
        other_analogs = None

    output_rate = []
    for idx, t in enumerate(input_times):
        if idx >= 1:
            output_rate.append(spike_list.mean_rate(input_times[idx - 1], t))

    isiis = single_spk.isi()[0]
    k = 2  # disregard initial transient
    n = len(isiis)
    if n > k:
        l = []
        for iddx, nn in enumerate(isiis):
            if iddx > k:
                l.append((nn - isiis[iddx - 1]) / (nn + isiis[iddx - 1]))
    else:
        l = 0
    A = np.sum(l) / (n - k - 1)

    if plot:
        fig = pl.figure(figsize=(15, 12))
        fig.suptitle('Neuron {}'.format(neuron_gid))
        ax1 = pl.subplot2grid((10, 3), (0, 0), rowspan=6, colspan=1)
        ax2 = pl.subplot2grid((10, 3), (0, 1), rowspan=6, colspan=1)
        ax3 = pl.subplot2grid((10, 3), (0, 2), rowspan=6, colspan=1)
        ax4 = pl.subplot2grid((10, 3), (7, 0), rowspan=3, colspan=3)

        props = {'xlabel': r'I [pA]', 'ylabel': r'Firing Rate [spikes/s]'}
        plotting.plot_io_curve(input_amplitudes[:-1], output_rate, ax=ax1, display=False, save=False, **props)

        props.update({'xlabel': r'$\mathrm{ISI}$', 'ylabel': r'$\mathrm{ISI} [\mathrm{ms}]$',
                      'title': r'$AI = {0}$'.format(str(A))})
        pr2 = props.copy()
        pr2.update({'inset': {'isi': isiis}})
        plotting.plot_singleneuron_isis(spike_list.isi()[0], ax=ax2, display=False, save=False, **pr2)

        props.update({'xlabel': r'$\mathrm{ISI}_{n} [\mathrm{ms}]$', 'ylabel': r'$\mathrm{ISI}_{n+1} [\mathrm{ms}]$',
                      'title': r'$AI = {0}$'.format(str(A))})
        plotting.recurrence_plot(isiis, ax=ax3, display=False, save=False, **props)

        vm_plot = plotting.AnalogSignalPlots(single_vm, start=interval[0], stop=interval[1])  # interval[0]+1000)
        props = {'xlabel': r'Time [ms]', 'ylabel': r'$V_{m} [\mathrm{mV}]$'}
        neuron_pars = nest.GetStatus([neuron_gid])[0]
        if other_analogs is not None:
            for signal in single_analogs:
                ax4.plot(signal.time_axis(), signal.as_array()[0, :], 'g')
        if 'V_reset' in list(neuron_pars.keys()) and 'V_th' in list(neuron_pars.keys()):
            ax4 = vm_plot.plot_Vm(ax=ax4, with_spikes=True, v_reset=neuron_pars['V_reset'],
                                  v_th=neuron_pars['V_th'], display=False, save=False,
                                  **props)
        else:
            if 'single_spk' in locals():
                spikes = single_spk.spiketrains[single_spk.id_list[0]].spike_times
                ax4.vlines(spikes, ymin=np.min(single_vm.raw_data()[:, 0]), ymax=np.max(single_vm.raw_data()[:, 0]))
            ax4 = vm_plot.plot_Vm(ax=ax4, with_spikes=False, v_reset=None,
                                  v_th=None, display=False, save=False, **props)
        if display:
            pl.show()
        if save:
            assert isinstance(save, str), "Please provide filename"
            fig.savefig(save + str(neuron_gid) + '_SingleNeuron_DCresponse.pdf')

    results = dict(input_amplitudes=input_amplitudes[:-1], input_times=input_times, output_rate=np.array(output_rate),
                isi=spike_list.isi()[0], vm=vm_list.analog_signals[list(vm_list.analog_signals.keys())[0]].signal,
                time_axis=vm_list.analog_signals[list(vm_list.analog_signals.keys())[0]].time_axis(), AI=A)

    idx = np.min(np.where(results['output_rate']))

    results.update({'min_rate': np.min(results['output_rate'][results['output_rate'] > 0.]),
                    'max_rate': np.max(results['output_rate'][results['output_rate'] > 0.])})

    x = np.array(results['input_amplitudes'])
    y = np.array(results['output_rate'])
    iddxs = np.where(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[iddxs], y[iddxs])
    results.update({'fI_slope': slope * 1000., 'I_rh': [results['input_amplitudes'][idx - 1],
                                                             results['input_amplitudes'][idx]]})

    if display:
        logger.info("Rate range for neuron {} = [{}, {}] Hz".format(str(neuron_gid), str(np.min(results['output_rate'][
                                                                                results['output_rate'] > 0.])),
                                                       str(np.max(results['output_rate'][results['output_rate'] >
                                                                                         0.]))))

        logger.info("Rheobase Current for neuron {} in [{}, {}]".format(str(neuron_gid),
                                                                        str(results['input_amplitudes'][idx - 1]),
                                                                        str(results['input_amplitudes'][idx])))

        logger.info("fI Slope for neuron {} = {} Hz/nA [linreg method]".format(neuron_gid, str(slope * 1000.)))

    return results


def single_neuron_ratetransfer(neuron_gid, input_amplitudes, input_times, spike_list, analogs, recordables, plot=True,
                             display=True, save=False):
    """
    Extract relevant data and analyse single neuron fI curves and other measures.

    :param neuron_gid: int, list or tuple with neuron id
    :param input_amplitudes: amplitude values
    :param input_times: times when amplitude changes
    :param plot: bool
    :param display: bool
    :param save: bool
    :return results: dict
    """
    output_rate = []
    output_analogs = {k: [] for k in recordables}

    for idx, t in enumerate(input_times):
        if idx >= 1:
            output_rate.append(spike_list.mean_rate(input_times[idx - 1], t))
            for an_idx, k in enumerate(recordables):
                analog = analogs[an_idx].time_slice(t_start=input_times[idx - 1], t_stop=t)
                output_analogs[k].append(analog.mean(axis=1))

    interval_idx = np.where(np.array(output_rate) > 0.)[0][0]
    analysis_time = [input_times[interval_idx], input_times[interval_idx+1]]
    s_list = spike_list.time_slice(t_start=analysis_time[0], t_stop=analysis_time[1])
    an_list = [x.time_slice(t_start=analysis_time[0], t_stop=analysis_time[1]) for x in analogs]
    results = single_neuron_responses(neuron_gid=neuron_gid, spike_list=s_list,
                                      analogs=an_list, recordables=recordables, plot=plot, display=display, save=save)

    if plot:
        fig = pl.figure(figsize=(10, 4))
        fig.suptitle('Neuron {}'.format(neuron_gid))
        n_vars = len(output_analogs.keys())+1
        axes = [fig.add_subplot(1, n_vars, idx+1) for idx in range(n_vars)]

        props = {'xlabel': r'$\nu_{\mathrm{in}}$ [Hz]', 'ylabel': r'$\nu_{\mathrm{out}}$ [Hz]', 'linewidth': 2,
                 'linestyle': '-', 'color': 'k'}
        plotting.plot_io_curve(np.array(input_amplitudes[:-1])/1000., output_rate, ax=axes[0], display=False,
                                        save=False, **props)

        props = {'linewidth': 2, 'linestyle': '--', 'color': 'r'}
        plotting.plot_io_curve(np.array(input_amplitudes[:-1])/1000., np.array(input_amplitudes[:-1])/1000., ax=axes[0],
                               display=False, save=False, **props)

        for idx, var in enumerate(recordables):
            props = {'xlabel': r'$\nu_{\mathrm{in}}$ [Hz]', 'ylabel': r'$<{}>$'.format(var)}
            plotting.plot_io_curve(np.array(input_amplitudes[:-1])/1000., output_analogs[var], ax=axes[idx+1],
                                   display=False, save=False, **props)
        for ax in axes:
            ax.set_xlim(min(input_amplitudes[:-1])/1000., max(input_amplitudes[:-1])/1000.)
        if display:
            pl.show()
        if save:
            assert isinstance(save, str), "Please provide filename"
            fig.savefig(save + str(neuron_gid) + '_SingleNeuron.pdf')
    return results


def single_neuron_responses(neuron_gid, spike_list, analogs, recordables, plot=True, display=True, save=False):
    """
    Responses of a single neuron to synaptic input.

    :param parameter_set:
    :param plot:
    :param display:
    :param save:
    :return:
    """
    results = dict(rate=0, isis=[0, 0], cv_isi=0, ff=None, vm=[], I_e=[], I_i=[], time_data=[])
    single_neuron_params = nest.GetStatus([neuron_gid])[0]
    results['time_data'] = analogs[0].time_axis()

    results['rate'] = spike_list.mean_rate()
    results['isi'] = spike_list.isi()
    if results['rate']:
        results['cv_isi'] = spike_list.cv_isi(True)[0]
        results['ff'] = spike_list.fano_factor(1.)
    else:
        logger.info("No spikes recorded")

    # Make the next for loop independent of the order of the elements in analog_activity_names.
    # "V_m" has to come before "g_ex" or "g_in".
    idx_Vm = recordables.index('V_m')
    # put V_m first as rest depends on it, and add other indices in unchanged order
    activity_indices = [idx_Vm] + np.delete(np.arange(len(recordables)), idx_Vm).tolist()

    recs = []
    for idx in activity_indices:
        activity_name = recordables[idx]
        activity_obj = analogs[idx]  # e.g. AnalogSignalList

        if list(activity_obj.raw_data()):
            iddds = list(activity_obj.analog_signals.keys())
            if activity_name == 'V_m':
                V_m = activity_obj.analog_signals[int(min(iddds))]
                results['vm'] = activity_obj.analog_signals[int(min(iddds))].signal
                recs.append(activity_name)
            elif activity_name == 'I_ex':
                results['I_e'] = -activity_obj.analog_signals[int(min(iddds))].signal
                # results['I_e'] /= 1000.
                recs.append(activity_name)
            elif activity_name == 'I_in':
                results['I_i'] = -activity_obj.analog_signals[int(min(iddds))].signal  # / 1000.
                recs.append(activity_name)
            elif activity_name == 'g_in':
                if "V_m" not in recordables:
                    raise ValueError("V_m is needed to compute results for g_in. Please make sure that "
                                         "analog_activity_names of the population object contain V_m")
                E_in = single_neuron_params['E_in']
                results['I_i'] = -activity_obj.analog_signals[int(min(iddds))].signal * (results['vm'] - E_in)
                results['I_i'] /= 1000.
                recs.append(activity_name)
            elif activity_name == 'g_ex':
                if "V_m" not in recordables:
                    raise ValueError("V_m is needed to compute results for g_ex. Please make sure that "
                                         "analog_activity_names of the population object contain V_m")
                E_ex = single_neuron_params['E_ex']
                results['I_e'] = -activity_obj.analog_signals[int(min(iddds))].signal * (results['vm'] - E_ex)
                results['I_e'] /= 1000.
                recs.append(activity_name)
            else:
                results[activity_name] = activity_obj.analog_signals[int(min(iddds))].signal
        else:
            logger.info("No recorded analog {0}".format(str(activity_name)))

    if plot:
        fig = pl.figure(figsize=(15, 10))
        ax1 = pl.subplot2grid((10, 10), (0, 0), rowspan=4, colspan=4)
        ax2 = pl.subplot2grid((10, 10), (0, 5), rowspan=4, colspan=5)
        ax3 = pl.subplot2grid((10, 10), (5, 0), rowspan=2, colspan=10)
        ax4 = pl.subplot2grid((10, 10), (8, 0), rowspan=2, colspan=10, sharex=ax3)
        fig.suptitle(r'Single Neuron [{}] Activity'.format(neuron_gid))
        props = {'xlabel': '', 'ylabel': '', 'xticks': [], 'yticks': [], 'yticklabels': '', 'xticklabels': ''}
        ax2.set(**props)
        if list(spike_list.raw_data()):
            ax2.text(0.4, 0.9, r'ACTIVITY REPORT [{}-{} ms]'.format(min(results['time_data']), max(results['time_data'])),
                     color='k', fontsize=14, va='center', ha='center')
            ax2.text(0.4, 0.6, r'- Firing Rate = ${0}$ spikes/s'.format(str(results['rate'])), color='k', fontsize=12,
                     va='center', ha='center')
            ax2.text(0.45, 0.4, r'- $CV_{0} = {1}$'.format('{ISI}', str(results['cv_isi'])), color='k', fontsize=12,
                     va='center', ha='center')
            ax2.text(0.4, 0.2, r'- Fano Factor = ${0}$'.format(str(results['ff'])), color='k', fontsize=12,
                     va='center', ha='center')

            props = {'xlabel': r'ISI', 'ylabel': r'Frequency', 'histtype': 'stepfilled', 'alpha': 1.}
            ax1.set_yscale('log')
            tools.visualization.plotting.plot_histogram(results['isi'], n_bins=10, norm=True, mark_mean=True, ax=ax1, color='b',
                                                        display=False,
                                                        save=False, **props)
            spikes = spike_list.spiketrains[spike_list.id_list[0]].spike_times

        if analogs is not None:
            props2 = {'xlabel': r'Time [ms]', 'ylabel': r'$V_{m} [mV]$'}
            ap = plotting.AnalogSignalPlots(V_m)
            if 'V_reset' in list(single_neuron_params.keys()) and 'V_th' in list(single_neuron_params.keys()):
                ax4 = ap.plot_Vm(ax=ax4, with_spikes=True, v_reset=single_neuron_params['V_reset'],
                                 v_th=single_neuron_params['V_th'], display=False, save=False, **props2)
            else:
                if 'spikes' in locals():
                    ax4.vlines(spikes, ymin=np.min(globals()['V_m'].raw_data()[:, 0]), ymax=np.max(globals()[
                                                                                                       'V_m'].raw_data()[
                                                                                                   :, 0]))
                ax4 = ap.plot_Vm(ax=ax4, with_spikes=False, v_reset=None,
                                 v_th=None, display=False, save=False, **props2)

            ax4.set_xticks(np.linspace(spike_list.t_start, spike_list.t_stop, 5))
            ax4.set_xticklabels([str(x) for x in np.linspace(spike_list.t_start, spike_list.t_stop, 5)])
            if 'I_e' in results and not operations.empty(results['I_e']):
                props = {'xlabel': '', 'xticklabels': '', 'ylabel': r'$I_{\mathrm{syn}} [nA]$'}
                ax3.set(**props)
                ax3.plot(results['time_data'], -results['I_e'] / 1000, 'b', lw=1)
                ax3.plot(results['time_data'], -results['I_i'] / 1000, 'r', lw=1)
                ax3.plot(results['time_data'], (-results['I_e'] - results['I_i']) / 1000, 'gray', lw=1)
            else:
                keys = [n for n in recs if n != 'V_m']
                for k in keys:
                    ax3.plot(results['time_data'], results[k], label=r'$' + k + '$')
                ax3.legend()
        if display:
            pl.show()
        if save:
            assert isinstance(save, str), "Please provide filename"
            fig.savefig(save + str(neuron_gid) + '_SingleNeuron.pdf')
    return results


# def visualize(connection_parameters, network):
#     """
#     Standard visualization of network architecture (structure, weight and delay distributions, etc)
#     :param connection_parameters:
#     :param network:
#     :return:
#     """
#     tp = helper.TopologyPlots(connection_parameters, network, colormap='jet')
#     tp.print_network()
    # tp.compile_weights()
    # tp.plot_spectral_radius()

    # tp.plot_connectivity(synapse_types=None, ax=None, display=True, save=False)
    # tp.plot_weight_histograms(synapse_types=None, ax=None, display=True, save=False)
    #
    # tp.plot_connectivity_delays(synapse_types=None, ax=None, display=True, save=False)
    # tp.plot_delay_histograms(synapse_types=None, ax=None, display=True, save=False)

    # brn_graph = tp.to_graph_object()