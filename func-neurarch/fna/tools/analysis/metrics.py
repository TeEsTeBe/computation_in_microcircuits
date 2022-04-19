"""
========================================================================================================================
Analysis Metrics Module
========================================================================================================================
Collection of analysis and utility functions that are used by other tools
(Note: this documentation is incomplete)

Functions:
------------
ccf 						- fast cross-correlation function, using fft
_dict_max 					- for a dict containing numerical values, return the key for the highest
crosscorrelate 				-
makekernel 					- creates kernel functions for convolution
simple_frequency_spectrum 	- simple calculation of frequency spectrum

========================================================================================================================
"""

import itertools
import time

import numpy as np
from matplotlib import pyplot as pl, pyplot as plt
from scipy import optimize as opt
from sklearn import decomposition as sk, manifold as man, metrics as met

from fna import tools
import fna.tools.visualization.plotting
from fna.tools import check_dependency, network_architect, signals
from fna.tools.visualization import helper as vis_helper


has_pyspike = check_dependency('pyspike')
if has_pyspike:
    import pyspike

logger = tools.utils.logger.get_logger(__name__)

np.seterr(all='ignore')


def compute_spikelist_metrics(spike_list, label, analysis_pars):
    """
    Computes the ISI, firing activity and synchrony statistics for a given spike list.

    :param spike_list: SpikeList object for which the statistics are computed
    :param label: (string) population name or something else
    :param analysis_pars: ParameterSet object containing the analysis parameters

    :return: dictionary with the results for the given label, with statistics as (sub)keys
    """
    ap = analysis_pars
    pars_activity = ap.population_activity
    results = {label: {}}

    results[label].update(compute_isi_stats(spike_list, summary_only=bool(ap.depth % 2 != 0)))

    # Firing activity statistics
    results[label].update(compute_spike_stats(spike_list, time_bin=pars_activity.time_bin,
                                              summary_only=bool(ap.depth % 2 != 0)))

    # Synchrony measures
    if not spike_list.empty():
        if len(np.nonzero(spike_list.mean_rates())[0]) > 10:
            results[label].update(compute_synchrony(spike_list, n_pairs=pars_activity.n_pairs,
                                                    time_bin=pars_activity.time_bin, tau=pars_activity.tau,
                                                    time_resolved=pars_activity.time_resolved, depth=ap.depth))
    return results


def ssa_lifetime(pop_obj, analysis_interval, input_off=1000., display=True):
    """
    Determine the lifetime of self-sustaining activity (specific)
    :param pop_obj:
    :param parameter_set:
    :param input_off: time when the input is turned off
    :param display:
    :return:
    """
    results = dict(ssa={})
    if display:
        logger.info("\nSelf-sustaining Activity Lifetime: ")
    if isinstance(pop_obj, network_architect.Network):
        gids = []
        new_SpkList = signals.spikes.SpikeList([], [], analysis_interval[0], analysis_interval[1],
                                                       np.sum(list(tools.utils.operations.iterate_obj_list(
                                       pop_obj.n_neurons))))
        for ii, n in enumerate(pop_obj.spiking_activity):
            gids.append(n.id_list)
            for idd in n.id_list:
                new_SpkList.append(idd, n.spiketrains[idd])
            results['ssa'].update({str(pop_obj.population_names[ii] + '_ssa'): {'last_spike': n.last_spike_time(),
                                                                                'tau': n.last_spike_time() -
                                                                                       input_off}})
            if display:
                logger.info("- {0} Survival = {1} ms".format(str(pop_obj.population_names[ii]), str(results['ssa'][str(
                    pop_obj.population_names[ii] + '_ssa')]['tau'])))

        results['ssa'].update({'Global_ssa': {'last_spike': new_SpkList.last_spike_time(),
                                              'tau': new_SpkList.last_spike_time() - input_off}})
        if display:
            logger.info("- {0} Survival = {1} ms".format('Global', str(results['ssa']['Global_ssa']['tau'])))

    elif isinstance(pop_obj, network_architect.Population):
        name = pop_obj.name
        spike_list = pop_obj.spiking_activity.spiking_activity
        results['ssa'].update({name + '_ssa': {'last_spike': spike_list.last_spike_time(),
                                               'tau': spike_list.last_spike_time() - input_off}})
        if display:
            logger.info("- {0} Survival = {1} ms".format(str(name), str(results['ssa'][name + '_ssa']['tau'])))
    else:
        raise ValueError("Input must be Network or Population object")

    return results


def analyse_state_matrix(state_matrix, stim_labels=None, epochs=None, label='', plot=True, display=True, save=False):
    """
    Use PCA to peer into the population responses.

    :param state_matrix: state matrix X
    :param stim_labels: stimulus labels (if each sample corresponds to a unique label)
    :param epochs:
    :param label: data label
    :param plot:
    :param display:
    :param save:
    :return results: dimensionality results
    """
    if isinstance(state_matrix, signals.analog.AnalogSignalList):
        state_matrix = state_matrix.as_array()
    assert (isinstance(state_matrix, np.ndarray)), "Activity matrix must be numpy array or AnalogSignalList"
    results = {}

    pca_obj = sk.PCA(n_components=3)
    X_r = pca_obj.fit(state_matrix.T).transform(state_matrix.T)

    if stim_labels is None:
        pca_obj = sk.PCA(n_components=min(state_matrix.shape))
        X = pca_obj.fit_transform(state_matrix.T)
        logger.info("Explained Variance (first 3 components): %s" % str(pca_obj.explained_variance_ratio_[:3]))
        results.update({'dimensionality': compute_dimensionality(state_matrix, pca_obj=pca_obj, display=True)})
        if plot:
            tools.visualization.plotting.plot_dimensionality(results['dimensionality'], pca_obj, X, data_label=label, display=display, save=save)
        if epochs is not None:
            for epoch_label, epoch_time in list(epochs.items()):
                resp = state_matrix[:, int(epoch_time[0]):int(epoch_time[1])]
                results.update({epoch_label: {}})
                results[epoch_label].update(analyse_state_matrix(resp, epochs=None, label=epoch_label, plot=False,
                                                                 display=False, save=False))
    else:
        logger.info("Explained Variance (first 3 components): %s" % str(pca_obj.explained_variance_ratio_))
        if not isinstance(stim_labels, dict):
            label_seq = np.array(list(tools.utils.operations.iterate_obj_list(stim_labels)))
            n_elements = np.unique(label_seq)
            if plot:
                fig1 = pl.figure()
                ax1 = fig1.add_subplot(111)
                tools.visualization.plotting.plot_matrix(state_matrix, stim_labels, ax=ax1, display=False, save=False)
                fig2 = pl.figure()
                fig2.clf()
                exp_var = [round(n, 2) for n in pca_obj.explained_variance_ratio_]
                fig2.suptitle(r'${0} - PCA (var = {1})$'.format(str(label), str(exp_var)),
                              fontsize=20)

                ax2 = fig2.add_subplot(111, projection='3d')
                colors_map = vis_helper.get_cmap(N=len(n_elements), cmap='Paired')
                ax2.set_xlabel(r'$PC_{1}$')
                ax2.set_ylabel(r'$PC_{2}$')
                ax2.set_zlabel(r'$PC_{3}$')

                ccs = [colors_map(ii) for ii in range(len(n_elements))]
                for color, index, lab in zip(ccs, n_elements, n_elements):
                    # locals()['sc_{0}'.format(str(index))] = \
                        ax2.scatter(X_r[np.where(np.array(list(itertools.chain(
                        label_seq))) == index)[0], 0], X_r[np.where(
                        np.array(list(itertools.chain(label_seq))) == index)[
                                                               0], 1], X_r[np.where(
                        np.array(list(itertools.chain(label_seq))) == index)[0], 2],
                                                                        s=50, c=color, label=lab)
                # scatters = [locals()['sc_{0}'.format(str(ind))] for ind in n_elements]
                # pl.legend(tuple(scatters), tuple(n_elements))
                pl.legend(loc=0)#, handles=scatters)

                if display:
                    pl.show(block=False)
                if save:
                    fig1.savefig(save + 'state_matrix_{0}.pdf'.format(label))
                    fig2.savefig(save + 'pca_representation_{0}.pdf'.format(label))
        else:
            if plot:
                fig1 = pl.figure()
                ax = fig1.add_subplot(111, projection='3d')
                ax.plot(X_r[:, 0], X_r[:, 1], X_r[:, 2], color='r', lw=2)
                ax.set_title(label + r'$ - (3PCs) $= {0}$'.format(
                    str(round(np.sum(pca_obj.explained_variance_ratio_[:3]), 1))))
                ax.grid()
                if display:
                    pl.show(False)
                if save:
                    fig1.savefig(save + 'pca_representation_{0}.pdf'.format(label))
    return results


# def evaluate_encoding(enc_layer, decoding_pars, encoding_pars, analysis_interval, input_signal, method=None,
#                       plot=True, display=True, save=False):
#     """
#     Determine the quality of the encoding method (if there are encoders), by reading out the state of the encoders
#     and training it to reconstruct the input signal. *needs testing
#
#     :param enc_layer:
#     :param parameter_set:
#     :param analysis_interval:
#     :param input_signal:
#     :param plot:
#     :param display:
#     :param save:
#     :return:
#     """
#     assert (isinstance(analysis_interval, list)), "Incorrect analysis_interval"
#     results = dict()
#     for idx, n_enc in enumerate(enc_layer.encoders):
#         # new_pars = parameters.ParameterSet(parameters.copy_dict(parameter_set.as_dict()))
#         # new_pars.kernel_pars.data_prefix = 'Input Encoder {0}'.format(n_enc.name)
#         # results['input_activity_{0}'.format(str(idx))] = characterize_population_activity(n_enc,
#         #                                                                   parameter_set=new_pars,
#         #                                                                   analysis_interval=analysis_interval,
#         #                                                                   epochs=None, time_bin=1., complete=False,
#         #                                                                   time_resolved=False, color_map='colorwarm',
#         #                                                                   plot=plot, display=display, save=save)
#
#         if isinstance(n_enc.spiking_activity, signals.spikes.SpikeList) and not n_enc.spiking_activity.empty():
#             inp_spikes = n_enc.spiking_activity.time_slice(analysis_interval[0], analysis_interval[1])
#             tau = decoding_pars.state_extractor.filter_tau
#             n_input_neurons = np.sum(encoding_pars.encoder.n_neurons)
#             if n_enc.decoding_layer is not None:
#                 inp_responses = n_enc.decoding_layer.extract_activity(start=analysis_interval[0],
#                                                                       stop=analysis_interval[1], save=False,
#                                                                       reset=False)[0]
#                 inp_readout_pars = tools.utils.operations.copy_dict(n_enc.decoding_layer.decoding_pars.readout[0])
#             else:
#                 inp_responses = inp_spikes.filter_spiketrains(dt=input_signal.dt,
#                                                               tau=tau, start=analysis_interval[0],
#                                                               stop=analysis_interval[1], N=n_input_neurons)
#                 inp_readout_pars = tools.utils.operations.copy_dict(decoding_pars.readout[0],
#                                                                     {'label': 'InputNeurons',
#                                                  'algorithm': decoding_pars.readout[0]['algorithm'][0]})
#
#             inp_readout = Readout(parameters.ParameterSet(inp_readout_pars))
#             analysis_signal = input_signal.time_slice(analysis_interval[0], analysis_interval[1])
#             inp_readout.train(inp_responses, analysis_signal.as_array())
#             inp_readout.test(inp_responses)
#             perf = inp_readout.measure_performance(analysis_signal.as_array(), method)
#
#             input_out = InputSignal()
#             input_out.load_signal(inp_readout.output.T, dt=input_signal.dt, onset=analysis_interval[0],
#                                   inherit_from=analysis_signal)
#
#             if plot:
#                 figure2 = pl.figure()
#                 figure2.suptitle(r'MAE = {0}'.format(str(perf['raw']['MAE'])))
#                 ax21 = figure2.add_subplot(211)
#                 ax22 = figure2.add_subplot(212, sharex=ax21)
#                 InputPlots(input_obj=analysis_signal).plot_input_signal(ax22, save=False, display=False)
#                 ax22.set_color_cycle(None)
#                 InputPlots(input_obj=input_out).plot_input_signal(ax22, save=False, display=False)
#                 ax22.set_ylim([analysis_signal.base - 10., analysis_signal.peak + 10.])
#                 inp_spikes.raster_plot(with_rate=False, ax=ax21, save=False, display=False)
#                 if display:
#                     pl.show(block=False)
#                 if save:
#                     figure2.savefig(save + '_EncodingQuality.pdf')
#     return results


# def analyse_state_divergence(parameter_set, net, clone, plot=True, display=True, save=False):
#     """
#     Analyse how responses from net and clone diverge (primarily for perturbation analysis).
#
#     :param parameter_set:
#     :param net: Network object
#     :param clone: Network object
#     :param plot:
#     :param display:
#     :param save:
#     :return:
#     """
#     results = dict()
#     pop_idx = net.population_names.index(parameter_set.kernel_pars.perturb_population)
#     start = parameter_set.kernel_pars.transient_t
#     stop = parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t
#     activity_time_vector = np.arange(parameter_set.kernel_pars.transient_t,
#                                      parameter_set.kernel_pars.sim_time + parameter_set.kernel_pars.transient_t,
#                                      parameter_set.kernel_pars.resolution)
#     perturbation_time = parameter_set.kernel_pars.perturbation_time + parameter_set.kernel_pars.transient_t
#     observation_time = max(activity_time_vector) - perturbation_time
#
#     if not tools.utils.operations.empty(net.populations[pop_idx].spiking_activity.spiketrains):
#         time_vec = net.populations[pop_idx].spiking_activity.time_axis(1)[:-1]
#         perturbation_idx = np.where(time_vec == perturbation_time)
#         rate_native = net.populations[pop_idx].spiking_activity.firing_rate(1, average=True)
#         rate_clone = clone.populations[pop_idx].spiking_activity.firing_rate(1, average=True)
#
#         # binary_native = net.populations[pop_idx].spiking_activity.compile_binary_response_matrix(
#         # 		parameter_set.kernel_pars.resolution, start=parameter_set.kernel_pars.transient_t,
#         # 		stop=parameter_set.kernel_pars.sim_time+parameter_set.kernel_pars.transient_t,
#         # 		N=net.populations[pop_idx].size, display=True)
#         # binary_clone = clone.populations[pop_idx].spiking_activity.compile_binary_response_matrix(
#         # 		parameter_set.kernel_pars.resolution, start=parameter_set.kernel_pars.transient_t,
#         # 		stop=parameter_set.kernel_pars.sim_time+parameter_set.kernel_pars.transient_t,
#         # 		N=clone.populations[pop_idx].size, display=True)
#
#         r_cor = []
#         # hamming = []
#         for idx, t in enumerate(time_vec):
#             if not tools.utils.operations.empty(np.corrcoef(rate_native[:idx], rate_clone[:idx])) and np.corrcoef(rate_native[:idx],
#                                                                                               rate_clone[:idx])[
#                 0, 1] != np.nan:
#                 r_cor.append(np.corrcoef(rate_native[:idx], rate_clone[:idx])[0, 1])
#             else:
#                 r_cor.append(0.)
#         # binary_state_diff = binary_native[:, idx] - binary_clone[:, idx]
#         # if not sg.empty(np.nonzero(binary_state_diff)[0]):
#         # 	hamming.append(float(len(np.nonzero(binary_state_diff)[0]))/float(net.populations[pop_idx].size))
#         # else:
#         # 	hamming.append(0.)
#
#         results['rate_native'] = rate_native
#         results['rate_clone'] = rate_clone
#         results['rate_correlation'] = np.array(r_cor)
#         # results['hamming_distance'] = np.array(hamming)
#         results['lyapunov_exponent'] = {}
#     if not tools.utils.operations.empty(net.populations[pop_idx].decoding_layer.activity):
#         responses_native = net.populations[pop_idx].decoding_layer.activity
#         responses_clone = clone.populations[pop_idx].decoding_layer.activity
#         response_vars = parameter_set.decoding_pars.state_extractor.state_variable
#         logger.info("\n Computing state divergence: ")
#         labels = []
#         for resp_idx, n_response in enumerate(responses_native):
#             logger.info("\t- State variable {0}".format(str(response_vars[resp_idx])))
#             response_length = len(n_response.time_axis())
#             distan = []
#             for t in range(response_length):
#                 distan.append(
#                     distance.euclidean(n_response.as_array()[:, t], responses_clone[resp_idx].as_array()[:, t]))
#                 if display:
#                     vis_helper.progress_bar(float(t) / float(response_length))
#
#             results['state_{0}'.format(str(response_vars[resp_idx]))] = np.array(distan)
#             labels.append(str(response_vars[resp_idx]))
#
#             if np.array(distan).any():
#                 initial_distance = distan[min(np.where(np.array(distan) > 0.0)[0])]
#             else:
#                 initial_distance = 0.
#             final_distance = distan[-1]
#             lyapunov = (np.log(final_distance) / observation_time) - np.log(initial_distance) / observation_time
#             logger.info("Lyapunov Exponent = {0}".format(lyapunov))
#             results['lyapunov_exponent'].update({response_vars[resp_idx]: lyapunov})
#
#     if plot:
#         if not tools.utils.operations.empty(net.populations[pop_idx].spiking_activity.spiketrains):
#             fig = pl.figure()
#             fig.suptitle(r'$LE = {0}$'.format(str(list(results['lyapunov_exponent'].items()))))
#             ax1a = pl.subplot2grid((14, 1), (0, 0), rowspan=8, colspan=1)
#             ax1b = ax1a.twinx()
#
#             ax2a = pl.subplot2grid((14, 1), (8, 0), rowspan=2, colspan=1, sharex=ax1a)
#             ax2b = ax2a.twinx()
#
#             ax3 = pl.subplot2grid((14, 1), (10, 0), rowspan=2, colspan=1, sharex=ax1a)
#             ax4 = pl.subplot2grid((14, 1), (12, 0), rowspan=2, colspan=1, sharex=ax1a)
#
#             rp1 = SpikePlots(net.populations[pop_idx].spiking_activity, start, stop)
#             rp2 = SpikePlots(clone.populations[pop_idx].spiking_activity, start, stop)
#
#             plot_props1 = {'xlabel': 'Time [ms]', 'ylabel': 'Neuron', 'color': 'red', 'linewidth': 0.5,
#                            'linestyle': '-'}
#             plot_props2 = {'xlabel': 'Time [ms]', 'ylabel': 'Neuron', 'color': 'k', 'linewidth': 0.5,
#                            'linestyle': '-'}
#             rp1.dot_display(ax=[ax1a, ax2a], with_rate=True, default_color='red', display=False, save=False,
#                             **plot_props1)
#             rp2.dot_display(ax=[ax1b, ax2b], with_rate=True, default_color='k', display=False, save=False,
#                             **plot_props2)
#
#             ax3.plot(time_vec, r_cor, '-b')
#             ax3.set_ylabel('CC')
#
#             if not tools.utils.operations.empty(net.populations[pop_idx].decoding_layer.activity):
#                 for lab in labels:
#                     ax4.plot(activity_time_vector, results['state_{0}'.format(lab)][:len(activity_time_vector)],
#                              label=lab)
#                 ax4.set_ylabel(r'$d_{E}$')
#
#             # mark perturbation time
#             yrange_1 = np.arange(ax1a.get_ylim()[0], ax1a.get_ylim()[1], 1)
#             ax1a.plot(perturbation_time * np.ones_like(yrange_1), yrange_1, 'r--')
#             yrange_2 = np.arange(ax2a.get_ylim()[0], ax2a.get_ylim()[1], 1)
#             ax2a.plot(perturbation_time * np.ones_like(yrange_2), yrange_2, 'r--')
#             yrange_3 = np.arange(ax3.get_ylim()[0], ax3.get_ylim()[1], 1)
#             ax3.plot(perturbation_time * np.ones_like(yrange_3), yrange_3, 'r--')
#             if display:
#                 pl.show(False)
#             if save:
#                 assert isinstance(save, str), "Please provide filename"
#                 fig.savefig(save + 'LE_analysis.pdf')
#
#         if not tools.utils.operations.empty(net.populations[pop_idx].decoding_layer.activity):
#             fig2 = pl.figure()
#             ax4 = fig2.add_subplot(211)
#             ax5 = fig2.add_subplot(212, sharex=ax4)
#             for lab in labels:
#                 ax4.plot(activity_time_vector, results['state_{0}'.format(lab)][:len(activity_time_vector)], label=lab)
#             ax4.set_ylabel(r'$d_{E}$')
#
#             if 'hamming_distance' in list(results.keys()):
#                 ax5.plot(time_vec, results['hamming_distance'], c='g')
#                 ax5.set_ylabel(r'$d_{H}$')
#
#             yrange_4 = np.arange(ax4.get_ylim()[0], ax4.get_ylim()[1], 1)
#             ax4.plot(perturbation_time * np.ones_like(yrange_4), yrange_4, 'r--')
#             yrange_5 = np.arange(ax5.get_ylim()[0], ax5.get_ylim()[1], 1)
#             ax5.plot(perturbation_time * np.ones_like(yrange_5), yrange_5, 'r--')
#
#             ax4.set_xlabel(r'')
#             # ax4.set_xticklabels([])
#             ax4.set_xlim(np.min(activity_time_vector), np.max(activity_time_vector))
#             ax4.legend(loc=0)
#             ax5.set_xlabel(r'Time [ms]')
#
#             if display:
#                 pl.show(False)
#             if save:
#                 assert isinstance(save, str), "Please provide filename"
#                 fig2.savefig(save + 'state_divergence.pdf')
#     return results


def get_state_rank(state_matrix):
    """
    Calculates the rank of all state matrices.
    :return:
    """
    return np.linalg.matrix_rank(state_matrix)


def ccf(x, y, axis=None):
    """
    Fast cross correlation function based on fft.

    Computes the cross-correlation function of two series.
    Note that the computations are performed on anomalies (deviations from
    average).
    Returns the values of the cross-correlation at different lags.

    Parameters
    ----------
    x, y : 1D MaskedArrays
        The two input arrays.
    axis : integer, optional
        Axis along which to compute (0 for rows, 1 for cols).
        If `None`, the array is flattened first.

    Examples
    --------
    >> z = np.arange(5)
    >> ccf(z,z)
    array([  3.90798505e-16,  -4.00000000e-01,  -4.00000000e-01,
            -1.00000000e-01,   4.00000000e-01,   1.00000000e+00,
             4.00000000e-01,  -1.00000000e-01,  -4.00000000e-01,
            -4.00000000e-01])
    """
    assert x.ndim == y.ndim, "Inconsistent shape !"
    if axis is None:
        if x.ndim > 1:
            x = x.ravel()
            y = y.ravel()
        npad = x.size + y.size
        xanom = (x - x.mean(axis=None))
        yanom = (y - y.mean(axis=None))
        Fx = np.fft.fft(xanom, npad, )
        Fy = np.fft.fft(yanom, npad, )
        iFxy = np.fft.ifft(Fx.conj() * Fy).real
        varxy = np.sqrt(np.inner(xanom, xanom) * np.inner(yanom, yanom))
    else:
        npad = x.shape[axis] + y.shape[axis]
        if axis == 1:
            if x.shape[0] != y.shape[0]:
                raise ValueError("Arrays should have the same length!")
            xanom = (x - x.mean(axis=1)[:, None])
            yanom = (y - y.mean(axis=1)[:, None])
            varxy = np.sqrt((xanom * xanom).sum(1) *
                            (yanom * yanom).sum(1))[:, None]
        else:
            if x.shape[1] != y.shape[1]:
                raise ValueError("Arrays should have the same width!")
            xanom = (x - x.mean(axis=0))
            yanom = (y - y.mean(axis=0))
            varxy = np.sqrt((xanom * xanom).sum(0) * (yanom * yanom).sum(0))
        Fx = np.fft.fft(xanom, npad, axis=axis)
        Fy = np.fft.fft(yanom, npad, axis=axis)
        iFxy = np.fft.ifft(Fx.conj() * Fy, n=npad, axis=axis).real
    # We just turn the lags into correct positions:
    iFxy = np.concatenate((iFxy[len(iFxy) // 2:len(iFxy)],
                           iFxy[0:len(iFxy) // 2]))
    return iFxy / varxy


def lag_ix(x, y):
    """
    Calculate lag position at maximal correlation
    :param x:
    :param y:
    :return:
    """
    corr = np.correlate(x, y, mode='full')
    pos_ix = np.argmax(np.abs(corr))
    lag_ix = pos_ix - (corr.size - 1) // 2
    return lag_ix


def cross_correlogram(x, y, max_lag=100., dt=0.1, plot=True):
    """
    Returns the cross-correlogram of x and y
    :param x:
    :param y:
    :param max_lag:
    :return:
    """
    corr = np.correlate(x, y, 'full')
    pos_ix = np.argmax(np.abs(corr))
    maxlag = (corr.size - 1) // 2
    lag = np.arange(-maxlag, maxlag + 1) * dt
    cutoff = [np.where(lag == -max_lag), np.where(lag == max_lag)]

    if plot:
        fig, ax = plt.subplots()
        ax.plot(lag, corr, lw=1)
        ax.set_xlim(lag[cutoff[0]], lag[cutoff[1]])
        ax.axvline(x=lag[pos_ix], ymin=np.min(corr), ymax=np.max(corr), linewidth=1.5, color='c')
        plt.show()
    return lag, corr


def simple_frequency_spectrum(x):
    """
    Simple frequency spectrum.

    Very simple calculation of frequency spectrum with no detrending,
    windowing, etc, just the first half (positive frequency components) of
    abs(fft(x))

    Parameters
    ----------
    x : array_like
        The input array, in the time-domain.

    Returns
    -------
    spec : array_like
        The frequency spectrum of `x`.

    """
    spec = np.absolute(np.fft.fft(x))
    spec = spec[:len(x) // 2]  # take positive frequency components
    spec /= len(x)  # normalize
    spec *= 2.0  # to get amplitudes of sine components, need to multiply by 2
    spec[0] /= 2.0  # except for the dc component
    return spec


def euclidean_distance(pos_1, pos_2, N=None):
    """
    Function to calculate the euclidian distance between two positions

    :param pos_1:
    :param pos_2:
    :param N:
    :return:
    """
    # If N is not None, it means that we are dealing with a toroidal space,
    # and we have to take the min distance on the torus.
    if N is None:
        dx = pos_1[0] - pos_2[0]
        dy = pos_1[1] - pos_2[1]
    else:
        dx = np.minimum(abs(pos_1[0] - pos_2[0]), N - (abs(pos_1[0] - pos_2[0])))
        dy = np.minimum(abs(pos_1[1] - pos_2[1]), N - (abs(pos_1[1] - pos_2[1])))
    return np.sqrt(dx * dx + dy * dy)


def rescale_signal(val, out_min, out_max):
    """
    Rescale a signal to a new range
    :param val: original signal (as a numpy array)
    :param out_min: new minimum
    :param out_max: new maximum
    :return:
    """
    in_min = np.min(val)
    in_max = np.max(val)
    return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))


def autocorrelation_function(x):
    """
    Determine the autocorrelation of signal x

    :param x:
    :return:
    """

    n = len(x)
    data = np.asarray(x)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
        return acf_lag  # round(acf_lag, 3)

    x = np.arange(n)  # Avoiding lag 0 calculation
    acf_coeffs = list(map(r, x))
    return acf_coeffs


def pairwise_dist(matrix):
    return np.tril(met.pairwise_distances(matrix))


def get_total_counts(spike_list, time_bin=50.):
    """
    Determines the total spike counts for neurons with consecutive nonzero counts in bins of the specified size
    :param spike_list: SpikeList object
    :param time_bin: bin width
    :return ctr: number of neurons complying
    :return total_counts: spike count array
    """
    assert isinstance(spike_list, signals.spikes.SpikeList), "Input must be SpikeList object"

    total_counts = []
    ctr = 0
    neuron_ids = []
    for n_train in spike_list.spiketrains:
        tmp = spike_list.spiketrains[n_train].time_histogram(time_bin=time_bin, normalized=False, binary=True)
        if np.mean(tmp) == 1:
            neuron_ids.append(n_train)
            ctr += 1
    logger.info("{0} neurons have nonzero spike counts in bins of size {1}".format(str(ctr), str(time_bin)))
    total_counts1 = []
    for n_id in neuron_ids:
        counts = spike_list.spiketrains[n_id].time_histogram(time_bin=time_bin, normalized=False, binary=False)
        total_counts1.append(counts)
    total_counts.append(total_counts1)
    total_counts = np.array(list(itertools.chain(*total_counts)))

    return neuron_ids, total_counts


def cross_trial_cc(total_counts, display=True):
    """

    :param total_counts:
    :return:
    """
    if display:
        logger.info("Computing autocorrelations..")
    units = total_counts.shape[0]

    r = []
    for nn in range(units):
        if display:
            vis_helper.progress_bar(float(nn) / float(units))
        rr = autocorrelation_function(total_counts[nn, :])
        if not np.isnan(np.mean(rr)):
            r.append(rr)  # [1:])

    return np.array(r)


def acc_function(x, a, b, tau):
    """
    Generic exponential function (to use whenever we want to fit an exponential function to data)
    :param x:
    :param a:
    :param b:
    :param tau: decay time constant
    :return:
    """
    return a * (np.exp(-x / tau) + b)


def err_func(params, x, y, func):
    """
    Error function for model fitting

    Parameters
    ----------
    params : tuple
        A tuple with the parameters of `func` according to their order of input

    x : float array
        An independent variable.

    y : float array
        The dependent variable.

    func : function
        A function with inputs: `(x, *params)`

    Returns
    -------
    The marginals of the fit to x/y given the params
    """
    return y - func(x, *params)


def check_signal_dimensions(input_signal, target_signal):
    """
    Raise error if dimensionalities of signals don't match

    :param input_signal: array
    :param target_signal: array
    :return:
    """
    if input_signal.shape != target_signal.shape:
        raise RuntimeError("Input shape (%s) and target_signal shape (%s) should be the same." % (input_signal.shape,
                                                                                                  target_signal.shape))


def compute_isi_stats(spike_list, summary_only=True):
    """
    Compute all relevant isi metrics
    :param spike_list: SpikeList object
    :param summary_only: bool - store only the summary statistics or all the data (memory!)
    :return: dictionary with all the relevant data
    """
    logger.info("Analysing inter-spike intervals...")
    t_start = time.time()
    results = dict()

    results['cvs'] = spike_list.cv_isi(float_only=True)
    results['lvs'] = spike_list.local_variation()
    results['lvRs'] = spike_list.local_variation_revised(float_only=True)
    results['ents'] = spike_list.isi_entropy(float_only=True)
    results['iR'] = spike_list.instantaneous_regularity(float_only=True)
    results['cvs_log'] = spike_list.cv_log_isi(float_only=True)
    results['isi_5p'] = spike_list.isi_5p(float_only=True)
    results['ai'] = spike_list.adaptation_index(float_only=True)

    if not summary_only:
        results['isi'] = np.array(list(itertools.chain(*spike_list.isi())))
    else:
        results['isi'] = []
        cvs = results['cvs']
        lvs = results['lvs']
        lvRs = results['lvRs']
        H = results['ents']
        iRs = results['iR']
        cvs_log = results['cvs_log']
        isi_5p = results['isi_5p']
        ai = results['ai']

        results['cvs'] = (np.mean(cvs), np.var(cvs))
        results['lvs'] = (np.mean(lvs), np.var(lvs))
        results['lvRs'] = (np.mean(lvRs), np.var(lvRs))
        results['ents'] = (np.mean(H), np.var(H))
        results['iR'] = (np.mean(iRs), np.var(iRs))
        results['cvs_log'] = (np.mean(cvs_log), np.var(cvs_log))
        results['isi_5p'] = (np.mean(isi_5p), np.var(isi_5p))
        results['ai'] = (np.mean(ai), np.var(ai))

    logger.info("Elapsed Time: {0} s".format(str(round(time.time() - t_start, 3))))

    return results


def compute_spike_stats(spike_list, time_bin=50., summary_only=False, display=False):
    """
    Compute relevant statistics on population firing activity (f. rates, spike counts)
    :param spike_list: SpikeList object
    :param time_bin: float - bin width to determine spike counts
    :param summary_only: bool - store only the summary statistics or all the data (memory!)
    :param display: bool - display progress / time
    :return: dictionary with all the relevant data
    """
    if display:
        logger.info("\nAnalysing spiking activity...")
        t_start = time.time()
    results = {}
    rates = np.array(spike_list.mean_rates())
    rates = rates[~np.isnan(rates)]
    counts = spike_list.spike_counts(dt=time_bin, normalized=False, binary=False)
    ffs = np.array(spike_list.fano_factors(time_bin))
    if summary_only:
        results['counts'] = (np.mean(counts[~np.isnan(counts)]), np.var(counts[~np.isnan(counts)]))
        results['mean_rates'] = (np.mean(rates), np.var(rates))
        results['ffs'] = (np.mean(ffs[~np.isnan(ffs)]), np.var(ffs[~np.isnan(ffs)]))
        results['corrected_rates'] = (np.mean(rates[np.nonzero(rates)[0]]), np.std(rates[np.nonzero(rates)[0]]))
    else:
        results['counts'] = counts
        results['mean_rates'] = rates
        results['corrected_rates'] = rates[np.nonzero(rates)[0]]
        results['ffs'] = ffs[~np.isnan(ffs)]
        results['spiking_neurons'] = spike_list.id_list
    if display:
        logger.info("Elapsed Time: {0} s".format(str(round(time.time() - t_start, 3))))
    return results


def compute_synchrony(spike_list, n_pairs=500, time_bin=1., tau=20., time_resolved=False, display=True, depth=4):
    """
    Apply various metrics of spike train synchrony
    Note: Has dependency on PySpike package.

    :param spike_list: SpikeList object
    :param n_pairs: number of neuronal pairs to consider in the pairwise correlation measures
    :param time_bin: time_bin (for pairwise correlations)
    :param tau: time constant (for the van Rossum distance)
    :param time_resolved: bool - perform time-resolved synchrony analysis (PySpike)
    :param summary_only: bool - retrieve only a summary of the results
    :param complete: bool - use all metrics or only the ccs (due to computation time, memory)
    :param display: bool - display elapsed time message
    :return results: dict
    """
    if display:
        logger.info("\nAnalysing spike synchrony...")
        t_start = time.time()

    if has_pyspike:
        spike_trains = signals.to_pyspike(spike_list)
    results = dict()

    if time_resolved and has_pyspike:
        results['SPIKE_sync_profile'] = pyspike.spike_sync_profile(spike_trains)
        results['ISI_profile'] = pyspike.isi_profile(spike_trains)
        results['SPIKE_profile'] = pyspike.spike_profile(spike_trains)

    if depth == 1 or depth == 3:
        results['ccs_pearson'] = spike_list.pairwise_pearson_corrcoeff(n_pairs, time_bin=time_bin, all_coef=False)
        ccs = spike_list.pairwise_cc(n_pairs, time_bin=time_bin)
        results['ccs'] = (np.mean(ccs), np.var(ccs))

        if depth >= 3:
            results['d_vp'] = spike_list.distance_victorpurpura(n_pairs, cost=0.5)
            results['d_vr'] = np.mean(spike_list.distance_van_rossum(tau=tau))
            if has_pyspike:
                results['ISI_distance'] = pyspike.isi_distance(spike_trains)
                results['SPIKE_distance'] = pyspike.spike_distance(spike_trains)
                results['SPIKE_sync_distance'] = pyspike.spike_sync(spike_trains)
    else:
        results['ccs_pearson'] = spike_list.pairwise_pearson_corrcoeff(n_pairs, time_bin=time_bin, all_coef=True)
        results['ccs'] = spike_list.pairwise_cc(n_pairs, time_bin=time_bin)

        if depth >= 3:
            results['d_vp'] = spike_list.distance_victorpurpura(n_pairs, cost=0.5)
            results['d_vr'] = spike_list.distance_van_rossum(tau=tau)
            if has_pyspike:
                results['ISI_distance_matrix'] = pyspike.isi_distance_matrix(spike_trains)
                results['SPIKE_distance_matrix'] = pyspike.spike_distance_matrix(spike_trains)
                results['SPIKE_sync_matrix'] = pyspike.spike_sync_matrix(spike_trains)
                results['ISI_distance'] = pyspike.isi_distance(spike_trains)
                results['SPIKE_distance'] = pyspike.spike_distance(spike_trains)
                results['SPIKE_sync_distance'] = pyspike.spike_sync(spike_trains)

    if display:
        logger.info("Elapsed Time: {0} s".format(str(round(time.time() - t_start, 3))))
    return results


def compute_analog_stats(population, parameter_set, variable_names, analysis_interval=None, plot=False):
    """
    Extract, analyse and store analog data, such as the mean membrane potential and amplitudes of E/I synaptic currents
    If a single neuron is being recorded, only this neuron's activity will be stored, otherwise, this function
    computes the distributions of the relevant variables over the recorded neurons

    :param population: Population object
    :param parameter_set: full ParameterSet object
    :param variable_names: names of the variables of interest.  [V_m, g_ex, g_in, I_in, I_ex] (depend on the
    recordable variables of the specific neuron model used)
    :param analysis_interval: time interval to analyse
    :param plot: bool
    :return results: dict
    """
    results = dict()
    pop_idx = parameter_set.net_pars.pop_names.index(population.name)
    if not population.analog_activity:
        results['recorded_neurons'] = []
        logger.info("No analog variables recorded from {0}".format(str(population.name)))
        return results
    else:
        analog_vars = dict()
        if isinstance(population.analog_activity, list):
            for idx, nn in enumerate(variable_names):
                analog_vars[nn] = population.analog_activity[idx]
                assert isinstance(analog_vars[nn],
                                  signals.analog.AnalogSignalList), "Analog activity should be saved as " \
                                                                         "AnalogSignalList"
        else:
            analog_vars[variable_names[0]] = population.analog_activity

        if plot:
            # pick one neuron to look at its signals (to plot)
            single_idx = np.random.permutation(analog_vars[variable_names[0]].id_list())[0]
            reversals = []

        # store the ids of the recorded neurons
        results['recorded_neurons'] = analog_vars[variable_names[0]].id_list()

        for idx, nn in enumerate(variable_names):
            if analysis_interval is not None:
                analog_vars[nn] = analog_vars[nn].time_slice(analysis_interval[0], analysis_interval[1])

            if plot and 'E_{0}'.format(nn[-2:]) in parameter_set.net_pars.neuron_pars[pop_idx]:
                reversals.append(parameter_set.net_pars.neuron_pars[pop_idx]['E_{0}'.format(nn[-2:])])

            if len(results['recorded_neurons']) > 1:
                results['mean_{0}'.format(nn)] = analog_vars[nn].mean(axis=1)
                results['std_{0}'.format(nn)] = analog_vars[nn].std(axis=1)

        if len(results['recorded_neurons']) > 1:
            results['mean_I_ex'] = []
            results['mean_I_in'] = []
            results['EI_CC'] = []
            for idxx, nnn in enumerate(results['recorded_neurons']):
                for idx, nn in enumerate(variable_names):
                    analog_vars['signal_' + nn] = analog_vars[nn].analog_signals[nnn].signal
                if ('signal_V_m' in analog_vars) and ('signal_g_ex' in analog_vars) and ('signal_g_in' in analog_vars):
                    E_ex = parameter_set.net_pars.neuron_pars[pop_idx]['E_ex']
                    E_in = parameter_set.net_pars.neuron_pars[pop_idx]['E_in']
                    E_current = analog_vars['signal_g_ex'] * (analog_vars['signal_V_m'] - E_ex)
                    E_current /= 1000.
                    I_current = analog_vars['signal_g_in'] * (analog_vars['signal_V_m'] - E_in)
                    I_current /= 1000.
                    results['mean_I_ex'].append(np.mean(E_current))
                    results['mean_I_in'].append(np.mean(I_current))
                    cc = np.corrcoef(E_current, I_current)
                    results['EI_CC'].append(np.unique(cc[cc != 1.]))
                elif ('signal_I_ex' in analog_vars) and ('signal_I_in' in analog_vars):
                    results['mean_I_ex'].append(
                        np.mean(analog_vars['signal_I_ex']) / 1000.)  # /1000. to get results in nA
                    results['mean_I_in'].append(np.mean(analog_vars['signal_I_in']) / 1000.)
                    cc = np.corrcoef(analog_vars['signal_I_ex'], analog_vars['signal_I_in'])
                    results['EI_CC'].append(np.unique(cc[cc != 1.]))
            results['EI_CC'] = np.array(list(itertools.chain(*results['EI_CC'])))
            # remove nans and infs
            results['EI_CC'] = np.extract(np.logical_not(np.isnan(results['EI_CC'])), results['EI_CC'])
            results['EI_CC'] = np.extract(np.logical_not(np.isinf(results['EI_CC'])), results['EI_CC'])

        if 'V_m' in variable_names and plot:
            results['single_Vm'] = analog_vars['V_m'].analog_signals[single_idx].signal
            results['single_idx'] = single_idx
            results['time_axis'] = analog_vars['V_m'].analog_signals[single_idx].time_axis()
            variable_names.remove('V_m')

            for idxx, nnn in enumerate(variable_names):
                cond = analog_vars[nnn].analog_signals[single_idx].signal

                if 'I_ex' in variable_names:
                    results['I_{0}'.format(nnn[-2:])] = cond
                    results['I_{0}'.format(nnn[-2:])] /= 1000.
                elif 'g_ex' in variable_names:
                    rev = reversals[idxx]
                    results['I_{0}'.format(nnn[-2:])] = cond * (results['single_Vm'] - rev)
                    results['I_{0}'.format(nnn[-2:])] /= 1000.
                else:
                    results['single_{0}'.format(nnn)] = cond

        return results


def compute_dimensionality(response_matrix, pca_obj=None, label='', plot=False, display=True, save=False):
    """
    Measure the effective dimensionality of population responses. Based on Abbott et al. (2001). Interactions between
    intrinsic and stimulus-evoked activity in recurrent neural networks.

    :param response_matrix: matrix of continuous responses to analyze (NxT)
    :param pca_obj: if pre-computed, otherwise None
    :param label:
    :param plot:
    :param display:
    :param save:
    :return: (float) dimensionality
    """
    assert (check_dependency('sklearn')), "PCA analysis requires scikit learn"
    if display:
        logger.info("Determining effective dimensionality...")
        t_start = time.time()
    if pca_obj is None:
        n_features, n_samples = np.shape(response_matrix)  # features = neurons
        if n_features > n_samples:
            logger.warning('WARNING - PCA n_features ({}) > n_samples ({}). Effective dimensionality will be computed '
                           'using {} components!'.format(n_features, n_samples, min(n_samples, n_features)))
        pca_obj = sk.PCA(n_components=min(n_features, n_samples))
    if not hasattr(pca_obj, "explained_variance_ratio_"):
        pca_obj.fit(response_matrix.T)  # we need to transpose here as scipy requires n_samples X n_features
    # compute dimensionality
    dimensionality = 1. / np.sum((pca_obj.explained_variance_ratio_ ** 2))
    if display:
        logger.info("Effective dimensionality = {0}".format(str(round(dimensionality, 2))))
        logger.info("Elapsed Time: {0} s".format(str(round(time.time() - t_start, 3))))
    if plot:
        X = pca_obj.fit_transform(response_matrix.T).T
        tools.visualization.plotting.plot_dimensionality(dimensionality, pca_obj, X, data_label=label, display=display, save=save)
    return dimensionality


def compute_timescale(response_matrix, time_axis, max_lag=1000, method=0, plot=True, display=True, save=False,
                      verbose=True):
    """
    Determines the time scale of fluctuations in the population activity.

    :param response_matrix: [np.array] with size NxT, continuous time
    :param time_axis:
    :param max_lag:
    :param method: based on autocorrelation (0) or on power spectra (1) - not implemented yet
    :param plot:
    :param display:
    :param save:
    :return:
    """
    # TODO modify / review / extend / correct / update
    time_scales = []
    final_acc = []
    errors = []
    acc = cross_trial_cc(response_matrix)
    initial_guess = 1., 0., 10.
    for n_signal in range(acc.shape[0]):
        try:
            fit, _ = opt.leastsq(err_func, initial_guess, args=(time_axis[:max_lag], acc[n_signal, :max_lag], acc_function))

            if fit[2] > 0.1:
                error_rates = np.sum((acc[n_signal, :max_lag] - acc_function(time_axis[:max_lag], *fit)) ** 2)
                if verbose:
                    logger.info("Timescale [ACC] = {0} ms / error = {1}".format(str(fit[2]), str(error_rates)))
                time_scales.append(fit[2])
                errors.append(error_rates)
                final_acc.append(acc[n_signal, :max_lag])
            elif 0. < fit[2] < 0.1:
                fit, _ = opt.leastsq(err_func, initial_guess,
                                     args=(time_axis[1:max_lag], acc[n_signal, 1:max_lag], acc_function))
                error_rates = np.sum((acc[n_signal, :max_lag] - acc_function(time_axis[:max_lag], *fit)) ** 2)
                if verbose:
                    logger.info("Timescale [ACC] = {0} ms / error = {1}".format(str(fit[2]), str(error_rates)))
                time_scales.append(fit[2])
                errors.append(error_rates)
                final_acc.append(acc[n_signal, :max_lag])
        except:
            continue
    final_acc = np.array(final_acc)

    mean_fit, _ = opt.leastsq(err_func, initial_guess, args=(time_axis[:max_lag], np.mean(final_acc, 0), acc_function))

    if mean_fit[2] < 0.1:
        mean_fit, _ = opt.leastsq(err_func, initial_guess,
                                  args=(time_axis[1:max_lag], np.mean(final_acc, 0)[1:max_lag], acc_function))

    error_rates = np.sum((np.mean(final_acc, 0) - acc_function(time_axis[:max_lag], *mean_fit)) ** 2)
    logger.info("*******************************************")
    logger.info("Timescale = {0} ms / error = {1}".format(str(mean_fit[2]), str(error_rates)))
    logger.info("Accepted dimensions = {0}".format(str(float(final_acc.shape[0]) / float(acc.shape[0]))))

    if plot:
        tools.visualization.plotting.plot_acc(time_axis[:max_lag], acc[:, :max_lag], mean_fit, acc_function, ax=None,
                                              display=display, save=save)

    return final_acc, mean_fit, acc_function, time_scales


def dimensionality_reduction(state_matrix, data_label='', labels=None, metric=None, standardize=True, plot=True,
                             colormap='jet', display=True, save=False):
    """
    Fit and test various algorithms, to extract a reasonable 3D projection of the data for visualization

    :param state_matrix: matrix to analyze (NxT)
    :param data_label:
    :param labels:
    :param metric: [str] metric to use (if None all will be tested)
    :param standardize:
    :param plot:
    :param colormap:
    :param display:
    :param save:
    :return:
    """
    # TODO extend and test - and include in the analyse_activity_dynamics function
    metrics = ['PCA', 'FA', 'LLE', 'IsoMap', 'Spectral', 'MDS', 't-SNE']
    if metric is not None:
        assert (metric in metrics), "Incorrect metric"
        metrics = [metric]

    if labels is None:
        raise TypeError("Please provide stimulus labels")
    else:
        n_elements = np.unique(labels)
    colors_map = vis_helper.get_cmap(N=len(n_elements), cmap=colormap)

    for met in metrics:
        if met == 'PCA':
            logger.info("Dimensionality reduction with Principal Component Analysis")
            t_start = time.time()
            pca_obj = sk.PCA(n_components=3)
            X_r = pca_obj.fit(state_matrix.T).transform(state_matrix.T)
            if display:
                logger.info("Elapsed time: {0} s".format(str(time.time() - t_start)))
                logger.info("Explained Variance (first 3 components): %s" % str(pca_obj.explained_variance_ratio_))
            exp_var = [round(n, 2) for n in pca_obj.explained_variance_ratio_]

            if plot:
                fig1 = pl.figure()
                ax11 = fig1.add_subplot(111, projection='3d')
                ax11.set_xlabel(r'$PC_{1}$')
                ax11.set_ylabel(r'$PC_{2}$')
                ax11.set_zlabel(r'$PC_{3}$')
                fig1.suptitle(r'${0} - PCA (var = {1})$'.format(str(data_label), str(exp_var)))
                tools.visualization.plotting.scatter_projections(X_r, labels, colors_map, ax=ax11)
                if save:
                    fig1.savefig(save + data_label + '_PCA.pdf')
                if display:
                    pl.show()
        elif met == 'FA':
            logger.info("Dimensionality reduction with Factor Analysis")
            t_start = time.time()
            fa2 = sk.FactorAnalysis(n_components=len(n_elements))
            state_fa = fa2.fit_transform(state_matrix.T)
            score = fa2.score(state_matrix.T)
            if display:
                logger.info("Elapsed time: {0} s / Score (NLL): {1}".format(str(time.time() - t_start), str(score)))
            if plot:
                fig2 = pl.figure()
                fig2.suptitle(r'{0} - Factor Analysis'.format(str(data_label)))
                # print(state_fa[:3, :].shape)
                ax21 = fig2.add_subplot(111, projection='3d')
                tools.visualization.plotting.scatter_projections(state_fa[:3, :].T, labels, colors_map, ax=ax21)
                if save:
                    fig2.savefig(save + data_label + '_FA.pdf')
                if display:
                    pl.show()
        elif met == 'LLE':
            logger.info("Dimensionality reduction with Locally Linear Embedding")
            if plot:
                fig3 = pl.figure()
                fig3.suptitle(r'{0} - Locally Linear Embedding'.format(str(data_label)))

            methods = ['standard', 'ltsa', 'hessian', 'modified']
            labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

            for i, method in enumerate(methods):
                t_start = time.time()
                fit_obj = man.LocallyLinearEmbedding(n_neighbors=199, n_components=3, eigen_solver='auto',
                                                     method=method, n_jobs=-1)
                Y = fit_obj.fit_transform(state_matrix.T)
                if display:
                    logger.info(
                        "\t{0} - {1} s / Reconstruction error = {2}".format(method, str(time.time() - t_start), str(
                            fit_obj.reconstruction_error_)))
                if plot:
                    ax = fig3.add_subplot(2, 2, i + 1, projection='3d')
                    ax.set_title(method)
                    tools.visualization.plotting.scatter_projections(Y, labels, colors_map, ax=ax)
            if plot and save:
                fig3.savefig(save + data_label + '_LLE.pdf')
            if plot and display:
                pl.show(False)
        elif met == 'IsoMap':
            logger.info("Dimensionality reduction with IsoMap Embedding")
            t_start = time.time()
            iso_fit = man.Isomap(n_neighbors=199, n_components=3, eigen_solver='auto', path_method='auto',
                                 neighbors_algorithm='auto', n_jobs=-1)
            iso = iso_fit.fit_transform(state_matrix.T)
            score = iso_fit.reconstruction_error()
            if display:
                logger.info("Elapsed time: {0} s / Reconstruction error = {1}".format(str(time.time() - t_start),
                            str(score)))
            if plot:
                fig4 = pl.figure()
                fig4.suptitle(r'{0} - IsoMap Embedding'.format(str(data_label)))
                ax41 = fig4.add_subplot(111, projection='3d')
                tools.visualization.plotting.scatter_projections(iso, labels, colors_map, ax=ax41)
                if save:
                    fig4.savefig(save + data_label + '_IsoMap.pdf')
                if display:
                    pl.show(False)
        elif met == 'Spectral':
            logger.info("Dimensionality reduction with Spectral Embedding")
            fig5 = pl.figure()
            fig5.suptitle(r'{0} - Spectral Embedding'.format(str(data_label)))

            affinities = ['nearest_neighbors', 'rbf']
            for i, n in enumerate(affinities):
                t_start = time.time()
                spec_fit = man.SpectralEmbedding(n_components=3, affinity=n, n_jobs=-1)
                spec = spec_fit.fit_transform(state_matrix.T)
                if display:
                    logger.info("Elapsed time: {0} s".format(str(time.time() - t_start)))
                if plot:
                    ax = fig5.add_subplot(1, 2, i + 1, projection='3d')
                    # ax.set_title(n)
                    tools.visualization.plotting.scatter_projections(spec, labels, colors_map, ax=ax)
                # pl.imshow(spec_fit.affinity_matrix_)
                if plot and save:
                    fig5.savefig(save + data_label + '_SE.pdf')
                if plot and display:
                    pl.show(False)
        elif met == 'MDS':
            logger.info("Dimensionality reduction with MultiDimensional Scaling")
            t_start = time.time()
            mds = man.MDS(n_components=3, n_jobs=-1)
            mds_fit = mds.fit_transform(state_matrix.T)
            if display:
                logger.info("Elapsed time: {0} s".format(str(time.time() - t_start)))
            if plot:
                fig6 = pl.figure()
                fig6.suptitle(r'{0} - MultiDimensional Scaling'.format(str(data_label)))
                ax61 = fig6.add_subplot(111, projection='3d')
                tools.visualization.plotting.scatter_projections(mds_fit, labels, colors_map, ax=ax61)
                if save:
                    fig6.savefig(save + data_label + '_MDS.pdf')
        elif met == 't-SNE':
            logger.info("Dimensionality reduction with t-SNE")
            t_start = time.time()
            tsne = man.TSNE(n_components=3, init='pca')
            tsne_emb = tsne.fit_transform(state_matrix.T)
            if display:
                logger.info("Elapsed time: {0} s".format(str(time.time() - t_start)))
            if plot:
                fig7 = pl.figure()
                fig7.suptitle(r'{0} - t-SNE'.format(str(data_label)))
                ax71 = fig7.add_subplot(111, projection='3d')
                tools.visualization.plotting.scatter_projections(tsne_emb, labels, colors_map, ax=ax71)
                if save:
                    fig7.savefig(save + data_label + '_t_SNE.pdf')
                if display:
                    pl.show(False)
        else:
            raise NotImplementedError("Metric {0} is not currently implemented".format(met))


def characterize_population_activity(population_object, parameter_set, analysis_interval, epochs=None, plot=True, display=True,
                                     save=False, color_map="coolwarm", color_subpop=False,
                                     analysis_pars=None):
    """
    Compute all the relevant metrics of recorded activity (spiking and analog signals), providing
    a thorough characterization and quantification of population dynamics

    :return results: dict
    :param population_object: Population or Network object whose activity should be analyzed
    :param parameter_set: complete ParameterSet
    :param prng: numpy.random object for precise experiment reproduction
    :param epochs:
    :param plot:
    :param display:
    :param save:
    :param color_map:
    :param color_subpop:
    :param analysis_pars:
    :return:
    """
    if analysis_pars is None:
        raise ValueError("Analysis parameters are required for characterizing population activity!")

    ap = analysis_pars
    pars_activity = ap.population_activity
    subpop_names = None

    if isinstance(population_object, network_architect.Population):
        gids = None
        base_population_object = None
    elif isinstance(population_object, network_architect.Network):
        merged_pop_name = ''.join(population_object.populations.keys())
        if not merged_pop_name in population_object.merged_populations.keys():
            new_population = network_architect.merge_subpopulations(
                sub_populations=list(population_object.populations.values()), name=merged_pop_name)
            merged_activity = network_architect.merge_population_activity({merged_pop_name: new_population},
                                                                          #population_object.merged_populations,
                                                                          start=analysis_interval[0],
                                                                          stop=analysis_interval[1],
                                                                          in_place=False)
            new_population.spiking_activity = merged_activity[merged_pop_name]['spiking_activity']
            new_population.analog_activity = merged_activity[merged_pop_name]['analog_activity']
        else:
            new_population = population_object.merged_populations[merged_pop_name]
            _ = network_architect.merge_population_activity(population_object.merged_populations,
                                                                          start=analysis_interval[0],
                                                                          stop=analysis_interval[1],
                                                                          in_place=True)

        gids = [n.id_list for n in list(tools.utils.operations.iterate_obj_list(population_object.spiking_activity))]
        subpop_names = population_object.population_names

        if not gids:
            gids = [np.array(n.gids) for n in list(tools.utils.operations.iterate_obj_list(population_object.populations))]

        base_population_object = population_object
        population_object = new_population
    else:
        raise TypeError("Incorrect population object. Must be Population or Network object")

    results = {'spiking_activity': {}, 'analog_activity': {}, 'metadata': {'population_name': population_object.name}}

    ########################################################################################################
    # Spiking activity analysis
    if population_object.spiking_activity:
        spike_list = population_object.spiking_activity
        assert isinstance(spike_list, signals.spikes.SpikeList), "Spiking activity should be SpikeList object"
        spike_list = spike_list.time_slice(analysis_interval[0], analysis_interval[1])

        results['spiking_activity'].update(compute_spikelist_metrics(spike_list, population_object.name, ap))

        if plot and ap.depth % 2 == 0:  # save all data
            tools.visualization.plotting.plot_isi_data(results['spiking_activity'][population_object.name],
                                                       data_label=population_object.name, color_map=color_map, location=0,
                                                       display=display, save=save)
            if has_pyspike:
                tools.visualization.plotting.plot_synchrony_measures(results['spiking_activity'][population_object.name],
                                                                     label=population_object.name, time_resolved=pars_activity.time_resolved,
                                                                     epochs=epochs, display=display, save=save)

        if pars_activity.time_resolved:
            # *** Averaged time-resolved metrics
            results['spiking_activity'][population_object.name].update(
                compute_time_resolved_statistics(spike_list, label=population_object.name,
                                                 time_bin=pars_activity.time_bin, epochs=epochs,
                                                 window_len=pars_activity.window_len, color_map=color_map,
                                                 display=display, plot=plot, save=save))
        if plot:
            results['metadata']['spike_list'] = spike_list

        if color_subpop and subpop_names:
            results['metadata'].update({'sub_population_names': subpop_names, 'sub_population_gids': gids,
                                        'spike_data_file': ''})

        if gids and ap.depth >= 3:
            if len(gids) == 2:
                locations = [-1, 1]
            else:
                locations = [0 for _ in range(len(gids))]

            for indice, name in enumerate(subpop_names):
                if base_population_object.spiking_activity[indice]:
                    sl = base_population_object.spiking_activity[indice]
                    results['spiking_activity'].update(compute_spikelist_metrics(sl, name, ap))

                if plot and ap.depth % 2 == 0:  # save all data
                    tools.visualization.plotting.plot_isi_data(results['spiking_activity'][name], data_label=name, color_map=color_map,
                                                               location=locations[indice], display=display, save=save)
                    if has_pyspike:
                        tools.visualization.plotting.plot_synchrony_measures(results['spiking_activity'][name], label=name,
                                                                             time_resolved=pars_activity.time_resolved,
                                                                             display=display, save=save)
                if pars_activity.time_resolved:
                    # *** Averaged time-resolved metrics
                    results['spiking_activity'][name].update(compute_time_resolved_statistics(spike_list,
                                                                                              label=population_object.name,
                                                                                              time_bin=pars_activity.time_bin,
                                                                                              epochs=epochs,
                                                                                              color_map=color_map,
                                                                                              display=display,
                                                                                              plot=plot, save=save,
                                                                                              window_len=pars_activity.window_len))
    else:
        logger.warning("Warning, the network is not spiking or no spike recording devices were attached.")

    # Analog activity analysis
    if population_object.analog_activity and base_population_object is not None:
        results['analog_activity'] = {}
        for pop_n, pop in enumerate(base_population_object.populations.values()):
            if bool(pop.analog_activity):
                results['analog_activity'].update({pop.name: {}})
                pop_idx = parameter_set.net_pars.pop_names.index(pop.name)
                if parameter_set.net_pars.analog_device_pars[pop_idx] is None:
                    break
                variable_names = list(np.copy(parameter_set.net_pars.analog_device_pars[pop_idx]['record_from']))

                results['analog_activity'][pop.name].update(compute_analog_stats(pop, parameter_set, variable_names,
                                                                                 analysis_interval, plot))
    if plot:
        tools.visualization.plotting.plot_state_analysis(parameter_set, results, summary_only=bool(ap.depth % 2 != 0),
                                                         start=analysis_interval[0], stop=analysis_interval[1],
                                                         display=display, save=save)
    return results


def compute_time_resolved_statistics(spike_list, label='', time_bin=1., window_len=100, epochs=None,
                                     color_map='colorwarm', display=True, plot=False, save=False):
    """

    :param spike_list:
    :param label:
    :param time_bin:
    :param window_len:
    :param epochs:
    :param color_map:
    :param display:
    :param plot:
    :param save:
    :return:
    """
    time_axis = spike_list.time_axis(time_bin=time_bin)
    steps = len(list(tools.utils.operations.moving_window(time_axis, window_len)))
    mw = tools.utils.operations.moving_window(time_axis, window_len)
    results = dict()
    logger.info("\nAnalysing activity in moving window..")

    for n in range(steps):
        if display:
            vis_helper.progress_bar(float(float(n) / steps))
        time_window = next(mw)
        local_list = spike_list.time_slice(t_start=min(time_window), t_stop=max(time_window))
        local_isi = compute_isi_stats(local_list, summary_only=True, display=False)
        local_spikes = compute_spike_stats(local_list, time_bin=time_bin, summary_only=True, display=False)

        if n == 0:
            rr = {k + '_profile': [] for k in list(local_isi.keys())}
            rr.update({k + '_profile': [] for k in list(local_spikes.keys())})
        else:
            for k in list(local_isi.keys()):
                rr[k + '_profile'].append(local_isi[k])
                if n == steps - 1:
                    results.update({k + '_profile': rr[k + '_profile']})
            for k in list(local_spikes.keys()):
                rr[k + '_profile'].append(local_spikes[k])
                if n == steps - 1:
                    results.update({k + '_profile': rr[k + '_profile']})
    if plot:
        tools.visualization.plotting.plot_averaged_time_resolved(results, spike_list, label=label, epochs=epochs,
                                                                 color_map=color_map, display=display, save=save)

    return results